# ----------------------------------------------------------------------------
from dataclasses import asdict, dataclass

from areal.api.cli_args import AgentRLConfig, load_expr_config

args = ["--config", "ASearcher/configs/asearcher_local.yaml"]
config, _ = load_expr_config(args, AgentRLConfig)
config: AgentRLConfig

from areal.utils.network import find_free_ports

SGLANG_PORT, MASTER_PORT = 11451, 14514

SGLANG_HOST = "127.0.0.1"

# ----------------------------------------------------------------------------
# Environment variables used by inference/train engines
import os
import subprocess
import sys

os.environ["AREAL_LLM_SERVER_ADDRS"] = f"{SGLANG_HOST}:{SGLANG_PORT}"
os.environ["MASTER_ADDR"] = "127.0.0.1"
os.environ["MASTER_PORT"] = str(MASTER_PORT)
os.environ["RANK"] = str(0)
os.environ["WORLD_SIZE"] = str(1)
os.environ["TOKENIZERS_PARALLELISM"] = "true"
os.environ["LOCAL_RANK"] = str(0)

# ----------------------------------------------------------------------------
# 启动sglang server
from areal.api.cli_args import SGLangConfig
from areal.utils.network import find_free_ports

config.sglang.log_level = "info"
config.sglang.decode_log_interval = 10
sglang_cmd = SGLangConfig.build_cmd(
    config.sglang,
    tp_size=1,
    base_gpu_id=1,
    host=SGLANG_HOST,
    port=SGLANG_PORT,
)
sglang_process = subprocess.Popen(
    sglang_cmd,
    shell=True,
    stdout=sys.stdout,
    stderr=sys.stderr,
)

print("sglang process is launched")

# ----------------------------------------------------------------------------

# load search dataset
from datasets import load_dataset

print("dataset is at {}".format(config.train_dataset.path))
dataset = load_dataset(
        path="json",
        split="train",
        data_files=config.train_dataset.path,
    )
print(f">>> dataset column names: {dataset.column_names}")
print(f">>> example data: {dataset[0]}")

# ----------------------------------------------------------------------------

import asyncio
import os
import sys
import uuid
import json
import time
import torch
import torch.distributed as dist
import numpy as np
from datasets import load_dataset
from datasets.distributed import split_dataset_by_node
from tensordict import TensorDict
from transformers import PreTrainedTokenizerFast, AutoTokenizer

from areal.api.cli_args import (
    GenerationHyperparameters,
    load_expr_config,
)
from areal.api.io_struct import (
    FinetuneSpec,
    LLMRequest,
    WeightUpdateMeta,
)
from areal.api.engine_api import InferenceEngine
from areal.engine.ppo.actor import FSDPPPOActor
from areal.engine.sglang_remote import RemoteSGLangEngine
from areal.utils.data import concat_padded_tensors
from areal.api.io_struct import (
    AllocationMode,
    FinetuneSpec,
    LLMRequest,
    WeightUpdateMeta,
)

tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_path)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

# ----------------------------------------------------------------------------

# setup dataloader

from torchdata.stateful_dataloader import StatefulDataLoader
dataloader = StatefulDataLoader(
    dataset,
    batch_size=config.train_dataset.batch_size,
    shuffle=True,
    collate_fn=lambda x: x,
    drop_last=True,
)

from itertools import cycle

data_generator = cycle(dataloader)

ft_spec = FinetuneSpec(
    total_train_epochs=config.total_train_epochs,
    dataset_size=len(dataloader) * config.train_dataset.batch_size,
    train_batch_size=config.train_dataset.batch_size,
)

batch = next(data_generator)
print(f">>> The type of a batch is: {type(batch)}\n")
print(f">>> Each piece of data has keys: {batch[0].keys()}\n")
print(f">>> Example input question: {batch[0]['question']}\n")

# ----------------------------------------------------------------------------
# setup tool

import asyncio
import aiohttp
import json

TOOL_SERVER_ADDR = "10.11.16.175:5001"

async def call_search_tool(**req_meta):
    async with aiohttp.ClientSession() as session:
        async with session.post(
            f"http://{TOOL_SERVER_ADDR}/retrieve", json=req_meta, timeout=aiohttp.ClientTimeout(total=120, sock_connect=120)
        ) as response:
            response.raise_for_status()
            res = await response.json()
            return res["result"]

result = asyncio.run(call_search_tool(queries=["China"], topk=5, return_scores=False))[0]
print(json.dumps(result, indent=4))

# -----------------------------------------------------------------------------

# parse tool calling

import re

def parse_search_query(text):
    pattern = r"<search>(.*?)</search>"
    matches = re.findall(pattern, text, re.DOTALL)
    if matches:
        return matches[-1].strip()
    return None

test_tool_str = "I would like to search for AI. <search> Artificial Intelligence </search>"
print(">>> input: ", test_tool_str)
print(">>> search query: ", parse_search_query(test_tool_str))


# -----------------------------------------------------------------------------

# parse answer

def parse_answer(text):
    pattern = r"<answer>(.*?)</answer>"
    matches = re.findall(pattern, text, re.DOTALL)
    if matches:
        return matches[-1].strip()
    return None

test_answer_str = "The answer can be concluded <answer> Artificial Intelligence </answer>"
print(">>> input: ", test_answer_str)
print(">>> answer: ", parse_answer(test_answer_str))

# -----------------------------------------------------------------------------

# F1 reward

def f1_score(pred_ans, gt):
    # 预处理文本（此处为简化版本）
    pred_ans = pred_ans.strip().lower()
    gt = gt.strip().lower()
    
    pred_tokens = set(pred_ans.split())
    gt_tokens = set(gt.split())
    
    if not gt_tokens or not pred_tokens:
        return 0
    
    # 计算共同的词数
    common_tokens = pred_tokens & gt_tokens
    
    # 计算精确率和召回率
    precision = len(common_tokens) / len(pred_tokens) if pred_tokens else 0
    recall = len(common_tokens) / len(gt_tokens) if gt_tokens else 0
    
    # 计算F1分数
    f1 = 0
    if precision + recall > 0:
        f1 = 2 * (precision * recall) / (precision + recall)
    
    return f1

print("f1_score('James Bond', 'James Bond'): {:.2f}".format(f1_score('James Bond', 'James Bond')))
print("f1_score('James Smith', 'James Bond'): {:.2f}".format(f1_score('James Smith', 'James Bond')))


# ----------------------------------------------------------------------------

SEARCH_ONLY_PROMPT_TEMPLATE="""A conversation between User and Assistant. The user asks a question, and the Assistant answers it. The Assistant analyzes the given question and information in the mind, retains important relevant information, calls a search engine to find necessary information, accesses web pages with certain urls, and provides the user with the answer. The Assistant conducts search by <search> query </search> and the top search results will be returned between <information> and </information>. The reasoning processes are enclosed within <think> </think>. Finally, the Assistant provides answer inside <answer> and </answer>, i.e. <answer> answer here </answer>. If there are multiple queries, ensure all answers are enclosed within <answer> </answer>, seperated with comma. 

User: 
{question}

The language of your answer should align with the question.

Assistant:
<think>"""

batch = next(data_generator)
prompt = SEARCH_ONLY_PROMPT_TEMPLATE.format(question=batch[0]["question"])

print(f">>> PROMPT: {prompt}")

# ---------------------------------------------------------------------------

# test LLM input/ouptut

# initialize inference engine
rollout_engine = RemoteSGLangEngine(config.rollout)
rollout_engine.initialize(None, None)

# generation config
gconfig = GenerationHyperparameters(max_new_tokens=256, stop=["</search>", "</answer>", "</access>"])

# tokenize the prompt
input_ids = tokenizer([prompt], add_special_tokens=False)["input_ids"][0]
req = LLMRequest(rid=uuid.uuid4().hex, input_ids=input_ids, gconfig=gconfig)

# generate rollout with inference engine
resp = asyncio.run(rollout_engine.agenerate(req))
completion_str = tokenizer.decode(resp.output_tokens)

# logging
print(f">>> prompt str: {tokenizer.decode(resp.input_tokens)}")
print(f">>> generated: {tokenizer.decode(resp.output_tokens)}")
print(f">>> search query: {parse_search_query(completion_str)}")


# ---------------------------------------------------------------------------

class SearchAgentWorkflow:
    def __init__(self, gconfig, tokenizer, max_tokens, max_turns, verbose):
        self.gconfig = gconfig
        self.tokenizer = tokenizer
        self.max_tokens = max_tokens
        self.max_turns = max_turns
        self.verbose = verbose
    
    async def arun_episode(self, engine: InferenceEngine, data):
        prompt = SEARCH_ONLY_PROMPT_TEMPLATE.format(question=data["question"])
        # an unique trajectory rid to ensure all requests goes to the same sglang server
        rid = uuid.uuid4().hex

        input_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        logprobs = [0.0] * len(input_ids)
        loss_mask = [0] * len(input_ids)
        texts = [prompt]

        answer, reward = None, 0
        
        num_turns = 0
        while num_turns < self.max_turns and len(input_ids) < self.max_tokens:
            num_turns += 1

            req = LLMRequest(
                rid=rid,
                input_ids=input_ids,
                gconfig=self.gconfig.new(n_samples=1),
            )
            resp = await engine.agenerate(req)
            completion_str = self.tokenizer.decode(resp.output_tokens)

            input_ids += resp.output_tokens
            input_ids += resp.output_tokens
            logprobs +=  resp.output_logprobs
            loss_mask += [1] * resp.output_len
            texts.append(completion_str)

            # parse search query & trigger tool call
            search_query = parse_search_query(completion_str)
            if search_query:
                search_results = (await call_search_tool(queries=[search_query], topk=3, return_scores=False))[0]
                search_results_str = "\n\n<information>\n" + "\n\n".join(['<p title="{}">\n{}\n</p>'.format(r["wikipedia_title"], r["contents"]) for r in search_results]) + "\n</information>"

                search_token_ids = self.tokenizer.encode(search_results_str, add_special_tokens=False)
                input_ids += search_token_ids
                logprobs += [0.0] * len(search_token_ids)
                loss_mask += [0] * len(search_token_ids)

                texts.append(search_results_str)
            
            # parse answer
            answer = parse_answer(completion_str)
            if answer:
                reward = max([f1_score(answer, gt) for gt in data["answer"]])
                break
                
            if input_ids[-1] in [self.tokenizer.pad_token_id, self.tokenizer.eos_token_id]:
                break
        
        res = dict(
            input_ids=torch.tensor(input_ids),
            logprobs=torch.tensor(logprobs),
            loss_mask=torch.tensor(loss_mask),
            rewards=torch.tensor(float(reward)),
            attention_mask=torch.ones(len(input_ids), dtype=torch.bool),
        )
        res = {k: v.unsqueeze(0) for k, v in res.items()}
        return TensorDict(res, batch_size=[1])
    
# -----------------------------------------------------------------------------------------------------------------

# initialize inference engine
rollout = RemoteSGLangEngine(config.rollout)
rollout.initialize(None, None)
try:
    # TODO: create workflow
    workflow = SearchAgentWorkflow(
        gconfig=GenerationHyperparameters(max_new_tokens=256, stop=["</answer>", "</search>"]), 
        tokenizer=tokenizer,
        max_tokens=1024,
        max_turns=32,
        verbose=True
    )
    sample_data = next(data_generator)[:4]
    res = rollout.rollout_batch(sample_data, workflow=workflow)
    print(res)
finally:
    rollout.destroy()

# log the trajectories
traj_lens = res["attention_mask"].sum(dim=1).numpy().tolist()
for i in range(4):
    token_ids = res["input_ids"][i, :traj_lens[i]]
    print(f">>> Trajectory {i} >>>\n{tokenizer.decode(token_ids)}")

# -------------------------------------------------------------------------------------------------------------------

# Group generation for GRPO

class GroupedSearchAgentWorkflow:
    def __init__(self, gconfig, tokenizer, max_tokens, max_turns, group_size, verbose):
        self.gconfig = gconfig
        self.tokenizer = tokenizer
        self.max_tokens = max_tokens
        self.max_turns = max_turns
        self.group_size = group_size
        self.verbose = verbose
    
    async def arun_episode(self, engine, data):
        workflows = [
            SearchAgentWorkflow(
                self.gconfig.new(n_samples=1),
                self.tokenizer,
                self.max_tokens,
                self.max_turns,
                self.verbose,
            )
            for _ in range(self.group_size)
        ]
        tasks = [workflow.arun_episode(engine, data) for workflow in workflows]
        results = await asyncio.gather(*tasks)
        return concat_padded_tensors(results)

# initialize inference engine
rollout = RemoteSGLangEngine(config.rollout)
rollout.initialize(None, None)
try:
    # TODO: create workflow
    workflow = GroupedSearchAgentWorkflow(
        gconfig=GenerationHyperparameters(max_new_tokens=256, stop=["</answer>", "</search>"]), 
        tokenizer=tokenizer,
        max_tokens=512,
        max_turns=32,
        group_size=4,
        verbose=True
    )
    sample_data = next(data_generator)[:2]
    res = rollout.rollout_batch(sample_data, workflow=workflow)
    print(res)
finally:
    rollout.destroy()


workflow = GroupedSearchAgentWorkflow(
        gconfig=GenerationHyperparameters(max_new_tokens=256, stop=["</answer>", "</search>"]), 
        tokenizer=tokenizer,
        max_tokens=1024,
        max_turns=32,
        group_size=4,
        verbose=True
    )
actor = FSDPPPOActor(config=config.actor)
actor.initialize(None, ft_spec)

rollout = RemoteSGLangEngine(config.rollout)
rollout.initialize(None, None)

weight_update_meta = WeightUpdateMeta.from_fsdp_nccl(
    AllocationMode.from_str("sglang.d1p1t1+d1p1t1"), actor
)

warmup_steps = 1
times = []
for global_step in range(5):
    if global_step >= warmup_steps:
        tik = time.perf_counter()
    batch = rollout.rollout_batch(next(data_generator), workflow=workflow)
    print(batch)
    batch = batch.to(actor.device)

    logp = actor.compute_logp(batch)
    batch["prox_logp"] = logp

    actor.compute_advantages(batch)

    stats = actor.ppo_update(batch)
    actor.step_lr_scheduler()

    rollout.pause()
    future = rollout.update_weights(weight_update_meta)
    actor.upload_weights(weight_update_meta)
    future.result()
    torch.cuda.synchronize()
    rollout.resume()

    actor.set_version(global_step + 1)
    rollout.set_version(global_step + 1)
    if global_step >= warmup_steps:
        times.append(time.perf_counter() - tik)
print(times)