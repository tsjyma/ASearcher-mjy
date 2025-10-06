import asyncio
import json
import logging
import os
from typing import Dict

from openai import AsyncOpenAI
from transformers import AutoTokenizer
from agent.asearcherweaver import AsearcherWeaverAgent
from evaluation.search_eval_async import process_single_access_query, process_single_llm_query, process_single_search_query
from tools.search_utils import make_search_client

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AsyncVLLMClient:
    """Async vLLM client using OpenAI compatible API"""
    def __init__(self, llm_url: str, model_name: str = "default", api_key: str = "EMPTY"):
        self.llm_url = llm_url.rstrip('/')
        self.model_name = model_name
        self.api_key = api_key
        
        self.client = AsyncOpenAI(
            base_url=f"{self.llm_url}/v1/",
            api_key=self.api_key,
        )
        logger.info(f"Initialized AsyncOpenAI client: {self.llm_url}")
    
    async def async_generate(self, prompt: str, sampling_kwargs: Dict) -> Dict:
        """Generate text asynchronously"""
        if isinstance(prompt, list) and len(prompt) > 0 and isinstance(prompt[0], dict) and "role" in prompt[0]:
            # Chat format
            completion_kwargs = {
                "model": self.model_name,
                "messages": prompt,
                "max_tokens": sampling_kwargs.get("max_new_tokens", 30000),
                "temperature": sampling_kwargs.get("temperature", 0.0),
                "top_p": sampling_kwargs.get("top_p", 1.0),
                "stream": False,
            }
            stop_sequences = sampling_kwargs.get("stop", [])
            if stop_sequences:
                completion_kwargs["stop"] = stop_sequences
            # logger.info(f"Calling vLLM API: {completion_kwargs}")
            logger.info(f"Calling vLLM Chat API with model {self.model_name}")
            response = await self.client.chat.completions.create(**completion_kwargs)
            if response.choices and len(response.choices) > 0:
                choice = response.choices[0]
                return {"text": choice.message.content, "finish_reason": choice.finish_reason}
            else:
                return {"text": "", "finish_reason": "unknown"}
        else:
            messages = [{"role": "user", "content": prompt}]
            completion_kwargs = {
                "model": self.model_name,
                "messages": messages,
                "max_tokens": sampling_kwargs.get("max_new_tokens", 30000),
                "temperature": sampling_kwargs.get("temperature", 0.0),
                "top_p": sampling_kwargs.get("top_p", 1.0),
                "stream": False,
            }
            stop_sequences = sampling_kwargs.get("stop", [])
            if stop_sequences:
                completion_kwargs["stop"] = stop_sequences
            # logger.info(f"Calling vLLM API: {completion_kwargs}")
            logger.info(f"Calling vLLM Chat API with model {self.model_name}")
            response = await self.client.chat.completions.create(**completion_kwargs)
            if response.choices and len(response.choices) > 0:
                choice = response.choices[0]
                return {"text": choice.message.content, "finish_reason": choice.finish_reason}
            else:
                return {"text": "", "finish_reason": "unknown"}
            # Completion format
            # completion_kwargs = {
            #     "model": self.model_name,
            #     "prompt": prompt,
            #     "max_tokens": sampling_kwargs.get("max_new_tokens", 30000),
            #     "temperature": sampling_kwargs.get("temperature", 0.0),
            #     "top_p": sampling_kwargs.get("top_p", 1.0),
            #     "stream": False,
            # }
            # stop_sequences = sampling_kwargs.get("stop", [])
            # if stop_sequences:
            #     completion_kwargs["stop"] = stop_sequences
            # #logger.info(f"Calling vLLM API: {completion_kwargs}")
            # logger.info(f"Calling vLLM Chat API with model {self.model_name}")
            # response = await self.client.completions.create(**completion_kwargs)
            # if response.choices and len(response.choices) > 0:
            #     choice = response.choices[0]
            #     return {"text": choice.text, "finish_reason": choice.finish_reason}
            # else:
            #     return {"text": "", "finish_reason": "unknown"}

def convert_agent_tool_calls_to_dict(agent_tool_calls):
    """Convert agent tool calls to dict format"""
    import re
    
    dict_tool_calls = []
    
    for tool_call_str in agent_tool_calls:
        # Parse <search>...</search>
        search_match = re.search(r'<search>(.*?)</search>', tool_call_str, re.DOTALL)
        if search_match:
            dict_tool_calls.append({"type": "search", "query": search_match.group(1).strip()})
            continue
            
        # Parse <access>...</access>
        access_match = re.search(r'<access>(.*?)</access>', tool_call_str, re.DOTALL)
        if access_match:
            dict_tool_calls.append({"type": "access", "content": access_match.group(1).strip()})
            continue
            
        # Parse <write_outline>
        outline_match = re.search(r'<write_outline>', tool_call_str, re.DOTALL)
        if outline_match:
            dict_tool_calls.append({"type": "write_outline", "content": ""})
            continue

        # Parse <terminate>
        terminate_match = re.search(r'<terminate>', tool_call_str, re.DOTALL)
        if terminate_match:
            dict_tool_calls.append({"type": "terminate", "content": ""})
            continue

        # Parse <retrieve>...</retrieve>
        retrieve_match = re.search(r'<retrieve>(.*?)</retrieve>', tool_call_str, re.DOTALL)
        if retrieve_match:
            dict_tool_calls.append({"type": "retrieve", "content": retrieve_match.group(1).strip()})
            continue

        # Parse <write_terminate>
        write_terminate_match = re.search(r'<write_terminate>', tool_call_str, re.DOTALL)
        if write_terminate_match:
            dict_tool_calls.append({"type": "write_terminate", "content": ""})
            continue

        # Parse <answer>...</answer>
        answer_match = re.search(r'<answer>(.*?)</answer>', tool_call_str, re.DOTALL)
        if answer_match:
            dict_tool_calls.append({"type": "answer", "content": answer_match.group(1).strip()})
            continue

    return dict_tool_calls

def parse_retrieved_ids(content):
    import re
    ids = re.findall(r'<id>(\d+)</id>', content, re.DOTALL)
    ids = [int(id_.strip()) for id_ in ids]
    return ids


async def process_single_work_item(semaphore, agent_type, llm, tokenizer, search_client, args, out_dir, process):
    """Process a single work item using agent v2"""
    import re
    async with semaphore:
        if "history" not in process:
            process["history"] = []
            process["running"] = True
            process["num_turns"] = 0
            process["outline"] = ""

        # Create fresh agent instance for thread safety
        agent = AsearcherWeaverAgent()
        agent.initialize_with_prompt(process)
        
        # Set tokenizer for V1 agents that need it
        if hasattr(agent, 'set_tokenizer'):
            agent.set_tokenizer(tokenizer)
        
        # The loop
        while process["running"] and agent.num_turns < agent.max_turns:
            # Check if agent is finished
            if agent.is_finished:
                process["running"] = False
                break
            
            try:
                # Get LLM query from agent
                prompt_or_messages, sampling_params = agent.prepare_llm_query()

                print(f"Process {process['id']} Turn {agent.num_turns+1} LLM Query:\n{prompt_or_messages if isinstance(prompt_or_messages, str) else prompt_or_messages[-1]['content']}\n{'-'*50}")

                # assert not agent.is_finished

                if isinstance(prompt_or_messages, str):
                    prompt = prompt_or_messages
                
                    # Process LLM query
                    llm_response = await process_single_llm_query(llm, tokenizer, prompt, sampling_params, args, qid=process["id"])
                    completion_text = llm_response.text

                elif isinstance(prompt_or_messages, list):
                    messages = prompt_or_messages
                
                    # Process LLM query
                    llm_response = await process_single_llm_query(llm, tokenizer, messages, sampling_params, args, qid=process["id"])
                    completion_text = llm_response.text

                print(f"Process {process['id']} Turn {agent.num_turns+1} LLM Response:\n{completion_text[-300:]}\n{'-'*50}")

                # Let agent consume LLM response and get tool calls
                tool_calls_raw = agent.consume_llm_response(llm_response, completion_text)
                # print(tool_calls_raw)

                tool_calls = convert_agent_tool_calls_to_dict(tool_calls_raw)

                print(tool_calls)
                # print("agent.isfinished:", agent.is_finished)
                # assert not agent.is_finished
                
                # Log progress
                if tool_calls:
                    print(f"Process {process['id']}: {', '.join([tc['type'] for tc in tool_calls])}")
                
                # Add to history in unified agent v2 format
                process["history"].append({
                    "type": "llm_response", 
                    "text": completion_text,
                    "tool_calls": tool_calls
                })
                
                # Process each tool call
                for tool_call in tool_calls:
                    print("Tool call:", tool_call)
                    if tool_call["type"] == "search":
                        search_result = await process_single_search_query(search_client, tool_call["query"])
                        if search_result:
                            # Handle different search result formats
                            if isinstance(search_result, dict):
                                documents = search_result.get("documents", []) or []
                                urls = search_result.get("urls", []) or []
                            elif isinstance(search_result, list):
                                documents = []
                                urls = []
                                for result in search_result:
                                    if isinstance(result, dict):
                                        result_docs = result.get("documents", []) or []
                                        result_urls = result.get("urls", []) or []
                                        documents.extend(result_docs)
                                        urls.extend(result_urls)
                            else:
                                documents = []
                                urls = []
                            
                            # Ensure we don't pass None values
                            documents = documents or []
                            urls = urls or []
                            
                            # Provide search result to agent in its expected format
                            tool_response = {
                                "type": "search",
                                "documents": documents,
                                "urls": urls
                            }
                            agent.consume_tool_response(tool_response)
                            
                            # Add to unified history in agent v2 format (regardless of agent's internal format)
                            process["history"].append({
                                "type": "search_result",
                                "query": tool_call["query"],
                                "documents": documents,
                                "urls": urls
                            })
                    
                    elif tool_call["type"] == "access":
                        
                        # get the goal if provided
                        goal = re.search(r'(.*?)<goal>(.*?)</goal>', tool_call.get("content", ""), re.DOTALL)
                        
                        if goal:
                            url = goal.group(1).strip()
                            goal = goal.group(2).strip()
                            print("Parsed URL:", url)
                            print("Parsed Goal:", goal)
                        else:
                            url = tool_call.get("content", "").strip()
                            goal = "Find relevant information to answer the question."
                        # Access the URL
                        access_result = await process_single_access_query(search_client, url)
                        if isinstance(access_result, list):
                            access_result = access_result[0] if access_result else ""
                        print("Access successfully!")
                        if access_result:
                            if isinstance(access_result, dict):
                                page = access_result.get("page", "") or ""
                            elif isinstance(access_result, str):
                                page = access_result or ""
                            else:
                                page = str(access_result) if access_result else ""
                            
                            # Ensure we don't pass None values
                            page = page or ""
                            # print("check page:", page[:200])                            
                            # Provide page access result to agent in its expected format
                            tool_response = {
                                "type": "access",
                                "page": page,
                                "goal": goal
                            }
                            agent.consume_tool_response(tool_response)
                            # print("check tool_response:")
                            # Add to unified history in agent v2 format (regardless of agent's internal format)
                            process["history"].append({
                                "type": "page_access",
                                "url": url,
                                "page": page,
                                "goal": goal
                            })

                    elif tool_call["type"] == "write_outline":
                        # Agent is requesting to write an outline
                        agent.consume_tool_response({"type": "outline"})

                        process["history"].append({
                            "type": "write_outline"
                        })

                    elif tool_call["type"] == "retrieve":
                        # retrieve from memory bank
                        content = tool_call["content"]
                        ids = parse_retrieved_ids(content)
                        # get the goal if provided
                        goal_match = re.search(r'<goal>(.*?)</goal>', content, re.DOTALL)
                        goal = goal_match.group(1).strip()
                        agent.write_goal = goal  # update agent's write goal
                        retrieved_summaries = [agent.memory_bank.get(id_, "") for id_ in ids if id_ in agent.memory_bank]
                        print(f"Retrieved {len(retrieved_summaries)} items from memory bank.")
                        retrieve_response ={
                            "type": "retrieve",
                            "ids": ids,
                            "summaries": retrieved_summaries,
                            "goal": goal
                        }
                        agent.consume_tool_response(retrieve_response)

                        # Add to unified history in agent v2 format (regardless of agent's internal format)
                        process["history"].append({
                                "type": "retrieve_response",
                                "ids": ids,
                                "goal": goal,
                        })

                    elif tool_call["type"] == "terminate":
                        # Agent has decided to terminate
                        # store the outline if available
                        process["outline"] = agent.outline if hasattr(agent, 'outline') else ""
                        writer_activate = {
                            "type": "write"
                        }
                        agent.consume_tool_response(writer_activate)
                        process["history"].append({
                            "type": "write_activate"
                        })
                        break

                    elif tool_call["type"] == "write_terminate":
                        # Agent has decided to terminate writing
                        process["report"] = agent.report if hasattr(agent, 'report') else ""
                        agent_answer = {
                            "type": "answer",
                        }
                        agent.consume_tool_response(agent_answer)
                        process["history"].append({
                            "type": "answer_activate"
                        })
                        break
                    
                    elif tool_call["type"] == "answer":
                        # Agent has provided final answer
                        process["pred_answer"] = tool_call["content"]
                        process["running"] = False
                        # save final state
                        with open(os.path.join(out_dir, f"{process['id']}.json"), "w") as f:
                            # Include agent memory for debugging
                            process_copy = process.copy()
                            if hasattr(agent, "current_process"):
                                process_copy = agent.current_process.copy()
                            if hasattr(agent, 'memory') and agent.memory:
                                process_copy["agent_memory"] = agent.memory.to_dict()
                                process_copy["agent_stats"] = agent.memory.logging_stats()
                            json.dump(process_copy, f, ensure_ascii=False)
                        break
                
                process["num_turns"] = agent.num_turns
                print(f"Process {process['id']} completed turn {agent.num_turns}.")
                # print("agent.isfinished:", agent.is_finished)
                assert not agent.is_finished
                
            except Exception as e:
                print(f"Error processing work item {process['id']}: {e}")
                process["running"] = False
                process["error"] = str(e)
                break

            # Save intermediate state
            with open(os.path.join(out_dir, f"{process['id']}.json"), "w") as f:
                # Include agent memory for debugging
                process_copy = process.copy()
                if hasattr(agent, "current_process"):
                    process_copy = agent.current_process.copy()
                if hasattr(agent, 'memory') and agent.memory:
                    process_copy["agent_memory"] = agent.memory.to_dict()
                    process_copy["agent_stats"] = agent.memory.logging_stats()
                json.dump(process_copy, f, ensure_ascii=False)

        
        # Ensure we have a final answer
        if "pred_answer" not in process and hasattr(agent, 'get_answer'):
            final_answer = agent.get_answer()
            if final_answer:
                process["pred_answer"] = final_answer
            else:
                process["pred_answer"] = ""
        
        return process
    
async def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="ASearcher Demo Service")
    parser.add_argument("--llm-url", default="http://0.0.0.0:50000", help="vLLM server URL")
    parser.add_argument("--model-name", default="ASearcher-Web-7B", help="Model name")
    parser.add_argument("--api-key", default="EMPTY", help="API key")
    parser.add_argument("--reload", action="store_true", help="Enable auto reload")
    parser.add_argument("--jina-api-key", default="EMPTY", help="Jina API key for web search")
    parser.add_argument("--temperature", type=float, default=0.1, help="LLM temperature")
    parser.add_argument("--top-p", type=float, default=0.7, help="LLM top-p")
    parser.add_argument("--top-k", type=int, default=1, help="LLM top-k")
    parser.add_argument("--max-tokens-per-call", type=int, default=1024, help="Max tokens per LLM call")
    parser.add_argument("--model_name_or_path", default="DeepSeek-V3.1", type=str)
  
    args = parser.parse_args()
    print(f"   vLLM server: {args.llm_url}")
    print(f"   Model name: {args.model_name}")
    print(f"   API key: {args.api_key}")
    print(f"   Jina API key: {args.jina_api_key}")

    if args.reload:
        os.environ.update({
            'ASEARCHER_LLM_URL': args.llm_url,
            'ASEARCHER_MODEL_NAME': args.model_name,
            'ASEARCHER_API_KEY': args.api_key,
            'JINA_API_KEY': args.jina_api_key,
        })
        if args.model_name != "default":
            os.environ['MODEL_PATH'] = args.model_name
    print("🔄 Auto reload mode enabled")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct", trust_remote_code=True)
    semaphore = asyncio.Semaphore(1)  # Limit to 1 concurrent request for simplicity

    prompt = \
"""
What animals that were mentioned in:
both Ilias Lagkouvardos's and Olga Tapia's papers on the alvei species of the genus named for Copenhagen outside the bibliographies and also in the 2021 article cited on the alvei species' Wikipedia page about a multicenter, randomized, double-blind study?
"""

    llm = AsyncVLLMClient(args.llm_url, args.model_name, args.api_key)
    search_client = make_search_client("async-web-search-access", True, args.jina_api_key)
    process = {
            "id": "demo",
            "prompt": prompt,
            "history": [],
            "running": True,
            "pred_answer": None
        }
    await process_single_work_item(semaphore, "asearcher-weaver", llm, tokenizer, search_client, args, ".", process)

if __name__ == "__main__":
    asyncio.run(main())