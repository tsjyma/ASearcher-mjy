# 项目概况
## ASearcher存在的问题
1. 摘要质量不高，例如未能准确引用相关链接；
2. 缺乏对历史网页和搜索结果的有效管理机制。

## 改进思路
为了解决这样的问题，我们提出了新的摘要机制**summary with goal**，新的摘要管理机制**memory bank**，新的引用与报告模式**planner and writer**，结合这三点改进与原有的ASearcher框架，我们提出了新的agent，**ASearcherWeaver**。

### ASearcherWeaver 工作流程
overview: start -> Planner Phase -> Writer Phase -> Answer Phase
1. **Planner Phase**: agent 进行search, access, summary与write outline几种操作，为后续的report (write phase)完成一份全面的大纲，并不断改进大纲的质量。

   (1) search: agent给出搜索词，通过搜索引擎进行搜索，返回搜索结果

   (2) access: agent给出要访问的URL与访问想要达到的目标，返回网页信息

   (3) summary: 为agent提供目标与网页信息，要求agent生成摘要

   (4) write outline: 为agent提供memory bank中新增的摘要与原有的outline，生成更加全面的outline并对摘要进行引用

2. **Writer Phase**: agent交替进行retrieve和write两种操作，先从memory bank中检索摘要，而后完成report。

   (1) retrieve: 根据outline中给出的摘要编号与要完成的对应章节信息，向memory bank检索摘要，返回对应的摘要

   (2) write: 根据outline，摘要与给出的要完成的章节信息，完成对应章节的写作

3. **Answer Phase**: 根据report, outline, history等信息，agent输出题目的最终答案

总流程：start -> ( (search -> access -> summary) * n -> write_outline ) * m -> terminate -> ( retrieve -> write ) -> write_terminate -> answer

### 机制介绍
1. **summary with goal** 重点为了解决摘要质量的问题。我们提出让agent阅读网页信息，生成摘要之前，提出自己阅读网页的目标goal，之后，在agent生成摘要时，我们并不将历史信息提供给agent，而是只将网页信息和目标提供给agent，使其专注在当前浏览的网页，防止其未准确引用相关链接；同时，这种机制也防止了agent阅读网页时没有侧重点的问题，goal可以明确给出agent阅读网页的指引，也可以让摘要对任务更有帮助；另外，为了进一步提高摘要质量，在设计提示词时，我们要求agent根据goal给出准确的Rational（给出网页内容中与goal直接相关的部分的位置），Evidence（给出与goal最相关的信息），Summary（给出摘要），这样可以进一步保证生成内容和网页信息相关。

2. **memory bank** 重点为了解决对摘要的管理机制。ASearcher有强大的多轮工具调用能力，而summary with goal生成的摘要长度又会很长，并且摘要内容对agent思考可能并没有过多的帮助，所以我们提出memory bank，在agent生成完摘要后将摘要存储在memory bank中，在有需要时进行调用，比如生成report或生成outline时。agent访问memory bank的形式是cite and retrieve，在生成outline时，agent会将摘要的编号引用在需要的位置，而之后在完成report时，agent会对相应编号的summary进行检索（retrieve），memory bank 返回对应的summary，并放入后续的提示词中。

3. **planner and writer** 这个结构重点为了连接起summary with goal和memory bank两个部分，让ASearcher拥有更强的summary与report能力，agent在任务的前半部分会扮演planner的角色，思考问题的解决路径，进行搜索（search），访问（access），总结（summary），以及编写大纲（write outline）；agent在后半部分的任务是扮演writer的角色，完成report，agent会先进行retrieve，根据outline中引用的summary编号向memmory bank检索对应的summary，之后，我们将检索到的summary与outline一同提供给agent，要求其一段一段地完成report。这样的结构形成了一套高效的对summary进行引用、检索和使用的流程框架。

### 实现逻辑
1. 修改原有prompt，新的prompt分为`ASearcherWeaverPlannerPrompt`与`ASearcherWeaverWriterPrompt`两类，分别支持了planner与writer的新操作，如write outline, retrieve 与 write；在planner阶段，新增的摘要与现有的大纲均会出现在提示词中。

2. summary with goal: 要求agent要进行access时提供goal，之后程序可以解析出URL与goal，访问成功后会将网页内容与goal都嵌入到summary提示词

3. memory bank: 作为agent的一个属性出现，本质上是一个字典，key是summary的编号，value是summary的内容。agent维护了一个summary的计数器，每次要进行summary时会加一，从而对summary进行编号，也便于后续的引用与检索

4. planner and writer: planner停止会输出“terminate” token 而后通过原有的history机制强制进入writer阶段

### 运行效果
样例见agent/demo.txt, agent/demo_report.md (题目来源ASearcher论文), agent/demo_report2.md (题目来源HLE)

# 修改内容
1. `agent`文件夹：新增`asearcherweaver.py`agent相关代码， `prompts.py` 提示词，`README.md`项目整体介绍，`demo.txt`项目运行实例。
2. `demo`文件夹：修改`asearcher_demo.py` 进行对已有agent的适配，但目前不支持asearcherweaver（其demo位于evaluation/inferrence.py）
3. `evaluation`文件夹：`inferrnce.py`模型推理代码，也是demo，`run_demo.sh`启动demo的脚本

# 快速体验
1. 进入`evaluation`文件夹，在`eval_config.yaml`中填入api信息。
2. 在`run_demo.sh`中`PYTHONPATH`填ASearcher项目路径，其余参数正常补充
3. 进入`inferrence.py`在文件最后`prompt`变量中填入提示词
4. 终端输入`./run_demo.sh`

# 代码结构
1. `prompts.py`: 模型提示词。
   1. `ASearcherWeaverPlannerPrompt`: Planner的最终目标是生成一个`outline`所以当前的`outline`会出现在提示词中。
      1. `PLANNER_THINK_AND_ACT_PROMPT_v1`: 用于planner工作，可以停止。模型对工具的调用分为：
         1. `search`: 与原代码相同
         2. `access`: 在原代码的基础上要求模型提供访问网页的目标`goal`，用于未来生成摘要，使用`<access> the url to access <goal> the goal to achieve by accessing this url </goal> </access>`的结构表示
         3. `write_outline`: 为后续的`report`生成`outline`.
         4. `terminate`: planner停止工作
      2. `THINK_AND_ACT_PROMPT`: 没有`terminate`
      3. `READ_PAGE_PROMPT`: 这个prompt来自https://github.com/Alibaba-NLP/DeepResearch/，用于根据`goal`生成网页内容的摘要，被`<summary></summary>`包裹
      4. `READ_SEARCH_RESULTS_PROMPT`: 与原代码相同。
      5. `WRITE_OUTLINE_PROMPT`: 要求直接引用摘要的id，形如`<cite> <id>id1</id>, <id>id2</id>, ... </cite>`
   2. `ASearcherWeaverWriterPrompt`
      1. `RETRIEVE_PROMPT`: 从`memory_bank`中获取`summary`
      2. `WRITE_PROMPT`: 继续根据`outline`写一节
2. `asearcherweaver.py`: 改编自`asearcher_reasoning.py`，改变内容包括:
   1. 新增属性：
      1. `summary_id_counter`: 维护当前summary的数量，每次access时加一 
      2. `summary_id_flag`: 维护已经被写入outline的最大id，之后agent要write outline时会将(summary_id_flag, summary_id_counter]的所有summary提供给agent
      3. `memory_bank`: 字典，存储所有`summary`，键为`id`(int)
      4. `outline`: 存储当前的`outline`
   2. 新的prompt来源，来自`prompts.py`。
   3. 新的`get_(.*?)_from_text`函数：
      1. 包括对新增的`<write_outline>`, `<retrieve></retrieve>`, `<terminate>`，`<write_terminate>`的初步处理；
      2. 新增`get_summary_from_text`可以返回生成的摘要和摘要编号，方便后续加入`memory_bank`
      3. 新增`get_outline_from_text`返回生成的`outline`
      4. 新增`get_report_from_text`返回生成的`report`
   4. `consume_tool_response`: 
      1. 修改对`<access>`的处理，增加`goal`属性
      2. 新增`<write_outline>`, `<retrieve></retrieve>`
   5. `prepare_llm_query`: 
      1. 修改处理`page`的prompt模式；
      2. 新增`history[-1]`无`text`属性时，类型为`retrieve`的提示词设置，使用了`WRITE_PROMPT`
      3. 新增类型为`write`的提示词设置，使用了`RETRIEVE_PROMPT`
      4. 新增类型`outline`的提示词设置，使用`WRITE_OUTLINE_PROMPT`
   6. `consume_llm_response`: 
      1. 新增`<write_outline>`, `<retrieve></retrieve>`, `<terminate>`；
      2. 若`get_summary_from_text`捕捉到`summary`则会立即将键值对加入`memory_bank`
      3. 若`get_outline_from_text`捕捉到`outline`则会立即更新`agent.outline`
      4. 若`get_report_from_text`捕捉到`report`则会立即更新`agent.report`，并在`history`中添加`type=write`的项
3. `inference.py`: 使用asearcherweaver进行推理，改编自`searcher_eval_async.py`，改变内容包括:
   1. `convert_agent_tool_calls_to_dict`: 新增`<write_outline>`, `<retrieve></retrieve>`, `<terminate>`，`<write_terminate>`；修改对`<access>`的处理，产生的`tool_call`不再使用`url`属性，而是使用`content`属性，后续会再次解析出`url`和`goal`
   2. 新增`parse_retrieved_ids`: 解析检索到的id，输出字符串数组
   3. `process_single_work_item`: 
      1. 分离出`planner loop`生成`outline`，`outline`由`process[outline]`维护。
      2. 新增对`<write_outline>`, `<terminate>`，`<write_terminate>`的处理；
      3. 修改对`<access>`的处理，增加`goal`属性。
      4. 新增对 `<retrieve></retrieve>`的处理，先获取`summaries`和`goal`，然后把任务直接交给`history`用于后续写作
      5. 新增`outline`，只在`planner`结束工作后更新
      6. 新增`report`，只在`writer`结束工作后更新