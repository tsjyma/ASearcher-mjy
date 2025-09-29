# 项目概况
我的代码主要实现了以下几点：
1. **summary with goal**: 在模型生成摘要时，要求模型根据goal和page生成摘要，而不是给模型全部的历史信息，这样可能可以防止模型被历史信息干扰，专注于生成高质量的摘要，这个方法来自https://github.com/Alibaba-NLP/DeepResearch/，测试发现，目前摘要质量不错
2. **memory bank**: memory bank可以存储所有的摘要信息，memory bank中的摘要可以通过id来查找。这个结构来自Li, Zijian, et al. "WebWeaver: Structuring Web-Scale Evidence with Dynamic Outlines for Open-Ended Deep Research." arXiv preprint arXiv:2509.13312 (2025). 这些摘要可以被用来生成outline和report
3. **planner & writer structure**: 我将模型的任务分为生成大纲和生成报告两个部分，planner的目标是生成大纲，大纲中每个章节直接引用summary的id，writer的目标是生成报告，报告中使用summary时直接从memory bank中获得。这也是为了保证，在部分简单但要求质量的任务中，没有历史信息干扰。这个结构也来自Li, Zijian, et al. "WebWeaver: Structuring Web-Scale Evidence with Dynamic Outlines for Open-Ended Deep Research." arXiv preprint arXiv:2509.13312 (2025). agent运行流程：start -> ( (search -> access -> summary) * n -> write_outline ) * m -> terminate -> ( retrieve -> write ) -> write_terminate

# 修改内容
1. `agent`文件夹：新增`asearcherweaver.py`agnet相关代码， `prompts.py` 提示词， `ASEARCHERWEAVER实现笔记.pdf` 实现的心路历程，`README.md`项目整体介绍。
2. `demo`文件夹：修改`asearcher_demo.py` 进行对已有agent的适配，但目前不支持asearcherweaver（其demo位于evaluation/inferrence.py）
3. `evaluation`文件夹：新增`fish.txt`, `fish1.txt`, `fish2.txt`项目demo，`inferrnce.py`模型推理代码，也是demo，`run_demo.sh`启动demo的脚本

# 快速体验
1. 进入`evaluation`文件夹，在`eval_config.yaml`中填入api信息。
2. 在`run_demo.sh`中`PYTHONPATH`填ASearcher项目路径，其余参数正常补充
3. 进入`inferrence.py`在文件最后`prompt`变量中填入提示词
4. 终端输入`./run_demo.sh`

# 样例展示
prompt: What is Dijkstra's algorithm and how does it work?

Turn 1 Search (search for information)
<search>Dijkstra's algorithm definition and how it works</search>
search result: 5 DOCs about Dijkstra's algorithm.

------------------------------------------------------------------------------------------------------------------------

Turn 2 Access (get the content in the webpage)
<access> https://en.wikipedia.org/wiki/Dijkstra%27s_algorithm <goal> To get a comprehensive and formal definition of Dijkstra's algorithm, its history, a detailed step-by-step explanation of how it works, its pseudocode, its properties (e.g., time complexity, inability to handle negative weights), and its applications. This will form the foundational content for the report outline. </goal> </access>

------------------------------------------------------------------------------------------------------------------------

Turn 3 Summary (read the page and summary with goal. Th summary contains 3 parts rationale, evidence and summary)
# More about summary:
1. **Content Scanning for Rational**: Locate the **specific sections/data** directly related to the user's goal within the webpage content
2. **Key Extraction for Evidence**: Identify and extract the **most relevant information** from the content, you never miss any important information, output the **full original context** of the content as far as possible, it can be more than three paragraphs.
3. **Summary Output for Summary**: Organize into a concise paragraph with logical flow, prioritizing clarity and judge the contribution of the information to the goal.
<summary>

<rationale>The user's goal is to gather foundational information on Dijkstra's algorithm for a report. The provided Wikipedia content is highly relevant as it contains a formal definition, historical context, an explanation of its operation, key properties (including time complexity and limitations), and its applications. The specific sections detailing the algorithm's purpose, its use of a priority queue, its performance, and its common uses directly address the user's request.</rationale>

<evidence>\n**Dijkstra's algorithm** ( [_DYKE-strəz_](https://en.wikipedia.org/wiki/Help:Pronunciation_respelling_key \"Help:Pronunciation respelling key\")) is an [algorithm](https://en.wikipedia.org/wiki/Algorithm \"Algorithm\") for finding the [shortest paths](https://en.wikipedia.org/wiki/Shortest_path_problem \"Shortest path problem\") between [nodes](https://en.wikipedia.org/wiki/Vertex_(graph_theory) \"Vertex (graph theory)\") in a weighted [graph](https://en.wikipedia.org/wiki/Graph_(abstract_data_type) \"Graph (abstract data type)\"), which may represent, for example, a [road network](https://en.wikipedia.org/wiki/Road_network \"Road network\")....... \n</evidence>

<summary>Dijkstra's algorithm is a foundational shortest-path algorithm conceived by Edsger W. Dijkstra in 1956 for finding the shortest paths between nodes in a weighted graph, such as a road network. It operates by using a min-priority queue to systematically determine the shortest path from a given source node to all other nodes, and it can be terminated early to find a path to a single destination. A key property is its requirement that edge weights be non-negative. Its time complexity is optimized to Θ(|E|+|V|log|V|) using a Fibonacci heap. Common applications include network routing protocols like IS-IS and OSPF, and it serves as a subroutine in other algorithms. The algorithm was famously designed in about twenty minutes without pencil and paper.</summary>

</summary>

------------------------------------------------------------------------------------------------------------------------

Repeat (search -> access -> summary)

------------------------------------------------------------------------------------------------------------------------

Turn 19 Write outline
<thought>However, given the depth already obtained, I can proceed to write the outline.\n</thought>\n\n<write_outline>
<outline>
Title: Dijkstra's Algorithm: A Comprehensive Overview
Section 1 <cite> <id>2</id> </cite>:\n    Introduction to Dijkstra's Algorithm\n    
Definition and Core Purpose\n    
Historical Context and Edsger W. Dijkstra\n    
Basic Principles and Greedy Approach\n\n

Section 2 <cite> <id>3</id> <id>6</id> </cite>:\n    
How Dijkstra's Algorithm Works\n    
Step-by-Step Operational Process\n        
Initialization of Distances\n        
Node Selection Mechanism\n        
Distance Update Procedure\n        
Marking Visited Nodes\n    
Visual Example and Walkthrough\n        
Sample Graph Illustration\n        
Iterative Distance Calculation\n\n

......
</outline>

------------------------------------------------------------------------------------------------------------------------

Turn 20 Access
<thought>The current outline is already quite comprehensive, covering all major aspects of Dijkstra's algorithm including definition, historical context, step-by-step operation, implementation details, theoretical foundations, variants, applications, and comparative analysis. The information gathered from Wikipedia and GeeksforGeeks provides solid coverage of these topics.\n\nHowever, the initial search results mentioned a YouTube video that could provide visual explanations, and while the Reddit thread was inaccessible, the YouTube resource might offer valuable visual demonstrations of the algorithm's working process. Since visual explanations can significantly enhance understanding of graph algorithms, accessing this video could provide additional insights for the \"How Dijkstra's Algorithm Works\" section, particularly for the visual example and walkthrough part.\n\nBefore finalizing the outline, I should access this YouTube video to see if it offers any unique visual explanations or demonstrations that could further enrich the report.

------------------------------------------------------------------------------------------------------------------------

Repeat (search -> access -> summary)

------------------------------------------------------------------------------------------------------------------------

Turn 26 Write outline
<thought>The current outline is already comprehensive, but the newly accessed YouTube video (summary10, 11, 12) provides exceptionally clear visual explanations and a step-by-step walkthrough that is highly praised by users. This is a valuable resource that should be integrated to enhance the \"How it Works\" section. I will update Section 2 to explicitly cite this visual source and add a subsection for the visual walkthrough. The key steps from the video's timestamps (marking unvisited, assigning distances, choosing the next node, updating distances) align perfectly with the existing subsections but provide a concrete, visual example to reference. I will cite the relevant YouTube summary IDs (10, 11) in this section to credit this excellent visual resource. No other major structural changes are needed as the outline already covers all critical aspects of the algorithm thoroughly.

------------------------------------------------------------------------------------------------------------------------

Turn 27 Terminate (planner finish)

------------------------------------------------------------------------------------------------------------------------

Turn 28 Retrieve (retrieve ids to get the summaries from memory bank, then they are used for write phase)
<retrieve> <id>3</id>, <id>6</id>, <id>10</id>, <id>11</id> <goal>Retrieve information for Section 2: How Dijkstra's Algorithm Works, covering step-by-step operational process and visual examples</goal>

------------------------------------------------------------------------------------------------------------------------

Turn 29 Write (write the report with given summaries)
given: <summary3><rationale>The user's goal is to find information relevant to answering a question. The entire provided content is about Dijkstra's algorithm. The most relevant sections are the detailed step-by-step description of the algorithm's process, the explanation of the accompanying illustration, and the complete pseudocode implementation with its explanation, as these form the core technical explanation of how the algorithm works.</rationale>\n    <evidence>\nThe algorithm requires a starting node, and computes the shortest distance from that starting node to each other node. Dijkstra's algorithm starts with infinite distances and tries to improve them step by step:\n\n1......
<write>
the content
</write>

------------------------------------------------------------------------------------------------------------------------

Repeat ( retrieve -> write )

------------------------------------------------------------------------------------------------------------------------

Turn 32 Write terminate

------------------------------------------------------------------------------------------------------------------------

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