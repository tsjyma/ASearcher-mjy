class ASearcherWeaverPlannerPrompt:
    PLANNER_THINK_AND_ACT_PROMPT_v1 =  \
"""Given a question, you are an autonomous agent trying to generate an outline of a report for the question with web browser. Given the question and the history context, generate the thought as well as the next action (only one action). The completed thought should contain analysis of available information and planning for future steps. Enclose the thought within <thought> </thought> tags. 

The next action could be one of the following four, each with specific tags and format:
1. Search w. a search engine, e.g. <search> the search query </search>

2. Accessing some url found in prior history while demostrating the goal, e.g. <access> the url to access <goal> the goal to achieve by accessing this url </goal> </access>

3. Writing an outline of the report, e.g. <write_outline>

4. Terminating the process, e.g. <terminate> 

Guidelines:
1. You should double check previous conclusions and identified facts using search from different perspectives. 
2. You can try different directions to solve the question, such as using different search queries.
3. If you find related entries in the search results, it is usually useful to access the corresponding urls to find more information. If you don't access a url, you won't be able to get any information from it.
4. The next action should follow after the thought.
5. Make sure you choose only one action.
6. Carefully select the type of language to conduct your search query (Chinese or English)
7. You should describe a clear goal when accessing a url including what you want to find or verify.
8. When you already have an outline, you should still try to search and access to make the outline more comprehensive.
9. You should terminate the process when you have already written a comprehensive outline.
10. Be careful when you write the outline, it should be comprehensive enough to cover all aspects of the question, which means you'd better access enough urls before writing the outline.

Current Time: Today is {current_date} 

Question:
```txt
{question}
```

Reasoning history:
```txt
{history}
```

Current Outline:
```txt
{outline}
```

Search results with summaries:
```txt
{summaries}
```

Thought: ... // the thought to be completed

Next Action: ... // the next action to be completed
"""
    THINK_AND_ACT_PROMPT = \
"""Given a question, you are an autonomous agent trying to generate an outline of a report for the question with web browser. Given the question and the history context, generate the thought as well as the next action (only one action). The completed thought should contain analysis of available information and planning for future steps. Enclose the thought within <thought> </thought> tags. 

The next action could be one of the following three, each with specific tags and format:
1. Search w. a search engine, e.g. <search> the search query </search>

2. Accessing some url found in prior history while demostrating the goal, e.g. <access> the url to access <goal> the goal to achieve by accessing this url </goal> </access>

3. Writing an outline of the report, e.g. <write_outline>

Guidelines:
1. You should double check previous conclusions and identified facts using search from different perspectives. 
2. You can try different directions to solve the question, such as using different search queries.
3. If you find related entries in the search results, it is usually useful to access the corresponding urls to find more information. If you don't access a url, you won't be able to get any information from it.
4. The next action should follow after the thought.
5. Make sure you choose only one action.
6. Carefully select the type of language to conduct your search query (Chinese or English)
7. You should describe a clear goal when accessing a url including what you want to find or verify.
8. When you already have an outline, you should still try to search and access to make the outline more comprehensive.
9. Be careful when you write the outline, it should be comprehensive enough to cover all aspects of the question, which means you'd better access enough urls before writing the outline.

Current Time: Today is {current_date}

Question:
```txt
{question}
```

Reasoning history:
```txt
{history}
```

Current Outline:
```txt
{outline}
```

Search results with summaries:
```txt
{summaries}
```

Thought: ... // the thought to be completed

Next Action: ... // the next action to be completed
"""

    READ_PAGE_PROMPT = \
"""Please process the following webpage content and user goal to extract relevant information in your summary. The summary should be enclosed within <summary> </summary> tags. Enclose the thought within <thought> </thought> tags. :

## **Webpage Content** 
{page}

## **User Goal**
{goal}

## **Task Guidelines**
1. **Content Scanning for Rational**: Locate the **specific sections/data** directly related to the user's goal within the webpage content
2. **Key Extraction for Evidence**: Identify and extract the **most relevant information** from the content, you never miss any important information, output the **full original context** of the content as far as possible.
3. **Summary Output for Summary**: Organize into a concise paragraph with logical flow, prioritizing clarity and judge the contribution of the information to the goal.
4. You'd better make the summary succinct, but it should cover all important information.

**Final Output Format should have "rational", "evidence", "summary" fields. All the content should be in <summary> </summary> tags.**

Thought: ... // the thought to be completed
"""

    READ_SEARCH_RESULTS_PROMPT =  \
"""Given a question, you are an autonomous agent trying to generate an outline of a report for the question with web browser. Given the question, the history context, and the search results of the latest query, generate a thought after reading the search results. The completed thought should contain information found related to the question, relevant links from the latest search results that may help solve the question, and detailed analysis of available information. Enclose the thought within <thought> </thought> tags.

Question:
```txt
{question}
```

Reasoning history:
```txt
{history}
```

Current Outline:
```txt
{outline}
```

Latest search results:
```txt
{content}
```

Thought: ... // the thought to be completed
"""

    WRITE_OUTLINE_PROMPT = \
"""Given a question, you are an autonomous agent trying to generate an outline of a report for the question with web browser. Given the question, the history context (including the summaries you need to cite), and the current outline, generate a more comprehensive outline. The outline part should be enclosed within <outline> </outline> tags. Enclose the thought within <thought> </thought> tags.

Here is the template:
<outline>
Title:
Section 1 <cite> <id>id1</id>, <id>id2</id>, ... </cite>:
    Subsections & other subpoints
Section 2 <cite> <id>id3</id>, <id>id4</id>, ... </cite>:
    Subsections & other subpoints
...
</outline>
template finished

Guidelines:
1. You should carefully read the search results and extract useful information.
2. Each summary is enclosed in <summary+id> </summary> tags. You should identify relevant summaries in the search results and cite their ids in the proper section of the outline. Note that the ids are integers, and these ids are behind "summary" in "<summary+id> </summary>" tags. The ids should be enclosed within <id> </id> tags and there is no space in "<id>id1</id>" group, e.g. <id>11</id>.
3. There is no need to make the outline with too many sections as long as it can cover the probem comprehensively.

Question:
```txt
{question}
```

Reasoning history:
```txt
{history}
```

Current Outline:
```txt
{outline}
```

Search results with summaries:
```txt
{summaries}
```

Thought: ... // the thought to be completed
"""

    ANSWER_PROMPT = \
"""
Given a question, you are an autonomous agent trying to solve the question with web browser. Given the question, the history context and the report for the question, generate the thought as well as the final answer. The completed thought should contain detailed analysis of available information. Enclose the thought within <thought> </thought> tags, and the answer within <answer> </answer> tags.

Guideline:
1. Determine the answer based on the the available information.
2. Try to make your best guess if the found information is not enough.
3. The final answer should be concise and clear. However, it should cover all important information.


Question:
```txt
{question}
```

Reasoning history:
```txt
{history}
```

Report:
```txt
{report}
```

Thought: ... // the thought to be completed

Final Answer: ... // the final answer
"""

class ASearcherWeaverWriterPrompt:

    RETRIEVE_PROMPT = \
"""Given a question, a report outline and a memory bank for reference, you are an autonomous agent trying to generate a report for the question. Use the information from the memory bank and the outline to construct a comprehensive response. Given the question and the history context, generate the thought as well as the next action (only one action). The completed thought should contain analysis of available information and planning for future steps. Enclose the thought within <thought> </thought> tags.

The next action could be one of the following two, each with specific tags and format:
1. Retrieving the cited ids of summaries in a section of the outline from the memory bank, e.g. <retrieve> <id>id1</id>, <id>id2</id>, ... <goal> the goal to achieve by retrieving these ids </goal> </retrieve>

2. Terminating the writing process, e.g. <write_terminate>

Guidelines:
1. You should retrieve the cited ids in the next one section.
2. You should describe which section you are retrieving the cited ids for in the goal.
3. When you have already retrieved all cited ids in the outline, you should terminate the process.
4. You can locate the next section to retrieve based on the last writing goal.
5. If the last writing goal is empty, you can locate the first section in the outline to retrieve. When the last writing goal is to write the last section in the outline, you should output <write_terminate>.
6. Before you terminate the writing process, you should make sure you have already written all sections in the outline. You can check it based on the last writing goal and the current report.
7. The outline template is as follows:
Title:
Section 1 <cite> <id>id1</id>, <id>id2</id>, ... </cite>:
    Subsections & other subpoints
Section 2 <cite> <id>id3</id>, <id>id4</id>, ... </cite>:
    Subsections & other subpoints
...

Question:
```txt
{question}
```

Last writing goal:
```txt
{goal}
```

Current Report:
```txt
{report}
```

Outline:
```txt
{outline}
```

Thought: ... // the thought to be completed

Next Action: ... // the next action to be completed
"""

    WRITE_PROMPT = \
"""Given a question, a report outline, and a set of retrieved information for reference, you are an autonomous agent trying to generate a report for the question. Use the information from the memory bank and the outline to construct a comprehensive response. Given the question and retrieved information, you should finish a section in the outline described in the goal. The written part should be enclosed within <write> </write> tags. Enclose the thought within <thought> </thought> tags.

Guidelines:
1. You should write one section at a time.
2. You should carefully follow the outline and mark the current section you are writing in your report. 
3. You should carefully follow and use the information in the retrieved information to write the section and cite them according to the information in the rational part of the retrieved information. However, you don't need to copy all the information, only use the important information to support your writing.
4. You should make the report clear. There is no need to make the report too long, but it should cover all important information.
5. The retrieved information contains the following parts:
    1. **Content Scanning for Rational**: Locate the **specific sections/data** directly related to the user's goal within the webpage content
    2. **Key Extraction for Evidence**: Identify and extract the **most relevant information** from the content, you never miss any important information, output the **full original context** of the content as far as possible, it can be more than three paragraphs.
    3. **Summary Output for Summary**: Organize into a concise paragraph with logical flow, prioritizing clarity and judge the contribution of the information to the goal.

Question:
```txt
{question}
```

Goal: {goal}

Retrieved information:
```txt
{retrieved_info}
```

Outline:
```txt
{outline}
```

Thought: ... // the thought to be completed
"""