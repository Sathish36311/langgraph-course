from dotenv import load_dotenv
load_dotenv()
from typing import Any, Dict
from langchain_core.documents import Document
from graph.state import GraphState
from langchain_tavily import TavilySearch


web_search_tool = TavilySearch(max_results=3)

def web_search(state: GraphState) -> Dict[str, Any]:
    print('---WEB SEARCH---')
    question = state['question']
    documents = state['documents']

    tavily_results = web_search_tool.invoke({"query": question})
    joined_tavily_results = "\n".join([tavily_result["content"] for tavily_result in tavily_results])

    web_results = Document(page_content=joined_tavily_results)
    if documents is not None:
        documents.append(web_results)
    else:
        documents = [web_results]
    return {"question": question, "documents": documents}