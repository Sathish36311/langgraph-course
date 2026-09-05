from typing import List, TypedDict

from langchain_core import documents


class GraphState(TypedDict):
    """
    Represents the state of a graph state.

    Attributes:
        question: question
        documents: list of documents
        generation: LLM generation
        web_search: whether to use web search
    """
    question: str
    documents: List[str]
    web_search: bool
    generation: str