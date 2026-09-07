from idlelib import scrolledlist
from typing import Any, Dict

from graph.chains.retrieval_grader import retrieval_grader
from graph.state import GraphState



def grade_documents(state: GraphState) -> Dict[str, Any]:
    """
    Determines whether the retrieved documents are relevant to the question
    If any document is not relevant, we will set a flag to run web_search

    Args:
        state (dict): GraphState object

    Returns:
        Dict[str, Any]: Dictionary of relevant documents and updated web_search state
    """
    print('--- CHECK DOCUMENT RELEVANCE TO QUESTION ---')
    question = state['question']
    documents = state['documents']

    filtered_docs = []
    web_search = False
    for document in documents:
        score = retrieval_grader.invoke({"question": question, "document": document.page_content})

        grade = score.binary_score
        if grade.lower() == "yes":
            print('--- RELEVANT TO QUESTION ---')
            filtered_docs.append(document)
        else:
            print('--- NOT RELEVANT TO QUESTION ---')
            web_search = True
            continue

    return {"filtered_docs": filtered_docs, "web_search": web_search, "question": question}