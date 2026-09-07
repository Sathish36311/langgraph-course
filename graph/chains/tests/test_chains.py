from dotenv import load_dotenv
load_dotenv()

from graph.chains.retrieval_grader import retrieval_grader, GradeDocuments
from ingestion import retriever


def test_retrieval_grader_answer_yes() -> None:
    question = "agent memory"
    docs =  retriever.invoke(question)
    docs_text = docs[1].page_content

    res: GradeDocuments = retrieval_grader.invoke(
        {"document": docs_text , "question": question}
    )

    assert res.binary_score == 'yes'


def test_retrieval_grader_answer_no() -> None:
    question = "agent memory"
    docs = retriever.invoke(question)
    docs_text = docs[1].page_content

    res: GradeDocuments = retrieval_grader.invoke(
        {"document": docs_text, "question": "Sports"}
    )

    assert res.binary_score == 'no'