from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI

reflection_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a master volleyball analyst reviewing a coach's advice. "
            "Critique the advice for tactical clarity, biomechanics, and actionable drill recommendations. "
            "Point out missing adjustments, risks, and ways to improve the response.",
        ),
        MessagesPlaceholder(variable_name="messages")
    ]
)

generation_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an elite volleyball coach and tactical expert. "
            "Provide detailed, actionable advice and drills for the player's request. "
            "If the user provides critique or reflection, refine and expand your previous coaching advice.",
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)


llm = ChatOpenAI()
generation_chain =  generation_prompt | llm
reflection_chain = reflection_prompt | llm

