import os

from dotenv import load_dotenv
from typing import TypedDict, Annotated, List
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph import END, StateGraph
from langgraph.graph.message import  add_messages
from chains import generation_chain, reflection_chain

load_dotenv()

class MessageGraph(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]

REFLECT="reflect"
GENERATE="generate"


def generation_node(state: MessageGraph):
    return { "messages": [generation_chain.invoke({"messages": state["messages"]}) ] }

def reflection_node(state: MessageGraph):
    res = reflection_chain.invoke({"messages": state["messages"]})
    return { "messages": [HumanMessage(content=res.content)]}

builder = StateGraph(state_schema=MessageGraph)
builder.add_node(GENERATE,generation_node)
builder.add_node(REFLECT,reflection_node)
builder.set_entry_point(GENERATE)


def should_end(state: MessageGraph):
    if len(state["messages"]) > 6:
        return END
    return REFLECT


builder.add_conditional_edges(GENERATE,should_end, {END: END, REFLECT: REFLECT})
builder.add_edge(REFLECT,GENERATE)

graph = builder.compile()
print(graph.get_graph().draw_mermaid())



if __name__ == "__main__":
    inputs = HumanMessage(content=("I play outside hitter. The opposing team has a tall, solid "
                                   "two-man block shutting down my line shots. What shot selections "
                                   "and approach adjustments should I use to score around or off the block?"))
    response = graph.invoke(inputs)

