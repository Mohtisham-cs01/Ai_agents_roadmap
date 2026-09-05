import os
from typing import Literal

from dotenv import load_dotenv
from pydantic import BaseModel

# from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, START, END
from dotenv import load_dotenv
from pydantic import BaseModel
from pathlib import Path

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

class Classification(BaseModel):
    category: Literal[
        "billing",
        "technical",
        "account",
        "general"
    ]
    confidence: float
    reason: str


CATEGORIES = [
    "billing",
    "technical",
    "account",
    "general"
]
from typing_extensions import TypedDict

llm = ChatOllama(
    model="qwen2.5:14b",
    temperature=0,
)
structured_llm = llm.with_structured_output(Classification)


class GraphState(TypedDict):
    user_input: str
    category: str
    confidence: float
    reason: str
    response: str


def classify(state: GraphState):
    result = structured_llm.invoke(
        state["user_input"]
    )

    return {
        "category": result.category,
        "confidence": result.confidence,
        "reason": result.reason,
    }

def route_category(state: GraphState):
    return state["category"]

def general_handler(state: GraphState):
    return {
        "response": "This request has been classified as a general request."
    }
def account_handler(state: GraphState):
    return {
        "response": "This request has been routed to the account team."
    }
def technical_handler(state: GraphState):
    return {
        "response": "This request has been routed to technical support."
    }
def billing_handler(state: GraphState):
    return {
        "response": "This request has been routed to the billing team."
    }

builder = StateGraph(GraphState)
builder.add_node("classify", classify)

builder.add_node("billing", billing_handler)
builder.add_node("technical", technical_handler)
builder.add_node("account", account_handler)
builder.add_node("general", general_handler)

builder.add_edge(START , "classify")
builder.add_conditional_edges(
    "classify",
    route_category,
    {
        "billing": "billing",
        "technical": "technical",
        "account": "account",
        "general": "general",
    }
)

builder.add_edge("billing", END)
builder.add_edge("technical", END)
builder.add_edge("account", END)
builder.add_edge("general", END)
graph = builder.compile()

result = graph.invoke({
    "user_input": "My credit card was charged twice."
})


print(result)
