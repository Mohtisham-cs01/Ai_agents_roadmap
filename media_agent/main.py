# project4_calculator_agent.py

import os

from pathlib import Path
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, ToolMessage
from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
from typing import TypedDict


# ============================================================
# 1. STATE
# ============================================================

class State(TypedDict):
    messages: list


# ============================================================
# 2. CALCULATOR TOOL
# ============================================================

@tool
def calculator(expression: str) -> str:
    """
    Calculate a mathematical expression.

    Example:
        calculator("25 * 4 + 10")
    """

    allowed = set("0123456789+-*/().% ")

    if not expression:
        return "Expression is empty."

    if not all(char in allowed for char in expression):
        return "Invalid expression."

    try:
        result = eval(
            expression,
            {"__builtins__": {}},
            {},
        )

        return str(result)

    except Exception as error:
        return f"Calculation error: {error}"


tools = {
    "calculator": calculator,
}


# ============================================================
# 3. LLM
# ============================================================
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

llm = ChatOpenAI(
    model="openai",
    api_key=os.environ["POLLINATIONS_API_KEY"],
    base_url="https://gen.pollinations.ai/v1",
    temperature=0,
)

llm = llm.bind_tools(
    [calculator]
)


# ============================================================
# 4. AGENT NODE
# ============================================================

def agent(state: State):

    response = llm.invoke(
        state["messages"]
    )

    return {
        "messages": [
            *state["messages"],
            response,
        ]
    }


# ============================================================
# 5. TOOL NODE
# ============================================================

def run_tool(state: State):

    last_message = state["messages"][-1]

    new_messages = []

    for tool_call in last_message.tool_calls:

        tool_name = tool_call["name"]

        tool_args = tool_call["args"]

        tool_result = tools[tool_name].invoke(
            tool_args
        )

        new_messages.append(
            ToolMessage(
                content=str(tool_result),
                tool_call_id=tool_call["id"],
            )
        )

    return {
        "messages": [
            *state["messages"],
            *new_messages,
        ]
    }


# ============================================================
# 6. ROUTER
# ============================================================

def route(state: State):

    last_message = state["messages"][-1]

    # Agent wants to use a tool
    if last_message.tool_calls:
        return "tool"

    # Agent has finished
    return END


# ============================================================
# 7. BUILD GRAPH
# ============================================================

graph_builder = StateGraph(State)

graph_builder.add_node(
    "agent",
    agent,
)

graph_builder.add_node(
    "tool",
    run_tool,
)

# START → Agent
graph_builder.add_edge(
    START,
    "agent",
)

# Agent → Tool OR END
graph_builder.add_conditional_edges(
    "agent",
    route,
    {
        "tool": "tool",
        END: END,
    },
)

# Tool → Agent
graph_builder.add_edge(
    "tool",
    "agent",
)

graph = graph_builder.compile()


# ============================================================
# 8. RUN
# ============================================================

def main():

    print("Calculator Agent")
    print("-----------------")
    print("Type 'exit' to quit.\n")

    while True:

        question = input("> ")

        if question.lower() == "exit":
            break

        result = graph.invoke(
            {
                "messages": [
                    HumanMessage(
                        content=question
                    )
                ]
            }
        )

        final_message = result["messages"][-1]

        print(
            "\n",
            final_message.content,
            "\n",
        )


if __name__ == "__main__":
    main()
