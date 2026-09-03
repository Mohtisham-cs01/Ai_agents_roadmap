import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI


# Load .env from parent directory
load_dotenv(Path(__file__).resolve().parent.parent / ".env")


# LLM
llm = ChatOpenAI(
    model="openai",
    base_url="https://gen.pollinations.ai/v1",
    api_key=os.environ["POLLINATIONS_API_KEY"],
)


# Tool
@tool
def calculator(expression: str) -> str:
    """Calculate a mathematical expression."""
    try:
        return str(eval(expression))
    except Exception as e:
        return f"Error: {e}"


# Give tool to LLM
llm_with_tools = llm.bind_tools([calculator])


# Conversation
messages = [
    (
        "user",
        input("You: ")
    )
]


# Agent/tool loop
while True:

    response = llm_with_tools.invoke(messages)

    # Add LLM response to conversation
    messages.append(response)

    # No tool call → final answer
    if not response.tool_calls:
        print("AI:", response.content)
        break

    # Execute requested tools
    for tool_call in response.tool_calls:

        print(
            f"🔧 Calling {tool_call['name']} "
            f"with {tool_call['args']}"
        )

        if tool_call["name"] == "calculator":

            tool_result = calculator.invoke(
                tool_call["args"]
            )

            messages.append(
                tool_call["id"]
                and {
                    "role": "tool",
                    "content": tool_result,
                    "tool_call_id": tool_call["id"],
                }
            )
