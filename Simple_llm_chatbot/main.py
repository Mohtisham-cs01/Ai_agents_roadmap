import os
from openai import OpenAI


client = OpenAI(
    base_url="https://gen.pollinations.ai/v1",
    api_key="sk_60a4qNn7hLwfTFjgkMwZrydPxHhBNEDg",
)


messages = [
    {
        "role": "system",
        "content": "You are a helpful AI assistant."
    }
]


while True:
    user_input = input("\nYou: ")

    if user_input.lower() in ["exit", "quit"]:
        print("Goodbye!")
        break

    messages.append({
        "role": "user",
        "content": user_input
    })

    response = client.chat.completions.create(
        model="openai",
        messages=messages,
    )

    assistant_message = response.choices[0].message.content

    print(f"AI: {assistant_message}")

    messages.append({
        "role": "assistant",
        "content": assistant_message
    })
