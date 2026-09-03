# structured output using simple prompt template
# import os
# from pathlib import Path
# from typing import Literal

# from dotenv import load_dotenv
# from openai import OpenAI
# from pydantic import BaseModel, Field, ValidationError


# # --------------------------------------------------
# # 1. Load .env from parent directory
# # --------------------------------------------------

# PROJECT_DIR = Path(__file__).resolve().parent
# ENV_FILE = PROJECT_DIR.parent / ".env"

# load_dotenv(ENV_FILE)


# # --------------------------------------------------
# # 2. OpenAI-compatible client
# # --------------------------------------------------

# client = OpenAI(
#     base_url="https://gen.pollinations.ai/v1",
#     api_key=os.environ["POLLINATIONS_API_KEY"],
# )


# # --------------------------------------------------
# # 3. Define expected structured output
# # --------------------------------------------------

# class TopicAnalysis(BaseModel):
#     topic: str
#     difficulty: Literal["beginner", "intermediate", "advanced"]
#     summary: str
#     keywords: list[str] = Field(
#         min_length=1,
#         max_length=5
#     )


# # --------------------------------------------------
# # 4. Get user input
# # --------------------------------------------------

# user_input = input("Enter a programming topic: ")


# # --------------------------------------------------
# # 5. Call LLM
# # --------------------------------------------------

# response = client.chat.completions.create(
#     model="openai",
#     messages=[
#         {
#             "role": "system",
#             "content": """
# You are a programming teacher.

# Analyze the user's programming topic.

# Return ONLY valid JSON in exactly this format this is most important to follow structure otherwise someones life is at risk:

# {
#     "topic": "string",
#     "difficulty": "beginner | intermediate | advanced",
#     "summary": "string",
#     "keywords": ["string"]
# }

# Rules:
# - keywords must contain between 1 and 5 items
# - difficulty must be beginner, intermediate, or advanced
# """
#         },
#         {
#             "role": "user",
#             "content": user_input
#         }
#     ],
# )


# # --------------------------------------------------
# # 6. Get raw response
# # --------------------------------------------------

# raw_output = response.choices[0].message.content

# print("\nRaw LLM response:")
# print(raw_output)


# # --------------------------------------------------
# # 7. Validate response
# # --------------------------------------------------

# try:
#     result = TopicAnalysis.model_validate_json(raw_output)

#     print("\n✅ Validation successful!")

#     print("\nValidated result:")
#     print(result)

#     print("\nTopic:", result.topic)
#     print("Difficulty:", result.difficulty)
#     print("Summary:", result.summary)
#     print("Keywords:", result.keywords)

# except ValidationError as error:
#     print("\n❌ Validation failed!")
#     print(error)
///////////////////////////////

# using openai sdk structured output

import os
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, Field


# Load .env from parent directory
PROJECT_DIR = Path(__file__).resolve().parent
load_dotenv(PROJECT_DIR.parent / ".env")


# OpenAI-compatible client
client = OpenAI(
    base_url="https://gen.pollinations.ai/v1",
    api_key=os.environ["POLLINATIONS_API_KEY"],
)


# Schema
class TopicAnalysis(BaseModel):
    topic: str
    difficulty: Literal["beginner", "intermediate", "advanced"]
    summary: str
    keywords: list[str] = Field(min_length=1, max_length=5)


# User input
user_input = input("Enter a programming topic: ")


# Structured output
response = client.beta.chat.completions.parse(
    model="openai",
    messages=[
        {
            "role": "system",
            "content": "You are a programming teacher. Analyze the user's programming topic."
        },
        {
            "role": "user",
            "content": user_input
        }
    ],
    response_format=TopicAnalysis,
)


# Already parsed + validated
result = response.choices[0].message.parsed


print("\n✅ Result:")
print(result)

print("\nTopic:", result.topic)
print("Difficulty:", result.difficulty)
print("Summary:", result.summary)
print("Keywords:", result.keywords)
