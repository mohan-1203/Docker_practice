import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)


def generate_prediction(question, context):

    prompt = f"""
You are a student performance prediction assistant.

Use ONLY the information provided in the context.

Do not invent rules or information.

Return the answer in this exact format:

Prediction: <prediction>
Reason: <short reason>

Context:
{context}

Student Question:
{question}
"""

    response = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0
    )

    return response.choices[0].message.content