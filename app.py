from fastapi import FastAPI
from pydantic import BaseModel

from rag import retrieve_context
from llm import generate_prediction


app = FastAPI()


class StudentRequest(BaseModel):
    attendance: float
    hours_studied: float


@app.get("/")
def home():
    return {
        "message": "Student GenAI application is running"
    }


@app.post("/predict")
def predict(student: StudentRequest):

    question = (
        f"Student has {student.attendance}% attendance "
        f"and studies {student.hours_studied} hours per day."
    )

    context = retrieve_context(question)

    print("QUESTION:", question)
    print("CONTEXT:", context)

    context_text = "\n\n".join(context)

    answer = generate_prediction(
        question,
        context_text
    )

    return {
        "question": question,
        "retrieved_context": context,
        "answer": answer
    }