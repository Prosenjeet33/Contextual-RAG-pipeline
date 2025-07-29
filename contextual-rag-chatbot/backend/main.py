from fastapi import FastAPI, Request
from rag_pipeline import get_contextual_answer

app = FastAPI()

@app.post("/chat")
async def chat(req: Request):
    data = await req.json()
    user_query = data["query"]
    return {"response": get_contextual_answer(user_query)}