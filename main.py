from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Dict
from uuid import uuid4

app = FastAPI()

# ✅ Enable CORS (allow all for testing; restrict in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # You can restrict this to your frontend's domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ Load the FLAN-T5 Large model for English Q&A and instruction-following
qa = pipeline("text2text-generation", model="google/flan-t5-base")

# ✅ In-memory chat sessions
chat_sessions: Dict[str, List[str]] = {}

# ✅ Request format
class Query(BaseModel):
    inputs: str
    session_id: str = None  # optional

# ✅ Main API route for generation
@app.post("/generate")
def generate_text(query: Query):
    session_id = query.session_id or str(uuid4())
    history = chat_sessions.get(session_id, [])

    # Use last 5 messages + current input
    full_prompt = "\n".join(history[-2:] + [f"User: {query.inputs}\nAI:"])

    output = qa(
        full_prompt,
        max_length=256,
        num_return_sequences=1,
        do_sample=True,
        top_k=50,
        top_p=0.95,
        temperature=0.9
    )[0]['generated_text']

    # Save chat history
    history.append(f"User: {query.inputs}")
    history.append(f"AI: {output}")
    chat_sessions[session_id] = history

    return {
        "result": output.strip(),
        "session_id": session_id,
        "history": history[-10:]  # return last 10 exchanges
    }
