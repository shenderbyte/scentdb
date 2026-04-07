from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pinecone import Pinecone
from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()

app = FastAPI()

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index("scentdb")
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

class SearchRequest(BaseModel):
    query: str
    top_k: int = 5

@app.get("/")
def root():
    return {"status": "scentdb api is running"}

@app.post("/search")
def search(request: SearchRequest):
    response = openai_client.embeddings.create(
        input=request.query,
        model="text-embedding-3-small"
    )
    vector = response.data[0].embedding
    
    results = index.query(vector=vector, top_k=request.top_k, include_metadata=True)
    
    matches = []
    for match in results.matches:
        matches.append({
            "name": match.metadata["name"],
            "brand": match.metadata["brand"],
            "score": round(match.score, 3),
            "description": match.metadata["text"]
        })
    
    return {"query": request.query, "results": matches}