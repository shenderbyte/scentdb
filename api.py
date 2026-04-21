from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pinecone import Pinecone
from openai import OpenAI
from dotenv import load_dotenv
import os
import json

load_dotenv()

# Load psychology lookup
try:
    with open("psych_lookup.json", "r") as f:
        PSYCH_LOOKUP = json.load(f)
    print(f"Loaded {len(PSYCH_LOOKUP)} psych entries")
except FileNotFoundError:
    PSYCH_LOOKUP = {}
    print("psych_lookup.json not found")

# Load map data
try:
    with open("map_data.json", "r") as f:
        MAP_DATA = json.load(f)
    print(f"Loaded {len(MAP_DATA)} map points")
except FileNotFoundError:
    MAP_DATA = []
    print("map_data.json not found")

app = FastAPI(title="scentdb", description="Vectorized fragrance search API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index("scentdb")
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

class SearchRequest(BaseModel):
    query: str
    top_k: int = 8

def get_psych_context(text: str) -> str:
    text_lower = text.lower()
    effects = []
    for word, word_effects in PSYCH_LOOKUP.items():
        if word in text_lower and word_effects:
            effects.append(word_effects[0])
    return ". ".join(effects[:3]) if effects else ""

@app.get("/")
def root():
    return {"status": "scentdb api is running"}

@app.get("/map")
def get_map():
    return {"points": MAP_DATA}

@app.get("/effects")
def get_effects(notes: str):
    return {"effects": get_psych_context(notes)}

@app.post("/search")
def search(request: SearchRequest):
    psych = get_psych_context(request.query)
    enriched = (
        f"Fragrance with olfactory qualities: {request.query}. "
        f"Focus on scent notes, accords, and sensory qualities only."
        + (f" Psychological effects: {psych}." if psych else "")
    )

    response = openai_client.embeddings.create(
        input=enriched,
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