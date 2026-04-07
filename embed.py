import json
import os
from openai import OpenAI
from pinecone import Pinecone
from dotenv import load_dotenv
import time

load_dotenv()

openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index("scentdb")

with open('fragrances_processed.json', 'r') as f:
    fragrances = json.load(f)

print(f"Loaded {len(fragrances)} fragrances")

def get_embedding(text):
    response = openai_client.embeddings.create(
        input=text,
        model="text-embedding-3-small"
    )
    return response.data[0].embedding

batch_size = 100
total = len(fragrances)

for i in range(0, total, batch_size):
    batch = fragrances[i:i+batch_size]
    vectors = []
    
    for j, f in enumerate(batch):
        try:
            embedding = get_embedding(f['text'])
            vectors.append({
                "id": str(i + j),
                "values": embedding,
                "metadata": {
                    "name": str(f.get('Perfume', '')),
                    "brand": str(f.get('Brand', '')),
                    "text": str(f.get('text', ''))[:1000]
                }
            })
            time.sleep(0.05)
        except Exception as e:
            print(f"Error on {f.get('Perfume')}: {e}")
            continue
    
    index.upsert(vectors=vectors)
    print(f"Uploaded {min(i+batch_size, total)}/{total}")

print("Done. All fragrances embedded and stored.")
