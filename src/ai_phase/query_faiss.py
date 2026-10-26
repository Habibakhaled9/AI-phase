import json
import os
from pathlib import Path

import faiss
import numpy as np
import requests
from dotenv import load_dotenv


load_dotenv()

API_KEY = os.getenv("embedding_model_key")

if not API_KEY:
    raise ValueError("OPENROUTER_API_KEY is missing")


MODEL = "nvidia/nemotron-3-embed-1b:free"

INDEX_FILE = Path("faiss_index.bin")
METADATA_FILE = Path("faiss_metadata.json")


def create_query_embedding(question: str) -> np.ndarray:
    response = requests.post(
        "https://openrouter.ai/api/v1/embeddings",
        headers={
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json",
        },
        json={
            "model": MODEL,
            "input": question,
        },
        timeout=120,
    )

    if not response.ok:
        raise RuntimeError(
            f"OpenRouter error {response.status_code}: "
            f"{response.text}"
        )

    result = response.json()

    vector = np.array(
        [result["data"][0]["embedding"]],
        dtype="float32",
    )

    faiss.normalize_L2(vector)

    return vector


# Load vector index
index = faiss.read_index(str(INDEX_FILE))

# Load corresponding text
with METADATA_FILE.open("r", encoding="utf-8") as file:
    metadata = json.load(file)


question = input("Enter your question: ")

query_vector = create_query_embedding(question)

TOP_K = 3

scores, indices = index.search(
    query_vector,
    TOP_K,
)


print("\nMost relevant chunks:\n")

for rank, (record_index, score) in enumerate(
    zip(indices[0], scores[0]),
    start=1,
):
    if record_index == -1:
        continue

    record = metadata[record_index]

    print(f"Result {rank}")
    print(f"Similarity: {score:.4f}")
    print(f"Page: {record['page']}")
    print(f"Chunk: {record['chunk']}")
    print(f"Text: {record['text']}")
    print("-" * 60)