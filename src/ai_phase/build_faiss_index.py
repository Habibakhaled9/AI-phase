import json
from pathlib import Path

import faiss
import numpy as np


EMBEDDINGS_FILE = Path("document_embeddings.json")
INDEX_FILE = Path("faiss_index.bin")
METADATA_FILE = Path("faiss_metadata.json")


with EMBEDDINGS_FILE.open("r", encoding="utf-8") as file:
    data = json.load(file)


records = data["chunks"]

vectors = np.array(
    [record["embedding"] for record in records],
    dtype="float32",
)

# Normalize vectors so inner product becomes cosine similarity
faiss.normalize_L2(vectors)

dimensions = vectors.shape[1]

# Exact cosine-similarity index
index = faiss.IndexFlatIP(dimensions)

index.add(vectors)

# Save the FAISS vector index
faiss.write_index(index, str(INDEX_FILE))


# Store text and metadata separately
metadata = []

for record in records:
    metadata.append(
        {
            "page": record["page"],
            "chunk": record["chunk"],
            "text": record["text"],
        }
    )


with METADATA_FILE.open("w", encoding="utf-8") as file:
    json.dump(
        metadata,
        file,
        ensure_ascii=False,
        indent=2,
    )


print("FAISS index created successfully!")
print(f"Stored vectors: {index.ntotal}")
print(f"Vector dimensions: {dimensions}")
print(f"Index saved to: {INDEX_FILE}")
print(f"Text metadata saved to: {METADATA_FILE}")