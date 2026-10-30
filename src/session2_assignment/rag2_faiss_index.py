import json
from pathlib import Path

import faiss
import numpy as np


BASE_DIR = Path(__file__).resolve().parent

EMBEDDINGS_FILE = (
    BASE_DIR / "rag2_document_embeddings.json"
)

INDEX_FILE = (
    BASE_DIR / "rag2_faiss_index.bin"
)

METADATA_FILE = (
    BASE_DIR / "rag2_faiss_metadata.json"
)


def main() -> None:
    if not EMBEDDINGS_FILE.exists():
        raise FileNotFoundError(
            f"Embeddings file was not found: "
            f"{EMBEDDINGS_FILE}"
        )

    with EMBEDDINGS_FILE.open(
        "r",
        encoding="utf-8",
    ) as file:
        data = json.load(file)

    records = data["chunks"]

    if not records:
        raise ValueError(
            "The embeddings file contains no chunks."
        )

    vectors = np.array(
        [
            record["embedding"]
            for record in records
        ],
        dtype="float32",
    )

    # Normalize vectors so inner product
    # becomes cosine similarity.
    faiss.normalize_L2(vectors)

    dimensions = vectors.shape[1]

    index = faiss.IndexFlatIP(dimensions)
    index.add(vectors)

    faiss.write_index(
        index,
        str(INDEX_FILE),
    )

    metadata = []

    for record in records:
        metadata.append(
            {
                "source": record["source"],
                "page": record["page"],
                "chunk": record["chunk"],
                "text": record["text"],
            }
        )

    with METADATA_FILE.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metadata,
            file,
            ensure_ascii=False,
            indent=2,
        )

    print(
        "FAISS index created successfully!"
    )
    print(f"Stored vectors: {index.ntotal}")
    print(
        f"Vector dimensions: {dimensions}"
    )
    print(f"Index saved to: {INDEX_FILE.resolve()}")
    print(f"Metadata saved to: {METADATA_FILE.resolve()}")


if __name__ == "__main__":
    main()