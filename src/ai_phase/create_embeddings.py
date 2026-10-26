import json
import os
from pathlib import Path
import requests
from dotenv import load_dotenv
from pypdf import PdfReader


load_dotenv()

API_KEY = os.getenv("embedding_model_key")

if not API_KEY:
    raise ValueError("OPENROUTER_API_KEY is missing from .env")


PDF_PATH = Path("ai-rag-chunking-sample.pdf")
OUTPUT_PATH = Path("document_embeddings.json")

MODEL = "nvidia/nemotron-3-embed-1b:free"

CHUNK_SIZE = 1200
CHUNK_OVERLAP = 200
BATCH_SIZE = 16


def split_text(text: str) -> list[str]:
    """Split text into overlapping chunks"""

    chunks = []
    step = CHUNK_SIZE - CHUNK_OVERLAP

    for start in range(0, len(text), step):
        chunk = text[start:start + CHUNK_SIZE].strip()

        if chunk:
            chunks.append(chunk)

    return chunks


def extract_pdf_chunks(pdf_path: Path) -> list[dict]:
    """Extract text from every page and create chunks"""

    reader = PdfReader(pdf_path)
    records = []

    for page_number, page in enumerate(reader.pages, start=1):
        page_text = page.extract_text() or ""

        for chunk_number, chunk in enumerate(
            split_text(page_text),
            start=1,
        ):
            records.append(
                {
                    "page": page_number,
                    "chunk": chunk_number,
                    "text": chunk,
                }
            )

    return records


def create_embeddings(texts: list[str]) -> list[list[float]]:
    """Send text batches to OpenRouter."""

    all_embeddings = []

    for start in range(0, len(texts), BATCH_SIZE):
        batch = texts[start:start + BATCH_SIZE]

        response = requests.post(
            "https://openrouter.ai/api/v1/embeddings",
            headers={
                "Authorization": f"Bearer {API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": MODEL,
                "input": batch,
            },
            timeout=120,
        )

        if not response.ok:
            raise RuntimeError(
                f"OpenRouter error {response.status_code}: "
                f"{response.text}"
            )

        result = response.json()

        # Ensure embeddings match the original input order
        ordered_data = sorted(
            result["data"],
            key=lambda item: item["index"],
        )

        batch_embeddings = [
            item["embedding"]
            for item in ordered_data
        ]

        all_embeddings.extend(batch_embeddings)

        print(
            f"Embedded {min(start + BATCH_SIZE, len(texts))}"
            f"/{len(texts)} chunks"
        )

    return all_embeddings


if not PDF_PATH.exists():
    raise FileNotFoundError(f"PDF not found: {PDF_PATH}")


records = extract_pdf_chunks(PDF_PATH)

if not records:
    raise ValueError(
        "No text was extracted. The PDF may require OCR."
    )

texts = [record["text"] for record in records]
embeddings = create_embeddings(texts)

if len(records) != len(embeddings):
    raise ValueError("Number of chunks and embeddings does not match")


for record, embedding in zip(records, embeddings):
    record["embedding"] = embedding


output_data = {
    "source": PDF_PATH.name,
    "embedding_model": MODEL,
    "chunk_size": CHUNK_SIZE,
    "chunk_overlap": CHUNK_OVERLAP,
    "chunks": records,
}


OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

with OUTPUT_PATH.open("w", encoding="utf-8") as file:
    json.dump(
        output_data,
        file,
        ensure_ascii=False,
    )


print("\nCompleted successfully!")
print(f"Chunks: {len(records)}")
print(f"Vector dimensions: {len(embeddings[0])}")
print(f"Saved to: {OUTPUT_PATH}")