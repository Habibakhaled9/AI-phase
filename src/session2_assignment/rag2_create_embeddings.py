import json
import os
from pathlib import Path

import requests
from dotenv import load_dotenv
from pypdf import PdfReader


load_dotenv()

API_KEY = os.getenv("embedding_model_key")

if not API_KEY:
    raise ValueError(
        "embedding_model_key is missing from the .env file"
    )


BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent.parent

DOCUMENTS_DIR = PROJECT_ROOT / "documents"

OUTPUT_PATH = (
    BASE_DIR / "rag2_document_embeddings.json"
)

EMBEDDING_MODEL = "nvidia/nemotron-3-embed-1b:free"

CHUNK_SIZE = 1200
CHUNK_OVERLAP = 200
BATCH_SIZE = 16


def split_text(text: str) -> list[str]:
    """Split text into overlapping character chunks."""

    chunks = []
    step = CHUNK_SIZE - CHUNK_OVERLAP

    for start in range(0, len(text), step):
        chunk = text[start:start + CHUNK_SIZE].strip()

        if chunk:
            chunks.append(chunk)

    return chunks


def extract_pdf_chunks(pdf_path: Path) -> list[dict]:
    """Extract and chunk text from one PDF."""

    reader = PdfReader(pdf_path)
    records = []

    for page_number, page in enumerate(
        reader.pages,
        start=1,
    ):
        page_text = page.extract_text() or ""

        for chunk_number, chunk in enumerate(
            split_text(page_text),
            start=1,
        ):
            records.append(
                {
                    "source": pdf_path.name,
                    "page": page_number,
                    "chunk": chunk_number,
                    "text": chunk,
                }
            )

    return records


def extract_all_documents(
    documents_directory: Path,
) -> list[dict]:
    """Extract chunks from every PDF in the directory."""

    pdf_files = sorted(
        documents_directory.glob("*.pdf")
    )

    if not pdf_files:
        raise FileNotFoundError(
            f"No PDF files were found in: "
            f"{documents_directory.resolve()}"
        )

    all_records = []

    for pdf_path in pdf_files:
        pdf_records = extract_pdf_chunks(pdf_path)
        all_records.extend(pdf_records)

        print(
            f"Extracted {len(pdf_records)} chunks "
            f"from {pdf_path.name}"
        )

    return all_records


def create_embeddings(
    texts: list[str],
) -> list[list[float]]:
    """Create embeddings using OpenRouter."""

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
                "model": EMBEDDING_MODEL,
                "input": batch,
            },
            timeout=120,
        )

        if not response.ok:
            raise RuntimeError(
                f"OpenRouter error "
                f"{response.status_code}: "
                f"{response.text}"
            )

        result = response.json()

        ordered_data = sorted(
            result["data"],
            key=lambda item: item["index"],
        )

        batch_embeddings = [
            item["embedding"]
            for item in ordered_data
        ]

        all_embeddings.extend(batch_embeddings)

        embedded_count = min(
            start + BATCH_SIZE,
            len(texts),
        )

        print(
            f"Embedded {embedded_count}/"
            f"{len(texts)} chunks"
        )

    return all_embeddings


def main() -> None:
    if not DOCUMENTS_DIR.exists():
        raise FileNotFoundError(
            f"Documents folder was not found: "
            f"{DOCUMENTS_DIR.resolve()}"
        )

    records = extract_all_documents(
        DOCUMENTS_DIR
    )

    if not records:
        raise ValueError(
            "No text was extracted from the PDFs."
        )

    texts = [
        record["text"]
        for record in records
    ]

    embeddings = create_embeddings(texts)

    if len(records) != len(embeddings):
        raise ValueError(
            "The number of chunks and embeddings "
            "does not match."
        )

    for record, embedding in zip(
        records,
        embeddings,
    ):
        record["embedding"] = embedding

    source_names = sorted(
        {
            record["source"]
            for record in records
        }
    )

    output_data = {
        "sources": source_names,
        "embedding_model": EMBEDDING_MODEL,
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
        "chunks": records,
    }

    with OUTPUT_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            output_data,
            file,
            ensure_ascii=False,
        )

    print("\nCompleted successfully!")
    print(f"Documents: {len(source_names)}")
    print(f"Chunks: {len(records)}")
    print(
        f"Vector dimensions: "
        f"{len(embeddings[0])}"
    )
    print(f"Saved to: {OUTPUT_PATH.resolve()}")


if __name__ == "__main__":
    main()