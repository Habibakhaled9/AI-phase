"""
Remove the two Khaled test documents from ChromaDB and metadata.csv.

This script does not delete the physical PDF files and does not affect
the other research papers.
"""

import shutil
import sys
from datetime import datetime
from pathlib import Path

# Add src to Python path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from graph_rag.embeddings.vector_store import vector_store
from graph_rag.ingestion.metadata_manager import metadata_manager


TARGET_DOCUMENTS = {
    "khaled-paper-a-academic-background-v2.pdf",
    "khaled-paper-b-technology-project-v2.pdf",
}


def clean_filename(value: str) -> str:
    """Return a lowercase filename without its directory path."""
    return Path(str(value)).name.strip().lower()


def main() -> None:
    metadata_path = metadata_manager.metadata_path
    dataframe = metadata_manager.load_metadata()

    target_names = {
        clean_filename(name)
        for name in TARGET_DOCUMENTS
    }

    matching_mask = dataframe["local_pdf"].apply(
        lambda value: clean_filename(value) in target_names
    )

    matching_rows = dataframe[matching_mask]

    print("=" * 65)
    print("Khaled Test Document Cleanup")
    print(f"Metadata file: {metadata_path}")
    print(f"Matching metadata rows: {len(matching_rows)}")
    print("=" * 65)

    if not matching_rows.empty:
        print("\nDocuments found in metadata.csv:")

        for _, row in matching_rows.iterrows():
            print(
                f"- {row.get('local_pdf', '')} "
                f"(hash: {row.get('file_hash', '')})"
            )

        # Create a timestamped backup before changing metadata.csv.
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

        backup_path = metadata_path.with_name(
            f"{metadata_path.stem}.backup-{timestamp}"
            f"{metadata_path.suffix}"
        )

        shutil.copy2(metadata_path, backup_path)

        print(f"\nMetadata backup created: {backup_path}")

    # Delete ChromaDB chunks for the original document names.
    print("\nDeleting ChromaDB chunks:")

    for document_name in TARGET_DOCUMENTS:
        vector_store.delete_document_chunks(document_name)
        print(f"- Deleted ChromaDB chunks for: {document_name}")

    # Remove only the matching metadata rows.
    if not matching_rows.empty:
        updated_dataframe = dataframe[~matching_mask].copy()

        temporary_path = metadata_path.with_suffix(".tmp")

        updated_dataframe.to_csv(
            temporary_path,
            index=False,
        )

        temporary_path.replace(metadata_path)

        print(
            f"\nRemoved {len(matching_rows)} row(s) "
            "from metadata.csv."
        )
    else:
        print(
            "\nNo matching metadata rows were found. "
            "ChromaDB cleanup was still attempted."
        )

    # Verification
    remaining_dataframe = metadata_manager.load_metadata()

    remaining_mask = remaining_dataframe["local_pdf"].apply(
        lambda value: clean_filename(value) in target_names
    )

    remaining_count = int(remaining_mask.sum())

    print("\nVerification:")
    print(f"- Remaining matching metadata rows: {remaining_count}")
    print(
        f"- Total ChromaDB chunks after cleanup: "
        f"{vector_store.get_total_chunks()}"
    )

    if remaining_count == 0:
        print("\nCleanup completed successfully.")
    else:
        print(
            "\nCleanup did not fully complete. "
            "Restore the backup and inspect metadata.csv."
        )


if __name__ == "__main__":
    main()