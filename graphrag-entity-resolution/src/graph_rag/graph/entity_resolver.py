"""
Entity-resolution module.

Links document-specific Author mentions to canonical people when
sufficient identity evidence is available.
"""

import hashlib
import logging
import re

from graph_rag.graph.entity_extractor import GraphExtractionResult
from graph_rag.graph.graph_builder import graph_builder

logger = logging.getLogger(__name__)


def normalize_value(value: str) -> str:
    """Normalize names and affiliations for identity comparison."""
    normalized = value.casefold().strip()
    return re.sub(r"\s+", " ", normalized)


def make_name_affiliation_id(
    name: str,
    affiliation: str,
) -> str:
    """Create a stable identity key from full name and affiliation."""
    value = (
        f"{normalize_value(name)}|"
        f"{normalize_value(affiliation)}"
    )

    digest = hashlib.sha256(
        value.encode("utf-8")
    ).hexdigest()

    return f"name-affiliation:{digest}"


class EntityResolver:
    """Resolves Author mentions into canonical people."""

    def __init__(self, graph=None) -> None:
        self.graph = graph or graph_builder

    def resolve_document_authors(
        self,
        document_name: str,
        result: GraphExtractionResult,
    ) -> int:
        """
        Resolve Authors extracted from one chunk.

        Resolution rules:
        1. Exact external ID -> automatic match, confidence 1.0.
        2. Full name plus affiliation -> automatic match,
           confidence 0.90.
        3. Name alone -> leave unresolved.
        """
        driver = self.graph.get_driver()

        if not driver:
            return 0

        authors = [
            entity
            for entity in result.entities
            if entity.category == "Author"
        ]

        resolved_count = 0

        with driver.session() as session:
            for author in authors:
                name = author.name.strip()
                normalized_name = normalize_value(name)

                external_id = getattr(
                    author,
                    "external_id",
                    "",
                ).strip()

                affiliation = getattr(
                    author,
                    "affiliation",
                    "",
                ).strip()

                # Strongest evidence: stable researcher identifier.
                if external_id:
                    canonical_id = (
                        f"external:"
                        f"{normalize_value(external_id)}"
                    )

                    resolution_method = "external_id"
                    confidence = 1.0

                # Second-best evidence: full name and affiliation.
                elif (
                    affiliation
                    and len(normalized_name.split()) >= 2
                ):
                    canonical_id = make_name_affiliation_id(
                        name=name,
                        affiliation=affiliation,
                    )

                    resolution_method = "name_and_affiliation"
                    confidence = 0.90

                else:
                    logger.info(
                        "Author left unresolved because identity "
                        "evidence is insufficient: %s",
                        name,
                    )
                    continue

                query = """
                MATCH (
                    d:Document {name: $document_name}
                )-[:MENTIONS]->(
                    mention:EntityMention {
                        category: "Author"
                    }
                )

                WHERE mention.normalized_name = $normalized_name

                MERGE (
                    person:CanonicalPerson {
                        canonical_id: $canonical_id
                    }
                )

                ON CREATE SET
                    person.name = $name,
                    person.normalized_name = $normalized_name,
                    person.created_at = datetime()

                SET
                    person.name = $name,
                    person.external_id =
                        CASE
                            WHEN $external_id <> ""
                            THEN $external_id
                            ELSE coalesce(
                                person.external_id,
                                ""
                            )
                        END,
                    person.affiliation =
                        CASE
                            WHEN $affiliation <> ""
                            THEN $affiliation
                            ELSE coalesce(
                                person.affiliation,
                                ""
                            )
                        END,
                    person.updated_at = datetime()

                MERGE (
                    mention
                )-[resolution:REFERS_TO]->(
                    person
                )

                SET
                    resolution.confidence = $confidence,
                    resolution.resolution_method =
                        $resolution_method,
                    resolution.resolved_at = datetime()

                RETURN count(mention) AS resolved
                """

                record = session.run(
                    query,
                    document_name=document_name,
                    normalized_name=normalized_name,
                    canonical_id=canonical_id,
                    name=name,
                    external_id=external_id,
                    affiliation=affiliation,
                    confidence=confidence,
                    resolution_method=resolution_method,
                ).single()

                if record:
                    resolved_count += int(
                        record["resolved"]
                    )

        return resolved_count


entity_resolver = EntityResolver()