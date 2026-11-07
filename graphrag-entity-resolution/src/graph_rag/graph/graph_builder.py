"""
Neo4j Graph Builder module.

Creates document-specific entity mentions so entities with the same name
in different research papers are not automatically treated as one entity.
"""

import hashlib
import logging
import re
from typing import Any, Dict, List, Optional, Set

from neo4j import Driver, GraphDatabase

from graph_rag.config import settings
from graph_rag.graph.entity_extractor import GraphExtractionResult

logger = logging.getLogger(__name__)


ALLOWED_RELATION_TYPES = {
    "PROPOSES",
    "EVALUATES_ON",
    "USES_METHOD",
    "PART_OF_TOPIC",
    "RELATED_TO",
}


def normalize_name(name: str) -> str:
    """
    Normalize an entity name for matching inside the same document.

    Examples:
        "  Khaled Ahmed " -> "khaled ahmed"
        "KHALED   AHMED"  -> "khaled ahmed"
    """
    normalized = name.casefold().strip()
    return re.sub(r"\s+", " ", normalized)


def make_mention_id(
    document_name: str,
    category: str,
    entity_name: str,
) -> str:
    """
    Create a stable ID for an entity mention inside one document.

    The document name is included so that Khaled in Paper A and Khaled
    in Paper B receive different IDs.
    """
    value = "|".join(
        [
            document_name.strip(),
            category.casefold().strip(),
            normalize_name(entity_name),
        ]
    )

    return hashlib.sha256(value.encode("utf-8")).hexdigest()


class GraphBuilder:
    """Manages Neo4j connection and knowledge-graph ingestion."""

    def __init__(
        self,
        uri: Optional[str] = None,
        user: Optional[str] = None,
        password: Optional[str] = None,
    ) -> None:
        self.uri = uri or settings.NEO4J_URI
        self.user = user or settings.NEO4J_USER
        self.password = password or settings.NEO4J_PASSWORD

        self._driver: Optional[Driver] = None
        self._is_offline: bool = False
        self._checked: bool = False

    def get_driver(self) -> Optional[Driver]:
        """Create a lazy Neo4j connection with fast failure."""
        if self._is_offline:
            return None

        if self._driver is None:
            try:
                self._driver = GraphDatabase.driver(
                    self.uri,
                    auth=(self.user, self.password),
                    connection_timeout=2.0,
                    max_connection_lifetime=60.0,
                )

                self._driver.verify_connectivity()
                self._is_offline = False

            except Exception as error:
                logger.warning(
                    "Neo4j is offline or unreachable (%s): %s",
                    self.uri,
                    error,
                )

                self._driver = None
                self._is_offline = True

        return self._driver

    def is_connected(self) -> bool:
        """Check whether Neo4j is reachable."""
        if self._is_offline and self._checked:
            return False

        self._checked = True
        return self.get_driver() is not None

    def init_constraints(self) -> None:
        """
        Create constraints and indexes.

        The old name/category constraint is removed because it incorrectly
        forces entities with the same name to become one global node.
        """
        driver = self.get_driver()
        if not driver:
            return

        queries = [
            # Remove the unsafe constraint from the previous implementation.
            """
            DROP CONSTRAINT entity_name_cat IF EXISTS
            """,
            # Each document must have a unique name.
            """
            CREATE CONSTRAINT doc_name_unique IF NOT EXISTS
            FOR (d:Document)
            REQUIRE d.name IS UNIQUE
            """,
            # Each extracted mention has its own unique mention ID.
            """
            CREATE CONSTRAINT entity_mention_id_unique IF NOT EXISTS
            FOR (m:EntityMention)
            REQUIRE m.mention_id IS UNIQUE
            """,
            # Helps later when searching for possible matching entities.
            """
            CREATE INDEX entity_mention_lookup IF NOT EXISTS
            FOR (m:EntityMention)
            ON (m.normalized_name, m.category)
            """,
            """
            CREATE CONSTRAINT canonical_person_id_unique IF NOT EXISTS
            FOR (person:CanonicalPerson)
            REQUIRE person.canonical_id IS UNIQUE
            """,
        ]

        with driver.session() as session:
            for query in queries:
                try:
                    session.run(query).consume()
                except Exception as error:
                    logger.warning(
                        "Could not initialize Neo4j constraint/index: %s",
                        error,
                    )

    def add_document_node(
        self,
        document_name: str,
        title: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Create or update a research-paper Document node."""
        driver = self.get_driver()
        if not driver:
            return

        meta = metadata or {}

        query = """
        MERGE (d:Document {name: $name})

        ON CREATE SET
            d.title = $title,
            d.topic = $topic,
            d.source = $source,
            d.created_at = datetime()

        ON MATCH SET
            d.title = $title,
            d.topic = $topic,
            d.source = $source
        """

        with driver.session() as session:
            session.run(
                query,
                name=document_name,
                title=title,
                topic=meta.get("topic", "General"),
                source=meta.get("source", "upload"),
            ).consume()

    def add_entities_and_relations(
        self,
        document_name: str,
        result: GraphExtractionResult,
    ) -> None:
        """
        Add document-specific entity mentions and relationships.

        Entities with the same name in different documents are kept
        separate until an entity-resolution process links them later.
        """
        driver = self.get_driver()

        if not driver or not result.entities:
            return

        # normalized name -> mention IDs inside this document
        mention_ids: Dict[str, Set[str]] = {}

        with driver.session() as session:
            # ----------------------------------------------------------
            # 1. Create document-specific entity mentions
            # ----------------------------------------------------------
            for entity in result.entities:
                normalized_name = normalize_name(entity.name)

                mention_id = make_mention_id(
                    document_name=document_name,
                    category=entity.category,
                    entity_name=entity.name,
                )

                mention_ids.setdefault(
                    normalized_name,
                    set(),
                ).add(mention_id)

                entity_query = """
                MATCH (d:Document {name: $document_name})

                MERGE (
                    m:EntityMention {
                        mention_id: $mention_id
                    }
                )

                SET
                    m.name = $name,
                    m.normalized_name = $normalized_name,
                    m.category = $category,
                    m.description = $description,
                    m.document_name = $document_name,
                    m.external_id = $external_id,
                    m.affiliation = $affiliation

                MERGE (d)-[:MENTIONS]->(m)
                """

                try:
                    session.run(
                    entity_query,
                    document_name=document_name,
                    mention_id=mention_id,
                    name=entity.name,
                    normalized_name=normalized_name,
                    category=entity.category,
                    description=entity.description,
                    external_id=getattr(
                        entity,
                        "external_id",
                        "",
                    ),
                    affiliation=getattr(
                        entity,
                        "affiliation",
                        "",
                    ),
                ).consume()

                except Exception as error:
                    logger.warning(
                        "Could not create entity mention '%s': %s",
                        entity.name,
                        error,
                    )

            # ----------------------------------------------------------
            # 2. Create relationships only inside this document
            # ----------------------------------------------------------
            for relation in result.relations:
                relation_type = (
                    relation.relation_type.strip().upper()
                )

                # Never insert an unrestricted relationship type from LLM output.
                if relation_type not in ALLOWED_RELATION_TYPES:
                    logger.warning(
                        "Rejected unsupported relationship type: %s",
                        relation_type,
                    )
                    continue

                source_name = normalize_name(relation.source)
                target_name = normalize_name(relation.target)

                source_candidates = mention_ids.get(
                    source_name,
                    set(),
                )

                target_candidates = mention_ids.get(
                    target_name,
                    set(),
                )

                # The relationship is skipped if the name is absent or
                # ambiguous inside the current document.
                if len(source_candidates) != 1:
                    logger.warning(
                        "Could not uniquely identify source '%s' "
                        "inside document '%s'.",
                        relation.source,
                        document_name,
                    )
                    continue

                if len(target_candidates) != 1:
                    logger.warning(
                        "Could not uniquely identify target '%s' "
                        "inside document '%s'.",
                        relation.target,
                        document_name,
                    )
                    continue

                source_id = next(iter(source_candidates))
                target_id = next(iter(target_candidates))

                relation_query = f"""
                MATCH (
                    source:EntityMention {{
                        mention_id: $source_id
                    }}
                )

                MATCH (
                    target:EntityMention {{
                        mention_id: $target_id
                    }}
                )

                MERGE (
                    source
                )-[relationship:{relation_type}]->(
                    target
                )

                SET relationship.description = $description
                """

                try:
                    session.run(
                        relation_query,
                        source_id=source_id,
                        target_id=target_id,
                        description=relation.description,
                    ).consume()

                except Exception as error:
                    logger.warning(
                        "Could not create relationship "
                        "'%s' -> '%s': %s",
                        relation.source,
                        relation.target,
                        error,
                    )

    def get_graph_stats(self) -> Dict[str, Any]:
        """Return Neo4j graph statistics."""
        driver = self.get_driver()

        if not driver:
            return {
                "nodes": 0,
                "edges": 0,
                "documents": 0,
                "entity_mentions": 0,
                "status": "offline",
            }

        try:
            with driver.session() as session:
                document_count = session.run(
                    """
                    MATCH (d:Document)
                    RETURN count(d) AS count
                    """
                ).single()["count"]

                node_count = session.run(
                    """
                    MATCH (node)
                    RETURN count(node) AS count
                    """
                ).single()["count"]

                edge_count = session.run(
                    """
                    MATCH ()-[relationship]->()
                    RETURN count(relationship) AS count
                    """
                ).single()["count"]

                mention_count = session.run(
                    """
                    MATCH (mention:EntityMention)
                    RETURN count(mention) AS count
                    """
                ).single()["count"]

                return {
                    "nodes": node_count,
                    "edges": edge_count,
                    "documents": document_count,
                    "entity_mentions": mention_count,
                    "status": "online",
                }

        except Exception as error:
            logger.warning(
                "Could not retrieve Neo4j statistics: %s",
                error,
            )

            return {
                "nodes": 0,
                "edges": 0,
                "documents": 0,
                "entity_mentions": 0,
                "status": "error",
            }

    def close(self) -> None:
        """Close the Neo4j connection."""
        if self._driver:
            self._driver.close()
            self._driver = None


graph_builder = GraphBuilder()