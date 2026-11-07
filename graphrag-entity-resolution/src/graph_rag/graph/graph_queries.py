"""
Graph Queries module.

Provides Cypher query helpers for hybrid graph retrieval.
Searches multi-hop entity neighborhoods and document relationship networks.
"""

from typing import Any, Dict, List, Optional
from graph_rag.graph.graph_builder import graph_builder


class GraphQueries:
    """Cypher query helpers for knowledge graph retrieval."""

    @staticmethod
    def search_entities(keyword: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Find entities whose name contains the search keyword."""
        driver = graph_builder.get_driver()
        if not driver:
            return []

        query = """
        MATCH (e:Entity)
        WHERE toLower(e.name) CONTAINS toLower($kw)
        RETURN e.name AS name, e.category AS category, e.description AS description
        LIMIT $limit
        """
        results = []
        try:
            with driver.session() as session:
                for record in session.run(query, kw=keyword, limit=limit):
                    results.append(
                        {
                            "name": record["name"],
                            "category": record["category"],
                            "description": record["description"] or "",
                        }
                    )
        except Exception:
            pass
        return results

    @staticmethod
    def find_entity_subgraph(entity_name: str, max_depth: int = 1) -> List[Dict[str, Any]]:
        """Retrieve 1-hop connected neighbors and relationships for an entity."""
        driver = graph_builder.get_driver()
        if not driver:
            return []

        query = """
        MATCH (e:Entity {name: $name})-[r]-(neighbor:Entity)
        RETURN e.name AS source, type(r) AS relation, neighbor.name AS target, neighbor.category AS target_category
        LIMIT 15
        """
        relationships = []
        try:
            with driver.session() as session:
                for record in session.run(query, name=entity_name):
                    relationships.append(
                        {
                            "source": record["source"],
                            "relation": record["relation"],
                            "target": record["target"],
                            "target_category": record["target_category"],
                        }
                    )
        except Exception:
            pass
        return relationships

    @staticmethod
    def get_related_documents_for_entities(entity_names: List[str]) -> List[str]:
        """Find papers that mention any of the specified entities."""
        driver = graph_builder.get_driver()
        if not driver or not entity_names:
            return []

        query = """
        MATCH (d:Document)-[:MENTIONS]->(e:Entity)
        WHERE e.name IN $names
        RETURN DISTINCT d.name AS document_name, d.title AS title
        LIMIT 10
        """
        docs = []
        try:
            with driver.session() as session:
                for record in session.run(query, names=entity_names):
                    docs.append(record["document_name"])
        except Exception:
            pass
        return docs


graph_queries = GraphQueries()
