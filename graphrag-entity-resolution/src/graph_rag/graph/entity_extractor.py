"""
Entity Extractor module.

Uses local Ollama LLM to extract structured entities and relationships
from research-paper text chunks.
"""

import json
import logging
import re
from dataclasses import dataclass, field
from typing import List, Optional

from graph_rag.llm.ollama_client import ollama_client

logger = logging.getLogger(__name__)


ALLOWED_CATEGORIES = {
    "Author",
    "Topic",
    "Method",
    "Dataset",
    "Finding",
}

ALLOWED_RELATION_TYPES = {
    "PROPOSES",
    "EVALUATES_ON",
    "USES_METHOD",
    "PART_OF_TOPIC",
    "RELATED_TO",
}


@dataclass
class ExtractedEntity:
    """An entity discovered in the text."""

    name: str
    category: str
    description: str = ""
    external_id: str = ""
    affiliation: str = ""


@dataclass
class ExtractedRelation:
    """A semantic relationship between two entities."""

    source: str
    target: str
    relation_type: str
    description: str = ""


@dataclass
class GraphExtractionResult:
    """Entities and relationships extracted from one chunk."""

    entities: List[ExtractedEntity] = field(default_factory=list)
    relations: List[ExtractedRelation] = field(default_factory=list)


EXTRACTION_PROMPT_TEMPLATE = """
You are an expert knowledge-graph engineer analyzing a scientific
research paper.

Document title:
{document_title}

Extract the important scientific entities and relationships from the
text.

Entity categories allowed:

- "Author": a named researcher or person
- "Topic": a research area or scientific subject
- "Method": a model, technique, architecture, algorithm, system,
  project, or technology solution
- "Dataset": a benchmark or dataset
- "Finding": a key result, conclusion, or scientific insight

Mandatory author extraction rules:

1. Extract every explicitly named researcher or person as an "Author".
2. A researcher mentioned in the title, metadata, abstract, or body
   must not be omitted.
3. Preserve the complete name. For example, use
   "Khaled Mahmoud Hassan", not only "Khaled".
4. Remove titles such as "Dr." and "Professor" from the entity name.
5. If a researcher identifier is present, copy it exactly into
   "external_id".
6. Examples of identifiers include ORCID, Researcher ID, and
   Synthetic Researcher ID.
7. If an affiliation is present, copy it exactly into "affiliation".
8. Do not invent missing identifiers, affiliations, people, or facts.
9. If an identifier or affiliation is absent, return an empty string.
10. Do not merge two people merely because their names are similar.

Scientific entity rules:

1. Use "Method" for a named system or technology solution such as
   CivicGrid.
2. Use "Topic" for a general research subject such as Knowledge Graphs.
3. Avoid duplicate entities in the same response.
4. Relationship source and target values must exactly match entity
   names included in the entities list.

Relation types allowed:

- "PROPOSES": Author or Method -> Method
- "EVALUATES_ON": Method -> Dataset
- "USES_METHOD": Method -> Method
- "PART_OF_TOPIC": Method or Dataset -> Topic
- "RELATED_TO": Entity -> Entity

Respond only with valid JSON using this exact structure:

{{
  "entities": [
    {{
      "name": "Khaled Mahmoud Hassan",
      "category": "Author",
      "description": "Researcher working on knowledge graphs",
      "external_id": "TEST-KMH-001",
      "affiliation": "Center for Responsible Computing, Alexandria"
    }},
    {{
      "name": "CivicGrid",
      "category": "Method",
      "description": "Municipal infrastructure maintenance solution",
      "external_id": "",
      "affiliation": ""
    }}
  ],
  "relations": [
    {{
      "source": "Khaled Mahmoud Hassan",
      "target": "CivicGrid",
      "relation_type": "PROPOSES",
      "description": "Served as the principal architect"
    }}
  ]
}}

Text to analyze:

{text}
"""


def normalize_name(name: str) -> str:
    """Normalize an entity name for deduplication."""

    normalized = name.casefold().strip()
    return re.sub(r"\s+", " ", normalized)


def remove_person_title(name: str) -> str:
    """Remove titles such as Dr. and Professor from a person's name."""

    return re.sub(
        r"^(?:dr\.?|prof\.?|professor)\s+",
        "",
        name.strip(),
        flags=re.IGNORECASE,
    ).strip()


class EntityExtractor:
    """Extracts entities and relationships using Ollama."""

    def __init__(self, client=None) -> None:
        self.client = client or ollama_client

    def extract(
        self,
        text: str,
        document_title: Optional[str] = None,
    ) -> GraphExtractionResult:
        """Extract entities and relationships from one text chunk."""

        # Increased from 1,500 characters to reduce the chance that
        # researcher information is removed before extraction.
        prompt_text = text[:4000]

        prompt = EXTRACTION_PROMPT_TEMPLATE.format(
            document_title=document_title or "Unknown document",
            text=prompt_text,
        )

        try:
            raw_response = self.client.generate(
                prompt=prompt,
                system_prompt=(
                    "You are a JSON-only entity extraction engine. "
                    "Return valid JSON without markdown or explanations."
                ),
                temperature=0.1,
            )

            result = self._parse_json_response(raw_response)

            # If the LLM returned no entities, try rule-based extraction.
            if not result.entities:
                logger.warning(
                    "LLM returned no valid entities. "
                    "Using rule-based extraction."
                )

                return self._rule_based_fallback(
                    text=text,
                    document_title=document_title,
                )

            return result

        except Exception as error:
            logger.warning(
                "Entity extraction failed: %s. "
                "Using rule-based extraction.",
                error,
            )

            return self._rule_based_fallback(
                text=text,
                document_title=document_title,
            )

    def _parse_json_response(
        self,
        raw_response: str,
    ) -> GraphExtractionResult:
        """Parse and validate the JSON returned by the LLM."""

        json_match = re.search(
            r"\{.*\}",
            raw_response,
            re.DOTALL,
        )

        if not json_match:
            logger.warning("No JSON object found in LLM response.")
            return GraphExtractionResult()

        try:
            data = json.loads(json_match.group(0))
        except json.JSONDecodeError as error:
            logger.warning("Invalid JSON returned by LLM: %s", error)
            return GraphExtractionResult()

        entity_map = {}

        for item in data.get("entities", []):
            name = str(item.get("name", "")).strip()
            category = str(item.get("category", "")).strip()
            description = str(
                item.get("description", "")
            ).strip()
            external_id = str(
                item.get("external_id", "")
            ).strip()
            affiliation = str(
                item.get("affiliation", "")
            ).strip()

            if not name or len(name) < 2:
                continue

            if category not in ALLOWED_CATEGORIES:
                logger.warning(
                    "Rejected unsupported entity category: %s",
                    category,
                )
                continue

            if category == "Author":
                name = remove_person_title(name)

            # Keep identity evidence inside the description too, because
            # the current GraphBuilder already stores description.
            identity_details = []

            if external_id:
                identity_details.append(
                    f"External ID: {external_id}"
                )

            if affiliation:
                identity_details.append(
                    f"Affiliation: {affiliation}"
                )

            if identity_details:
                details_text = "; ".join(identity_details)

                if description:
                    description = (
                        f"{description}; {details_text}"
                    )
                else:
                    description = details_text

            key = (
                normalize_name(name),
                category,
            )

            # Deduplicate repeated entities in the same chunk.
            if key not in entity_map:
                entity_map[key] = ExtractedEntity(
                    name=name,
                    category=category,
                    description=description,
                    external_id=external_id,
                    affiliation=affiliation,
                )
            else:
                existing = entity_map[key]

                if not existing.external_id and external_id:
                    existing.external_id = external_id

                if not existing.affiliation and affiliation:
                    existing.affiliation = affiliation

                if not existing.description and description:
                    existing.description = description

        entities = list(entity_map.values())
        valid_entity_names = {
            entity.name for entity in entities
        }

        relations: List[ExtractedRelation] = []

        for item in data.get("relations", []):
            source = str(item.get("source", "")).strip()
            target = str(item.get("target", "")).strip()
            relation_type = str(
                item.get("relation_type", "")
            ).strip().upper()
            description = str(
                item.get("description", "")
            ).strip()

            source = remove_person_title(source)
            target = remove_person_title(target)

            if not source or not target or source == target:
                continue

            if relation_type not in ALLOWED_RELATION_TYPES:
                logger.warning(
                    "Rejected unsupported relationship type: %s",
                    relation_type,
                )
                continue

            # GraphBuilder requires relationship endpoints to correspond
            # to extracted entities.
            if (
                source not in valid_entity_names
                or target not in valid_entity_names
            ):
                logger.warning(
                    "Rejected relationship with missing endpoint: "
                    "%s -> %s",
                    source,
                    target,
                )
                continue

            relations.append(
                ExtractedRelation(
                    source=source,
                    target=target,
                    relation_type=relation_type,
                    description=description,
                )
            )

        return GraphExtractionResult(
            entities=entities,
            relations=relations,
        )

    def _rule_based_fallback(
        self,
        text: str,
        document_title: Optional[str],
    ) -> GraphExtractionResult:
        """Fallback extraction if the LLM is unavailable."""

        entities: List[ExtractedEntity] = []
        relations: List[ExtractedRelation] = []
        entity_keys = set()

        # Find names preceded by Dr., Prof., or Professor.
        person_pattern = re.compile(
            r"\b(?:Dr\.?|Prof\.?|Professor)\s+"
            r"([A-Z][A-Za-z'-]+"
            r"(?:\s+[A-Z][A-Za-z'-]+){1,4})"
        )

        for match in person_pattern.finditer(text):
            person_name = match.group(1).strip()
            key = (
                normalize_name(person_name),
                "Author",
            )

            if key in entity_keys:
                continue

            entity_keys.add(key)

            entities.append(
                ExtractedEntity(
                    name=person_name,
                    category="Author",
                    description=(
                        "Researcher identified by rule-based fallback"
                    ),
                )
            )

        # Find common capitalized model acronyms.
        acronyms = set(
            re.findall(r"\b[A-Z][A-Z0-9-]{1,10}\b", text)
        )

        excluded_acronyms = {
            "THE",
            "FOR",
            "AND",
            "WITH",
            "THAT",
            "THIS",
        }

        for acronym in acronyms:
            if acronym in excluded_acronyms:
                continue

            key = (
                normalize_name(acronym),
                "Method",
            )

            if key in entity_keys:
                continue

            entity_keys.add(key)

            entities.append(
                ExtractedEntity(
                    name=acronym,
                    category="Method",
                    description=(
                        "Model or acronym identified by "
                        "rule-based fallback"
                    ),
                )
            )

        return GraphExtractionResult(
            entities=entities,
            relations=relations,
        )


entity_extractor = EntityExtractor()