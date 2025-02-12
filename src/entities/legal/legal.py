import os

from pydantic import BaseModel
from typing import List
from src.utils.log import logger


class LegalEntityType(BaseModel):
    entity_type: str
    description: str

    def dump(self):
        return {
            "entity_type": self.entity_type,
            "description": self.description
        }


class LegalEntitySchema(BaseModel):
    legal_entities: List[LegalEntityType]

    def dump(self):
        return [entity.dump() for entity in self.legal_entities]


def load_legal_entities(file_path: str) -> LegalEntitySchema:
    """
    Reads Agreement.txt and loads entity types with descriptions into a LegalEntitySchema.

    :param file_path: Path to the Agreement.txt file.
    :return: LegalEntitySchema containing entity types and descriptions.
    """
    legal_entities = []

    with open(file_path, "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:  # Skip empty lines or malformed entries
                continue

            try:
                # Parse entity type and description
                entity_type, description = line.split(":", 1)
                entity_type = entity_type.strip()
                description = description.strip()

                # Create LegalEntityType instance
                legal_entities.append(LegalEntityType(entity_type=entity_type, description=description))

            except ValueError:
                logger.info(f"Skipping malformed line: {line}")

    return LegalEntitySchema(legal_entities=legal_entities)


if __name__ == "__main__":
    # Example usage
    file_path = os.path.join(os.path.dirname(__file__), "data\\Agreement.txt")  # Update with the actual path
    legal_entities_schema = load_legal_entities(file_path)
    print(legal_entities_schema.model_dump_json(indent=4))
    print(legal_entities_schema.dump())
