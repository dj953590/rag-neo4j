# 2. Function to fetch states from PostgreSQL
from enum import Enum
from datetime import datetime
from typing import Tuple

from src.storage.db.sql.pgdb import PGDB


def get_document_codes():
    try:
        pg = PGDB()
        result = pg.select("document_state")
        return [(row.code, row.description) for row in result]

    except Exception as e:
        print(f"Error fetching document states: {e}")
        return []


def generate_documented_enum_class(states: Tuple[str, str]):
    """Generate Enum class with state descriptions"""
    enum_code = f'''# Auto-generated on {datetime.now().astimezone().tzinfo}
from enum import Enum

class DocumentState(Enum):
    """
    Document states with descriptions from DOCUMENT_STATE table
    
    Attributes:
        value (str): The actual state value from the database
        description (str): Human-readable description of the state
    """
    
    def __new__(cls, value, description):
        obj = object.__new__(cls)
        obj._value_ = value
        obj.description = description
        return obj

'''

    for code, description in states:
        # Clean and format state name
        name = code.upper().replace(' ', '_').replace('-', '_')
        safe_desc = description.replace('"', "'")  # Escape quotes

        enum_code += f'    {name} = ("{code}", "{safe_desc}")\n'

    return enum_code


def save_enum_file(content, filename="document_states.py"):
    with open(filename, 'w') as f:
        f.write(content)
    print(f"Enum file generated: {filename}")


if __name__ == "__main__":

    document_states = get_document_codes()

    if not document_states:
        print("No document states found in the database!")
        exit(1)

    enum_content = generate_documented_enum_class(document_states)
    save_enum_file(enum_content)
