import json
from typing import List, Dict
from pathlib import Path

base_dir = Path(__file__).parent
executed_date_prompts = (base_dir / '..' / 'prompts' / 'templates' / 'executed_date.json').resolve()

def load_prompts_from_json(file_path: str) -> List[Dict[str, any]]:
    with open(file_path, "r", encoding="utf-8") as f:
        prompts = json.load(f)

    # Validate fields
    for p in prompts:
        required = ["name", "description", "examples", "document_types"]
        for field in required:
            if field not in p:
                raise ValueError(f"Missing required field '{field}' in prompt: {p}")
        if not isinstance(p["document_types"], list):
            raise TypeError(f"'document_types' should be a list in prompt: {p['name']}")

    return prompts
if __name__ == "__main__":
    prompt_list = load_prompts_from_json(str(executed_date_prompts))
    for prompt in prompt_list:
        print(f"Name: {prompt['name']}")
        print(f"Description: {prompt['description']}")
        print("Examples:")
        for ex in prompt["examples"]:
            print(f" - {ex}")
        print("Applicable Document Types:", ", ".join(prompt["document_types"]))
        print("-" * 50)
