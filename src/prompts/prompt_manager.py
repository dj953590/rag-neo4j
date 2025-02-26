from jinja2 import Environment, FileSystemLoader, Template
from typing import Dict, Any
import os

class PromptManager:
    def __init__(self, template_dir: str = None):
        if template_dir is None:
            template_dir = os.path.join(os.getcwd(), 'templates')
        self.env = Environment(loader=FileSystemLoader(template_dir))

    def render_prompt(self, template_name: str, context: Dict[str, Any]) -> str:
        template: Template = self.env.get_template(template_name)
        return template.render(context)

# Example usage
if __name__ == "__main__":
    template_name = 'similarity_check.jinja'
    context = {
        'original_prompt': 'This is a test prompt',
        'cached_prompt': 'This is a cached prompt',
    }

    prompt_manager = PromptManager()
    prompt = prompt_manager.render_prompt(template_name, context)
    print(prompt)