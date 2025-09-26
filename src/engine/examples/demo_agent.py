import os
from typing import ClassVar, Any, Self, TYPE_CHECKING
from pydantic import BaseModel, Field
from pydantic_ai import Agent
from pydantic_ai.models import Model
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai  import ModelSettings
from pydantic_ai.messages import ModelMessage as Message, UserPromptPart as UserMessage, SystemPromptPart as SystemMessage
from openai import OpenAI


# --- 1. Define the Custom Model to wrap the OpenAI API call ---

class MyCustomOpenAIModel(Model):
    # Class variable to hold a cached OpenAI client instance
    # This prevents re-initialization on every call
    _client: ClassVar[OpenAI | None] = None

    # Provider is used for configuration, we can reuse the official one
    provider: OpenAIProvider = Field(default_factory=OpenAIProvider)

    # The actual OpenAI model name to use (e.g., gpt-3.5-turbo, gpt-4o)
    openai_model_name: str

    if TYPE_CHECKING:
        # Define the constructor signature for type checking
        def __init__(self, openai_model_name: str, **kwargs: Any) -> None: ...

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        # Initialize the OpenAI client only once
        if MyCustomOpenAIModel._client is None:
            # Assumes OPENAI_API_KEY is set in environment variables
            MyCustomOpenAIModel._client = OpenAI(
                base_url=self.provider.base_url,
                api_key=self.provider.api_key
            )

    @classmethod
    def from_name(cls, name: str, **kwargs: Any) -> Self:
        """
        Required method for Model class: Instantiates the model from a string name.
        The name format is 'custom_openai:model-name'.
        """
        try:
            _, model_name = name.split(":", 1)
        except ValueError:
            raise ValueError(f"Invalid name format for MyCustomOpenAIModel: {name}. Expected 'custom_openai:model-name'.")

        return cls(openai_model_name=model_name, **kwargs)

    async def _run(self, messages: list[Message], settings: ModelSettings) -> Message:
        """
        The core asynchronous method to make the LLM API call.
        """
        # Convert pydantic-ai Message objects to OpenAI API format
        openai_messages = []
        for msg in messages:
            if isinstance(msg, UserMessage):
                role = "user"
            elif isinstance(msg, SystemMessage):
                role = "system"

            else:
                continue # Skip other message types like ToolMessage

            openai_messages.append({"role": role, "content": msg.content})

        try:
            # Make the actual OpenAI API call
            response = await self._client.chat.completions.create(
                model=self.openai_model_name,
                messages=openai_messages,
                temperature=settings.temperature,
                max_tokens=settings.max_tokens,
                # Note: Tool calling logic is complex and omitted for this simple example
            )

            # Extract the content from the response
            content = response.choices[0].message.content or ""

            return UserMessage(content=content)
        except Exception as e:
            print(f"OpenAI API Error: {e}")
            # In a real app, handle retries or raise a specific error
            raise e

# --- 2. Define Agent's Output Model ---

class Joke(BaseModel):
    """A short, funny joke."""
    setup: str = Field(description="The first part of the joke, ending in a question or an opening line.")
    punchline: str = Field(description="The conclusion that makes the joke funny.")
    category: str = Field(description="The topic or category of the joke (e.g., 'animal', 'programming').")

# --- 3. Initialize and Run the Agent with the Custom Model ---

if __name__ == "__main__":
    # NOTE: You must have the OPENAI_API_KEY environment variable set
    if not os.getenv("OPENAI_API_KEY"):
        print("Please set the OPENAI_API_KEY environment variable.")
    else:
        # 1. Instantiate the custom model function
        custom_model_instance = MyCustomOpenAIModel(openai_model_name="gpt-3.5-turbo")

        # 2. Initialize the agent with the custom model instance
        joke_agent = Agent(
            model=custom_model_instance,
            result_type=Joke,
            system_prompt="You are a professional comedian. You always respond by filling the Joke schema with a single joke."
        )

        # 3. Run the agent
        print("Running agent with custom LLM function...")
        try:
            result = joke_agent.run_sync("Tell me a short joke about computers.")

            print("\n--- Agent Result (Validated Pydantic Model) ---")
            print(f"Setup: {result.setup}")
            print(f"Punchline: {result.punchline}")
            print(f"Category: {result.category}")
            print("-" * 40)

        except Exception as e:
            print(f"\nAn error occurred during agent execution: {e}")