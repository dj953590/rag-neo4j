# mcp_client/mcp_client.py
import asyncio
import os
import requests
import time
from requests_oauthlib import OAuth2Session
from dynaconf import Dynaconf
from pydantic_ai import Agent
from pydantic_ai.mcp import MCPServerStreamableHTTP
from pydantic_ai.models.openai import OpenAIChatModel, OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider
from fastmcp.client.auth import BearerAuth
from pydantic_ai.result import FinalResult as Result
from typing import Dict, Any, Optional

# Configure Dynaconf to read from settings.yaml and.secrets.yaml
settings = Dynaconf(
    settings_files=["settings.yaml"],
    secrets_files=[".secrets.yaml"],
)

# Configure the LLM to use Gemini's OpenAI-compatible API endpoint.
gemini_provider = OpenAIProvider(
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    api_key=os.getenv("GEMINI_API_KEY"),
)

class TokenManager:
    """Manages the lifecycle of an access token, including auto-refresh."""
    def __init__(self, config: Dict[str, Any], renewal_threshold: int):
        self._config = config
        self._token: Optional[str] = None
        self._expires_at: Optional[int] = None
        self._renewal_threshold = renewal_threshold

    async def get_token(self) -> str:
        """Returns a valid access token, refreshing it if necessary."""
        if not self._token or self._is_token_expired():
            await self._refresh_token()
        return self._token

    def _is_token_expired(self) -> bool:
        """Checks if the token is within the renewal threshold."""
        return self._expires_at and (self._expires_at - self._renewal_threshold) < time.time()

    async def _refresh_token(self):
        """Acquires a new token using the Client Credentials flow."""
        print(f"Token for '{self._config['client_id']}' is expired or missing, acquiring a new one...")

        token_data = {
            "grant_type": "client_credentials",
            "client_id": self._config['client_id'],
            "client_secret": self._config['client_secret'],
        }

        try:
            response = requests.post(self._config['token_url'], data=token_data)
            response.raise_for_status()

            token_response = response.json()
            self._token = token_response.get("access_token")
            expires_in = token_response.get("expires_in")

            if not self._token or not expires_in:
                raise ValueError("Access token or expiry not found in response.")

            # Set a more conservative expiry time to account for network latency
            self._expires_at = time.time() + expires_in
            print(f"Successfully acquired new token for '{self._config['client_id']}'.")

        except requests.exceptions.RequestException as e:
            print(f"Error during token acquisition for '{self._config['client_id']}': {e}")
            raise

async def get_authenticated_clients_with_refresh():
    """
    Dynamically loads server configurations from Dynaconf,
    initializes TokenManagers, and returns a list of MCP clients.
    """
    mcp_clients = []

    # Iterate through each server configured in settings.yaml
    for server_name, server_config in settings.mcp_servers.items():
        token_manager = TokenManager(
            config=server_config,
            renewal_threshold=settings.token_renewal_threshold_secs
        )

        try:
            # Get the initial token
            initial_token = await token_manager.get_token()

            # Create a Pydantic AI MCP client with the Bearer token and refresh logic
            mcp_client = MCPServerStreamableHTTP(
                url=server_config.url,
                auth=BearerAuth(token=initial_token),
                tool_prefix=server_name + "_"
            )
            mcp_client._token_manager = token_manager # Attach the token manager to the client
            mcp_clients.append(mcp_client)

        except Exception as e:
            print(f"Could not initialize client for '{server_name}': {e}")

    return mcp_clients

async def main():
    mcp_clients = await get_authenticated_clients_with_refresh()

    if not mcp_clients:
        print("No clients could be authenticated. Exiting.")
        return

    # Create a single Pydantic AI agent with all the authenticated clients
    agent = Agent(
        model=OpenAIModel("gemini-1.5-flash", provider=gemini_provider),
        toolsets=mcp_clients,
        system_prompt="You are a helpful assistant that can answer mathematical and weather-related questions. Use the available tools to find the answers."
    )

    async with agent:
        print("\nAgent is ready to answer questions using multiple secure servers.\n")

        # Test 1: First call to the 'calculator' server
        query_1 = "What is the sum of 10 and 5?"
        print(f"User Query: {query_1}")
        result_1: Result = await agent.run(query_1)
        print(f"Agent Response: {result_1.output}")

        print("-" * 20)

        # Test 2: Wait for the token to expire and make a second call to trigger auto-refresh
        print("Waiting for token to expire to test auto-refresh...")
        await asyncio.sleep(45) # Token expires in 60s, renews at 45s

        query_2 = "What is the weather forecast for Seattle?"
        print(f"User Query: {query_2}")
        result_2: Result = await agent.run(query_2)
        print(f"Agent Response: {result_2.output}")

        print("-" * 20)

        # Test 3: A final call to show that the new token is being used
        query_3 = "What is 25 + 35?"
        print(f"User Query: {query_3}")
        result_3: Result = await agent.run(query_3)
        print(f"Agent Response: {result_3.output}")


if __name__ == "__main__":
    if not os.getenv("GEMINI_API_KEY"):
        print("Error: GEMINI_API_KEY environment variable is not set.")
    else:
        asyncio.run(main())