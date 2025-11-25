"""Generalized LLM API client for making requests to various providers."""

import httpx
from typing import List, Dict, Any, Optional
from .config import OPENROUTER_API_KEY, OPENROUTER_API_URL, DEEPSEEK_API_KEY, OPENAI_API_KEY


# Define Deepseek API URL
DEEPSEEK_API_URL = "https://api.deepseek.com/chat/completions"
# Define OpenAI API URL
OPENAI_API_URL = "https://api.openai.com/v1/chat/completions"

# Map providers to their configurations
PROVIDER_CONFIGS = {
    "openrouter": {
        "api_key": OPENROUTER_API_KEY,
        "api_url": OPENROUTER_API_URL,
        "model_prefix": "openrouter::",
        "header_auth_prefix": "Bearer "
    },
    "deepseek": {
        "api_key": DEEPSEEK_API_KEY,
        "api_url": DEEPSEEK_API_URL,
        "model_prefix": "deepseek::",
        "header_auth_prefix": "Bearer "
    },
    "openai": {
        "api_key": OPENAI_API_KEY,
        "api_url": OPENAI_API_URL,
        "model_prefix": "openai::",
        "header_auth_prefix": "Bearer "
    }
    # Add other providers here
}


async def query_model(
    model_identifier: str, # Renamed to avoid clash with internal 'model' variable
    messages: List[Dict[str, str]],
    timeout: float = 120.0
) -> Optional[Dict[str, Any]]:
    """
    Query a single model via the appropriate LLM API.

    Args:
        model_identifier: Identifier for the model, in the format "provider::model_name"
                          (e.g., "openrouter::openai/gpt-4o", "deepseek::deepseek-chat").
        messages: List of message dicts with 'role' and 'content'
        timeout: Request timeout in seconds

    Returns:
        Response dict with 'content' and optional 'reasoning_details', or None if failed
    """
    provider_name = None
    model_name = model_identifier

    # Determine provider and actual model name
    for provider, config in PROVIDER_CONFIGS.items():
        if model_identifier.startswith(config["model_prefix"]):
            provider_name = provider
            model_name = model_identifier[len(config["model_prefix"]):]
            break
    
    # If no specific provider prefix, assume openrouter as default for backward compatibility
    if not provider_name:
        provider_name = "openrouter" # Default to OpenRouter
        # In this case, model_name is already model_identifier

    config = PROVIDER_CONFIGS.get(provider_name)

    if not config:
        print(f"Error: Unknown LLM provider for model identifier: {model_identifier}")
        return None

    api_key = config["api_key"]
    api_url = config["api_url"]
    header_auth_prefix = config["header_auth_prefix"]

    if not api_key:
        print(f"Error: API key not configured for provider {provider_name} (model: {model_identifier})")
        return None

    headers = {
        "Authorization": f"{header_auth_prefix}{api_key}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": model_name,
        "messages": messages,
    }

    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(
                api_url,
                headers=headers,
                json=payload
            )
            response.raise_for_status()

            data = response.json()
            message = data['choices'][0]['message']

            return {
                'content': message.get('content'),
                'reasoning_details': message.get('reasoning_details')
            }

    except Exception as e:
        print(f"Error querying model {model_identifier} (provider: {provider_name}, URL: {api_url}): {e}")
        return None


async def query_models_parallel(
    model_identifiers: List[str], # Renamed to clarify it's a list of full identifiers
    messages: List[Dict[str, str]]
) -> Dict[str, Optional[Dict[str, Any]]]:
    """
    Query multiple models in parallel.

    Args:
        model_identifiers: List of model identifiers (e.g., "provider::model_name")
        messages: List of message dicts to send to each model

    Returns:
        Dict mapping model identifier to response dict (or None if failed)
    """
    import asyncio

    # Create tasks for all models
    tasks = [query_model(model_id, messages) for model_id in model_identifiers]

    # Wait for all to complete
    responses = await asyncio.gather(*tasks)

    # Map models to their responses
    return {model_id: response for model_id, response in zip(model_identifiers, responses)}
