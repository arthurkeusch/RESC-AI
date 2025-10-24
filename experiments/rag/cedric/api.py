import json
import requests

URL = "http://localhost:1234"
MODELS_ENDPOINT = "/v1/models"
COMPLETION_ENDPOINT = "/v1/chat/completions"


def get_available_models() -> list:
    """
    Returns the list of available model IDs from the LM server.
    Returns:
        list: A list of model IDs.
    """

    response = requests.get(f"{URL}{MODELS_ENDPOINT}")
    response.raise_for_status()
    models_info = response.json()
    return [model["id"] for model in models_info["data"]]


def prompt_str(model_id: str, text: str) -> str:
    """
    Constructs a prompt string for the given model and text.
    Args:
        model_id (str): The model ID.
        text (str): The input text.
    Returns:
        str: The constructed prompt string.
    """

    response = requests.post(
        f"{URL}{COMPLETION_ENDPOINT}",
        json={
            "model": model_id,
            "messages": [
                { "role": "user", "content": text }
            ]
        }
    )
    response.raise_for_status()
    lm_result = response.json()["choices"][0]["message"]["content"].strip()
    return lm_result
