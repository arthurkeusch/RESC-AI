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


def main():
    """
    Main function of the script.
    """

    def print_commands():
        """
        Prints the available commands for the user.
        """

        print("  Available commands:\n"
            "    /help - Show this help message\n"
            "    /model - Change the current model\n"
            "    /exit - Exit the program")

    def select_model() -> str:
        """
        Prompts the user to select a model from the available models.
        Returns:
            str: The selected model ID.
        """

        # Get and display available models
        models = get_available_models()
        print("Available models:")
        for i, model in enumerate(models):
            print(f"[{i + 1}]  {model}")

        # Prompt user to select a model
        while True:
            selected = input("Selected model: ")
            if selected in models:
                model = selected
                break
            if selected.isdigit() and 1 <= int(selected) <= len(models):
                model = models[int(selected) - 1]
                break
            # Handle invalid model selection
            print("Invalid model selection.")

    # Initialize values
    model = select_model()

    while True:
        print("\n------------------------\n")
        text = input(f"[{model}]  ")

        # Skip empty input
        if not text.strip():
            continue

        # Detect commands (starting with '/')
        if text.startswith("/"):

            # Help command
            if text.lower() == "/help":
                print_commands()
            # Model change command
            elif text.lower() == "/model":
                model = select_model()
            # Exit command
            elif text.lower() == "/exit":
                break
            # Unknown command
            else:
                print("  Unknown command. Type /help for a list of commands.")

        # Else, it's a prompt
        else:
            response = prompt_str(model, text).replace("\n", "\n  ")
            print(response)


if __name__ == "__main__":
    main()
