import api


def print_commands():
    """
    Prints the available commands.
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
    models = api.get_available_models()
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
    
    return model


def main():
    """
    Main function of the script.
    """

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
            response = api.prompt_str(model, text).replace("\n", "\n  ")
            print(response)


if  __name__ == "__main__":
    main()