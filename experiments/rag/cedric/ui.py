import api
from PyQt6 import uic
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QApplication, QWidget
import sys

class Input(QWidget):
    """
    Base class for user input widgets.
    """

    def __init__(self, text: str):
        """
        Initializes the user input widget.
        Args:
            text (str): The initial text for the input.
        """

        super(Input, self).__init__()
        uic.loadUi("message.ui", self)
        self.text.setText(text)


class UserInput(Input):
    """
    Class for user input widget.
    """

    def __init__(self, text: str = ""):
        """
        Initializes the user input widget.
        Args:
            text (str): The initial text for the input.
        """

        super(UserInput, self).__init__(text)
        self.setContentsMargins(0, 0, 50, 0)
        self.text.setStyleSheet(self.text.styleSheet() + "background-color: #003F7F;")
        self.text.setAlignment(Qt.AlignmentFlag.AlignLeft)


class SystemInput(Input):
    """
    Class for system input widget.
    """

    def __init__(self, text: str = ""):
        """
        Initializes the system input widget.
        Args:
            text (str): The initial text for the input.
        """

        super(SystemInput, self).__init__(text)
        self.setContentsMargins(50, 0, 0, 0)
        self.text.setStyleSheet(self.text.styleSheet() + "background-color: #7F3F00;")
        self.text.setAlignment(Qt.AlignmentFlag.AlignRight)


class App(QApplication):
    """
    Main application class.
    """

    def __init__(self, sys_argv):
        """
        Initializes the application.
        """

        # Initialize the application
        super(App, self).__init__(sys_argv)
        self.window = uic.loadUi("interface.ui")

        # Load the available models
        models = api.get_available_models()
        self.window.model_selector.addItems(models)

        # Connect button to function
        self.window.prompt_button.clicked.connect(self.send_prompt)

    def show(self):
        """
        Displays the main window.
        """

        self.window.show()

    def send_prompt(self):
        """
        Sends the user prompt to the API and displays the response.
        """

        # Get the user input
        prompt = self.window.prompt_text.toPlainText()
        # Avoid empty input
        if not prompt.strip():
            return
        # Display the user input
        self.window.messages_layout.addWidget(UserInput(prompt))
        # Clear the input field
        self.window.prompt_text.clear()

        # Send the user input to the API
        response = api.prompt_str(self.window.model_selector.currentText(), prompt)
        self.window.messages_layout.addWidget(SystemInput(response))


def main():
    """
    Main function to run the UI application.
    """

    app = App(sys.argv)
    app.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
