import api
from PyQt6 import uic
from PyQt6.QtCore import Qt, QThread, pyqtSignal
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
        self.layout().setAlignment(self.text, Qt.AlignmentFlag.AlignLeft)


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
        self.layout().setAlignment(self.text, Qt.AlignmentFlag.AlignRight)


class App(QApplication):
    """
    Main application class.
    """

    class ModelsThread(QThread):
        """
        Thread to fetch available models from the API.
        """

        data_ready = pyqtSignal(dict)

        def run(self):
            """
            Fetch available models from the API.
            """

            data = { "models": api.get_available_models() }
            self.data_ready.emit(data)

    class PromptThread(QThread):
        """
        Thread to send user prompt to the API.
        """

        data_ready = pyqtSignal(dict)  # Signal émis quand la requête est finie

        def __init__(self, model, prompt):
            """
            Initializes the prompt thread.
            """

            super().__init__()
            self.model = model
            self.prompt = prompt

        def run(self):
            """
            Send the prompt to the API and get the response.
            """
            
            data = { "response": api.prompt_str(self.model, self.prompt) }
            self.data_ready.emit(data)

    def __init__(self, sys_argv):
        """
        Initializes the application.
        """

        # Initialize the application
        super(App, self).__init__(sys_argv)
        self.window = uic.loadUi("interface.ui")
        self.prompt_thread = None

        # Load the available models
        self.model_thread = self.ModelsThread()
        self.model_thread.data_ready.connect(self.on_models_response)
        self.model_thread.start()

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

        # Check that everything is loaded
        if not self.window.model_selector.isEnabled():
            return

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
        self.prompt_thread = self.PromptThread(self.window.model_selector.currentText(), prompt)
        self.prompt_thread.data_ready.connect(self.on_prompt_response)
        self.prompt_thread.start()

    def on_models_response(self, data):
        """
        Handles the response from the API for available models.
        Args:
            data (dict): The response data from the API.
        """

        self.window.model_selector.clear()
        self.window.model_selector.addItems(data["models"])
        self.window.model_selector.setEnabled(True)

    def on_prompt_response(self, data):
        """
        Handles the response from the API.
        Args:
            data (dict): The response data from the API.
        """

        self.window.messages_layout.addWidget(SystemInput(data["response"]))


def main():
    """
    Main function to run the UI application.
    """

    app = App(sys.argv)
    app.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
