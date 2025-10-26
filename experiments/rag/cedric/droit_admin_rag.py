from time import time_ns
from api import prompt_str, select_model
from rag import RAG
import os
import xml.etree.ElementTree as ET

RAG_MODEL_NAME = "intfloat/e5-large"
JURISPRUDENCE_OUT_FILE = "out/jurisprudence_rag.pkl"
NB_JURISPRUDENCE_TEXTS = 5

count = 1
def create_jurisprudence_rag(source_dir: str) -> RAG:
    """
    Create and save a RAG instance for jurisprudence texts.
    Args:
        source_dir (str): The path to the source directory containing XML jurisprudence texts.
    """

    def load_texts(dir: str) -> list[str]:
        """
        Load jurisprudence texts from XML files in the source directory and recursively in subdirectories.
        Args:
            dir (str): The path to the source directory containing XML jurisprudence texts and subdirectories.
        Returns:
            list[str]: A list of jurisprudence texts from the directory and its subdirectories.
        """

        texts = []
        for path in os.listdir(dir):
            try:
                full_path = os.path.join(dir, path)

                # Recursively load texts from subdirectories
                if os.path.isdir(full_path):
                    texts.extend(load_texts(full_path))
                
                # Load texts from XML files
                elif path.endswith(".xml"):
                    with open(full_path, "r", encoding="utf-8") as f:
                        xml_content = f.read().replace("<br>", "\n").replace("<br/>", "\n").replace("<br />", "\n")
                        root = ET.fromstring(xml_content)

                        # Retrieve relevant fields
                        title = root.find(".//TITRE")
                        subject = root.find(".//TYPE_REC")
                        content = root.find(".//CONTENU")

                        # Handle malformed XML content
                        if content is None:
                            print(f"❌ Missing <CONTENU> in file '{full_path}'. Skipping this file.")
                            continue
                        if title is None:
                            print(f"⚠️ Missing <TITRE> in file '{full_path}'.")
                        if subject is None:
                            print(f"⚠️ Missing <TYPE_REC> in file '{full_path}'.")

                        # Create the full text
                        full_text = f"Titre: {title.text}\n\n" + \
                                    f"Sujet: {subject.text}\n\n" + \
                                    "\n".join([
                                        line
                                        for line in content.text.split("\n")
                                        if line.strip() != ""
                                    ])
                        texts.append(full_text)

                        # Progress indication
                        global count
                        print(f"Loaded {count} files", end="\r")
                        count += 1

                # Handle unsupported file formats
                else:
                    print(f"❌ Unsupported file format '{full_path}'. Only XML files are processed. Skipping this file.")
            except Exception as e:
                print(f"❌ Error processing file '{full_path}': {e}. Skipping this file.")
                continue

        return texts

    # Read the jurisprudence texts from the source directory
    timestamp = time_ns()
    texts = load_texts(source_dir)
    print(f"Loaded {len(texts)} jurisprudence texts in {int((time_ns() - timestamp) / 1_000_000) / 1_000} seconds.")

    # Create the RAG instance
    rag = RAG(texts, model_name=RAG_MODEL_NAME)

    # Save the RAG instance
    os.makedirs("out", exist_ok=True)
    rag.save(JURISPRUDENCE_OUT_FILE)


def main():
    """
    Main function to run the RAG-based assistant for jurisprudence texts.
    """

    if not os.path.exists(JURISPRUDENCE_OUT_FILE):
        create_jurisprudence_rag("in/jade/")

    # Initialize the instance
    jurisprudence_rag = RAG.load(JURISPRUDENCE_OUT_FILE)
    model = select_model()

    # Process user queries
    while True:
        prompt = input("[User]  ")
        texts = jurisprudence_rag.search_texts(prompt, k=NB_JURISPRUDENCE_TEXTS)

        print(f"\n[Context]\n" + "\n----------\n".join(texts).replace("\n\n", "\n").strip() + "\n")

        response = prompt_str(model,
            "You are an assistant that must answer questions thanks to some or all of the provided jurisprudence texts in context.\n" + \
            "Your answer **must** include the **title** of each text you **use**.\n\n" + \
            "<context>\n" + \
            "\n\n".join(texts) + \
            "\n</context>\n\n" +
            prompt
        )

        while ("  " in response):
            response = response.replace("  ", " ")
        while ("\n\n" in response):
            response = response.replace("\n\n", "\n")
        print(f"[Assistant]  {response}\n")


if __name__ == "__main__":
    main()
