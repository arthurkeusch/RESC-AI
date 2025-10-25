import pickle
from time import time_ns
from typing import Dict, List
from sentence_transformers import SentenceTransformer
import faiss

DEFAULT_K = 3
MODEL_NAME = "all-MiniLM-L6-v2"
# MODEL_NAME = "paraphrase-multilingual-mpnet-base-v2"
# MODEL_NAME = "intfloat/e5-large"  # <= Very good, but requires more VRAM


class RAG:
    """
    A class for Retrieval-Augmented Generation (RAG) using SentenceTransformer and FAISS.
    This class provides methods to encode texts into embeddings and search for similar texts.
    Attributes:
        model (SentenceTransformer): The sentence transformer model for encoding texts.
        index (faiss.IndexFlatL2): The FAISS index for storing and searching embeddings.
        texts (list of str): The list of texts stored in the RAG system.
    """

    def __init__(self, texts: List[str], model_name: str = MODEL_NAME):
        """
        Initializes the RAG class with a list of texts.
        Args:
            texts (list of str): The texts to be encoded and stored.
        """

        timestamp = time_ns()

        # Initialize values
        self.model = SentenceTransformer(MODEL_NAME, device="cuda")
        self.texts = texts
        self.embeddings = None
        self.index = faiss.IndexFlatL2(self.model.get_sentence_embedding_dimension())
        
        if texts:
            # Encode the texts to get their embeddings
            self.embeddings = self.model.encode(texts, convert_to_numpy=True)

            # Create a FAISS index and add the embeddings
            self.index = faiss.IndexFlatL2(self.embeddings.shape[1])
            self.index.add(self.embeddings)

        print(f"RAG instance created with {len(texts)} texts in {int((time_ns() - timestamp) / 1_000_000) / 1_000} seconds.")
    
    def load(file_path: str) -> 'RAG':
        """
        Load a RAG instance from a file.
        Args:
            file_path (str): The path to the file containing the saved RAG instance.
        Returns:
            RAG: The loaded RAG instance.
        """

        timestamp = time_ns()

        # Load the texts and embeddings using numpy
        with open(file_path, "rb") as f:
            data = pickle.load(f)
            file_size = f.tell()
        
        # Create a new RAG instance and populate its attributes
        rag = RAG([])
        rag.texts = data["texts"]
        rag.embeddings = data["embeddings"]
        rag.index.add(rag.embeddings)

        print(f"RAG instance loaded with {len(rag.texts)} texts in {int((time_ns() - timestamp) / 1_000_000) / 1_000} seconds. File size: {int(file_size / 1_000) / 1_000} MB.")
        return rag

    def save(self, file_path: str):
        """
        Save the RAG instance to a file.
        Args:
            file_path (str): The path to the file where the RAG instance will be saved.
        """

        timestamp = time_ns()

        # Save the texts and embeddings using numpy
        with open(file_path, "wb") as f:
            pickle.dump({
                "texts": self.texts,
                "embeddings": self.embeddings
            }, f)

            print(f"RAG instance saved with {len(self.texts)} texts in {int((time_ns() - timestamp) / 1_000_000) / 1_000} seconds. File size: {int(f.tell() / 1_000) / 1_000} MB.")

    def add_text(self, new_text: str):
        """
        Add a new text to the RAG system.
        Args:
            new_text (str): The new text to be added.
        """

        self.add_texts([new_text])

    def add_texts(self, new_texts: List[str]):
        """
        Add new texts to the RAG system.
        Args:
            new_texts (list of str): The new texts to be added.
        """

        # Encode the new texts to get their embeddings
        new_embeddings = self.model.encode(new_texts, convert_to_numpy=True)

        # Add the new embeddings to the FAISS index
        self.index.add(new_embeddings)

        # Update the stored texts
        self.texts.extend(new_texts)

    def search_texts(self, query: str, k: int = DEFAULT_K) -> List[str]:
        """
        Search for the most similar texts to a given query using the FAISS index.
        Args:
            query (str): The query text to search for similar texts.
            k (int): The number of similar texts to retrieve.
        Returns:
            list of str: The list of most similar texts.
        """

        # Encode the query text to get its embedding
        query_embedding = self.model.encode([query], convert_to_numpy=True)

        # Search for the most similar texts in the FAISS index
        _, indices = self.index.search(query_embedding, k)

        # Retrieve and return the similar texts
        return [self.texts[i] for i in indices[0].tolist()]

    def get_context(self, prompt: str, k: int = DEFAULT_K) -> Dict[str, str]:
        """
        Create a context dictionary augmented with the most similar texts.
        Args:
            prompt (str): The original prompt text.
            k (int): The number of similar texts to include.
        Returns:
            dict: A dictionary containing the augmented prompt.
        """

        texts = self.search_texts(prompt, k)
        print(f"Retrieved {len(texts)}/{k} relevant texts for the prompt.")
        for i, text in enumerate(texts):
            print(f"  [{i+1}] {text}\n")
        return {"system": text for text in texts}


# Check if CUDA is available for PyTorch
import torch
if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available. Please ensure you have a compatible GPU and the correct drivers installed.")
else:
    print(f"CUDA is available. Using GPU {torch.cuda.get_device_name(0)} for computations.")