import json
import os
import re
import sys
from sentence_transformers import SentenceTransformer


def __load_vocab(file_path: str) -> list[str]:
	"""
	Load a vocabulary file and return a list of words.
	The file must be a `.tsv` file from Lexique.org, or a similar format:
	- The first line must be a header with at least the column "*ortho*".
	- Each subsequent line must contain the data for each word.
	- Each column must be separated by a `\\t` character.

	Args:
		file_path (str): The path to the vocabulary file.
	
	Returns:
		List[str]: A list of words loaded from the vocabulary file.
	"""
	
	# Read lines
	with open(file_path, "r", encoding="utf-8") as tsv_file:
		lines = tsv_file.readlines()

	# Parse header
	headers = re.split(r"\t", lines[0])
	ortho_idx = headers.index("ortho")

	# Read words
	words: list[str] = []
	for line in lines[1:]:
		for column in [re.split(r"\t", line)]:

			# Extract word, lemma, and phonetic transcription
			word = column[ortho_idx]

			# Skip empty or duplicate words
			if word == "": continue
			for w in words:
				if w == word: continue
			
			# Append to list
			words.append(word)
	
	# Return words
	return words


def __get_embeddings(words: list[str]) -> list[list[float]]:
	"""
	Get embeddings for a list of words using a pre-trained model.

	Args:
		words (list[str]): A list of words to get embeddings for.
	
	Returns:
		list[list[float]]: A list of embeddings for each word.
	"""

	# Load pre-trained model
	model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

	# Get embeddings
	embeddings = model.encode(words, show_progress_bar=True)

	# Return embeddings
	return embeddings.tolist()


if  __name__ == "__main__":
	in_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), sys.argv[1])

	words = __load_vocab(in_path)
	embeddings = __get_embeddings(words)

	out_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), sys.argv[2])

	with open(out_path, "w", encoding="utf-8") as f:
		json.dump({
			"words": words,
			"embeddings": embeddings
		}, f)