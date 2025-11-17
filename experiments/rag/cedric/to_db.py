

import json
import os
import struct
import sys


MAX_TEXTS = 100000


def __max_length(strings: list[str]) -> int:
    """
    Get the maximum length of strings in a list.

    Args:
        strings (list[str]): List of strings.
    
    Returns:
        int: Maximum length of strings.
    """

    return max(len(s) for s in strings)


def __check_embeddings(embeddings: list[list[float]]) -> bool:
    """
    Check if all embeddings have the same length.

    Args:
        embeddings (list[list[float]]): List of embeddings.

    Returns:
        bool: True if all embeddings have the same length, False otherwise.
    """

    length = len(embeddings[0])
    return all(len(emb) == length for emb in embeddings)


def __fill_string(s: str, length: int) -> str:
    """
    Fill a string with null characters to a specified length.

    Args:
        s (str): String to fill.
        length (int): Desired length of the string.
    
    Returns:
        str: Filled string.
    """

    return s + "\0" * (length - len(s))


def __float_to_bytes(floats: list[float]) -> bytes:
    """
    Convert a list of floats to a bytes object.

    Args:
        floats (list[float]): List of floats to convert.
    
    Returns:
        bytes: Bytes object representing the floats.
    """

    return b"".join([struct.pack(">f", f) for f in floats])


def __db_line(text: str|tuple[int, int], embedding: list[float], str_length: int, long_text: bool = False) -> bytes:
    """
    Create a database line from a text and its embedding.

    Args:
        text (str|tuple[int, int]): Text string or tuple of offset and length for long texts.
        embedding (list[float]): Corresponding embedding.
        str_length (int): Length to which the text should be filled.
        long_text (bool): Whether the text is long.
    
    Returns:
        bytes: Bytes object representing the database line.
    """

    if not long_text:
        filled_text = __fill_string(text, str_length) # type: ignore
        text_bytes = filled_text.encode("utf-32-be")
    else:
        offset, length = text
        text_bytes = struct.pack(">qq", offset, length)
    embedding_bytes = __float_to_bytes(embedding)
    return text_bytes + embedding_bytes


def __load_json(in_path: str) -> tuple[list[str], list[list[float]]]:
    """
    Load texts and embeddings from a JSON file.

    Args:
        in_path (str): Path to the JSON file.
    
    Returns:
        tuple[list[str], list[list[float]]]: Tuple containing list of texts and list of embeddings.
    """

    with open(in_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    texts = data["texts"] if "texts" in data else data["words"]
    embeddings = data["embeddings"]

    return texts, embeddings


def __to_db(in_path: str, out_path: str, sorted: bool, long_texts: bool):
    """
    Convert texts and embeddings from a JSON file to a binary database file.

    Args:
        in_path (str): Path to the input JSON file.
        out_path (str): Path to the output binary database file.
        sorted (bool): Whether the words are sorted.
        long_texts (bool): Whether the texts are long (and thus should be handled differently).
    """

    texts, embeddings = __load_json(in_path)

    if not __check_embeddings(embeddings):
        raise ValueError("All embeddings must have the same length.")

    str_length = __max_length(texts) * 4 if not long_texts \
                 else 16  # Equivalent to two long (8 bytes each), but multiplied by 4 later, so 4*4=16

    with open(out_path, "wb") as f:
        f.write(struct.pack(">I", str_length))
        f.write(struct.pack(">I", len(embeddings[0])))
        f.write(struct.pack("?", sorted))
        f.write(struct.pack("?", long_texts))

        if not long_texts:
            for text, embedding in zip(texts[:MAX_TEXTS], embeddings[:MAX_TEXTS]):
                line = __db_line(text, embedding, str_length)
                f.write(line)
        
        else:
            offset = 0
            with open(out_path[:-4] + ".rot", "wb") as f2:
                
                for text, embedding in zip(texts[:MAX_TEXTS], embeddings[:MAX_TEXTS]):

                    encoded_text = (text.encode("utf-8"))

                    line = __db_line((offset, len(encoded_text),), embedding, str_length, True)
                    f.write(line)

                    f2.write(encoded_text)
                    offset += len(encoded_text)


if  __name__ == "__main__":
    in_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), sys.argv[1])
    out_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), sys.argv[2])
    sorted = "sorted" in sys.argv
    long_texts = "long" in sys.argv
    __to_db(in_path, out_path, sorted, long_texts)