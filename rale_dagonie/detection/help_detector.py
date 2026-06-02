import unicodedata
from difflib import SequenceMatcher

def normalize(text: str) -> str:
    text = text.lower()
    text = ''.join(
        c for c in unicodedata.normalize('NFD', text)
        if unicodedata.category(c) != 'Mn'
    )
    return text

def similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()

STRONG_KEYWORDS = [
    "au secours",
    "a l aide",
    "aide moi",
    "aidez moi",
    "appelez les secours",
    "appelez une ambulance",
    "appelez le samu",
    "appelez les pompiers",
    "je n arrive pas a respirer",
    "je suffoque",
    "je m etouffe",
    "urgence",
    "c est une urgence",
]

WEAK_KEYWORDS = [
    "aide",
    "aider",
    "aidez",
    "secours",
    "mal",
    "douleur",
    "danger",
    "peur",
    "panique",
    "etouffe",
    "respirer",
]

def detect_help(text: str) -> bool:
    if not text:
        return False

    text = normalize(text)
    words = text.split()

    for keyword in STRONG_KEYWORDS:
        if similarity(text, keyword) > 0.75:
            return True
    score = 0

    for keyword in WEAK_KEYWORDS:
        if similarity(text, keyword) > 0.65:
            score += 1

    for word in words:
        for keyword in WEAK_KEYWORDS:
            if similarity(word, keyword) > 0.8:
                score += 1

    return score >= 2