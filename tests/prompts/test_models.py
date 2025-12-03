import json
import requests

# --- CONFIG ---
INPUT_FILE = "prompts.json"       # Ton fichier source
OUTPUT_FILE = "results.txt"       # Fichier résultat
LM_SERVER_URL = "http://localhost:1234/v1/chat/completions"  # Adresse LM Studio
MODEL_NAME = "mistral-7b-instruct-v0.2"  # Ou le modèle que tu utilises localement

# --- FONCTIONS ---

def get_full_context(prompt_item, parent_contexts):
    """
    Concatène tous les contextes parents avec le contexte du prompt (si existant)
    """
    contexts = [ctx for ctx in parent_contexts if ctx]  # ignore None
    if "context" in prompt_item and prompt_item["context"]:
        contexts.append(prompt_item["context"])
    return "\n".join(contexts)

def process_prompts(prompts, parent_contexts, indent_level=0):
    """
    Parcourt la liste des prompts, envoie à LM Studio, et retourne les résultats formatés
    """
    results = []
    indent = "    " * indent_level

    for prompt_item in prompts:
        full_context = get_full_context(prompt_item, parent_contexts)
        text_to_send = f"{full_context}\n\nPrompt: {prompt_item['prompt']}"

        # --- Appel LM Studio ---
        response = requests.post(
            LM_SERVER_URL,
            json={
                "model": MODEL_NAME,
                "messages": [
                    { "role": "user", "content": text_to_send }
                    for text_to_send in parent_contexts
                ] + [{"role": "user", "content": text_to_send}]
            }
        )
        response.raise_for_status()
        lm_result = response.json()["choices"][0]["message"]["content"].strip()

        results.append(f"{indent}{prompt_item['title']}: \"{lm_result}\"")
        print(f"{indent}{prompt_item['title']}: \"{lm_result}\"")
    
    return results

def process_category(category, parent_contexts=None, indent_level=0):
    """
    Parcourt récursivement une catégorie et ses sous-catégories
    """
    if parent_contexts is None:
        parent_contexts = []
    
    results = []
    indent = "    " * indent_level

    # Concatène le contexte parent et celui de cette catégorie
    current_contexts = parent_contexts.copy()
    if "context" in category and category["context"]:
        current_contexts.append(category["context"])
    
    # Ajoute le titre de la catégorie
    results.append(f"{indent}{category['title']}")
    print(f"{indent}{category['title']}")

    # Si elle a des prompts
    if "prompts" in category:
        results.extend(process_prompts(category["prompts"], current_contexts, indent_level+1))

    # Si elle a des sous-catégories
    if "categories" in category:
        for subcat in category["categories"]:
            results.extend(process_category(subcat, current_contexts, indent_level+1))

    return results

# --- MAIN ---
def main():
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    final_results = []
    for top_category in data:
        final_results.extend(process_category(top_category))

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(final_results))
    
    print(f"Résultats écrits dans {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
