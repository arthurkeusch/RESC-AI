import json
import os
import re
import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm
from openai import OpenAI
from dotenv import load_dotenv

# ==========================================
# CHARGEMENT DES CONFIGURATIONS (.env)
# ==========================================
load_dotenv()

INPUT_FILENAME = "benchmark_results_0001.json" 

# Configuration conforme à votre serveur privé (LiteLLM)
BASE_URL_BRIDGE = "https://llm.ai.anhilyx.fr/v1"
API_KEY_BRIDGE = os.getenv("MODEL_TOKEN")  # Ajusté pour correspondre à votre projet

MODEL_JUDGE_NAME = "Mistral Large (Web)"
REPEAT_EVALUATIONS = 3  # Nombre d'évaluations par prompt pour la robustesse


def get_paths(filename: str) -> tuple:
    """Calcule dynamiquement les chemins d'entrée et de sortie."""
    input_path = os.path.join(".in", filename)
    folder_name = os.path.splitext(filename)[0]
    output_dir = os.path.join(".out", folder_name)
    return input_path, output_dir


def load_benchmark_json(file_path: str) -> dict:
    """Charge le fichier JSON depuis le dossier .in/."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Le fichier cible est introuvable au chemin : {file_path}")
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_llm_score_bridge(client: OpenAI, prompt_text: str, chunks: list) -> int:
    """Interroge votre serveur LiteLLM avec un délai proactif pour respecter le quota Mistral."""
    context = "\n---\n".join([f"- Extrait : {c}" for c in chunks])

    system_prompt = (
        "Tu es un expert en évaluation de systèmes RAG.\n"
        "Évalue si les extraits fournis contiennent la réponse exacte à la question.\n"
        "Attribue une note stricte de 0 à 5 selon ce barème :\n"
        "5 : Réponse exacte, complète et explicite.\n"
        "4 : Indices majeurs.\n"
        "3 : Indices partiels mais utiles.\n"
        "2 : Indices très vagues ou douteux.\n"
        "1 : Réponse très vague ou non pertinente.\n"
        "0 : Totalement hors-sujet ou vide.\n"
        "Réponds UNIQUEMENT par un chiffre entier unique (Ex: 4). Ne donne aucune explication, ni aucun autre texte."
    )

    user_content = f"Question : {prompt_text}\n\nExtraits RAG :\n{context}\n\nNote (0 à 5) :"

    max_retries = 5
    backoff_factor = 2

    for attempt in range(max_retries):
        try:
            # Respect strict du quota de l'API Mistral (1 req / seconde maximum)
            time.sleep(1.0)
            
            chat_completion = client.chat.completions.create(
                model=MODEL_JUDGE_NAME,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content}
                ],
                temperature=0,  # Fixé à 0 comme dans votre code Agent pour un maximum de régularité
                max_tokens=5,
            )
            
            response_text = chat_completion.choices[0].message.content.strip()
            
            match = re.search(r"\d", response_text)
            if match:
                score = int(match.group(0))
                return min(max(score, 0), 5)
            return 0

        except Exception as e:
            if "429" in str(e) or "rate_limit" in str(e).lower():
                sleep_time = (backoff_factor ** attempt) + 3
                tqdm.write(f"⏳ Limitation détectée (429). Pause de sécurité de {sleep_time}s...")
                time.sleep(sleep_time)
            else:
                tqdm.write(f"❌ Erreur lors de la communication avec le pont : {e}")
                return 0
    return 0


def process_data(json_data: dict, client: OpenAI) -> tuple:
    """Parcourt le JSON, extrait les durées et appelle votre API."""
    all_times = []
    all_scores = []

    total_prompts = 0
    for model_repo, model_data in json_data.items():
        if model_data.get("success", True):
            for doc_name, doc_data in model_data.get("documents", {}).items():
                if doc_data.get("success", True):
                    total_prompts += len([p for p, p_data in doc_data.get("prompts", {}).items() if p_data.get("success", True)])

    with tqdm(total=total_prompts, desc="Évaluation via votre Serveur Pont", unit="prompt") as pbar:
        for model_repo, model_data in json_data.items():
            if not model_data.get("success", True):
                continue

            model_short_name = model_repo.split("/")[-1]
            pbar.set_postfix_str(f"Modèle RAG: {model_short_name}")
            
            model_init_time = model_data.get("time_ms", 0)
            doc_times = []
            prompt_times = []

            for doc_name, doc_data in model_data.get("documents", {}).items():
                if not doc_data.get("success", True):
                    continue

                doc_times.append(doc_data.get("time_ms", 0))
                
                for prompt_text, prompt_data in doc_data.get("prompts", {}).items():
                    if not prompt_data.get("success", True):
                        continue

                    prompt_times.append(prompt_data.get("time_ms", 0))
                    chunks = prompt_data.get("retrieved_chunks", [])
                    
                    scores = [
                        get_llm_score_bridge(client, prompt_text, chunks)
                        for _ in range(REPEAT_EVALUATIONS)
                    ]
                    
                    avg_score = sum(scores) // len(scores)
                    all_scores.append({
                        "Modèle": model_short_name,
                        "Document": doc_name,
                        "Prompt": prompt_text,
                        "Score": avg_score
                    })
                    
                    tqdm.write(f"  [Note: {avg_score}/5] {prompt_text[:50]}...")
                    pbar.update(1)

            all_times.append({
                "Modèle": model_short_name,
                "Initialisation Modèle": model_init_time,
                "Indexation Document": np.mean(doc_times) if doc_times else 0,
                "Requête RAG": np.mean(prompt_times) if prompt_times else 0
            })

    return pd.DataFrame(all_times), pd.DataFrame(all_scores)


def generate_time_charts(df_times: pd.DataFrame, output_dir: str):
    """Génère les graphiques temporels sur 3 échelles verticales."""
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(12, 16), sharex=True)
    
    metrics = [
        ("Initialisation Modèle", "Temps de chargement initial du modèle", "#3A86FF"),
        ("Indexation Document", "Temps moyen d'indexation d'un document (Parse PDF + Embeddings)", "#FF006E"),
        ("Requête RAG", "Temps moyen d'exécution d'une requête RAG (Inférence unitaire)", "#8338EC")
    ]
    
    for idx, (col_name, title, color) in enumerate(metrics):
        ax = axes[idx]
        sns.barplot(data=df_times, x="Modèle", y=col_name, ax=ax, color=color, hue="Modèle", legend=False)
        ax.set_title(title, fontsize=13, fontweight="bold", pad=10)
        ax.set_ylabel("Temps (ms)", fontsize=11)
        ax.set_xlabel("")
        
        for p in ax.patches:
            height = p.get_height()
            if height > 0:
                ax.annotate(f"{height:,.1f} ms", (p.get_x() + p.get_width() / 2., height),
                            ha='center', va='center', xytext=(0, 8), textcoords='offset points', 
                            fontsize=10, fontweight="semibold")

    axes[-1].set_xticklabels(axes[-1].get_xticklabels(), rotation=15, ha="right", fontsize=11)
    axes[-1].set_xlabel("Modèles d'Embedding", fontsize=12, fontweight="bold", labelpad=10)
    fig.suptitle("Analyse Multi-Échelle des Temps d'Exécution", fontsize=18, fontweight="bold", y=0.98)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "times.png"), dpi=300)
    plt.close()


def generate_score_chart(df_scores: pd.DataFrame, output_dir: str):
    """Génère le graphique de performance par document."""
    sns.set_theme(style="whitegrid")
    df_avg_scores = df_scores.groupby(["Document", "Modèle"])["Score"].mean().unstack()
    plt.figure(figsize=(14, 8))
    short_doc_names = [d[:35] + "..." if len(d) > 35 else d for d in df_avg_scores.index]
    
    ax = df_avg_scores.plot(kind="bar", width=0.75, ax=plt.gca(), cmap="plasma")
    plt.title("Score moyen de pertinence des extraits par Document (Mistral Large)", fontsize=15, fontweight="bold", pad=15)
    plt.ylabel("Note attribuée par le modèle juge (0 à 5)", fontsize=12)
    plt.ylim(0, 5.5)
    ax.set_xticklabels(short_doc_names, rotation=15, ha="right", fontsize=11)
    plt.legend(title="Modèles évalués", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "scores.png"), dpi=300)
    plt.close()


def main():
    input_path, output_dir = get_paths(INPUT_FILENAME)
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        if not API_KEY_BRIDGE:
            print("❌ Erreur : La variable 'MODEL_TOKEN' n'est pas définie dans votre fichier .env.")
            return
            
        json_data = load_benchmark_json(input_path)
        
        client = OpenAI(
            api_key=API_KEY_BRIDGE,
            base_url=BASE_URL_BRIDGE
        )

        df_times, df_scores = process_data(json_data, client)

        print("\nGénération des rendus graphiques...")
        generate_time_charts(df_times, output_dir)
        generate_score_chart(df_scores, output_dir)

        print(f"\n[Succès] Fichiers enregistrés dans : {output_dir}")

    except Exception as e:
        print(f"\n[Erreur] : {e}")


if __name__ == "__main__":
    print(f"DEBUG - Clé envoyée : [{os.getenv('MODEL_TOKEN')}]")
    main()