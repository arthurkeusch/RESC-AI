import json
import os
import re
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from tqdm import tqdm  # Importation de la barre de progression

# ==========================================
# CONFIGURATION DU FICHIER D'ENTRÉE
# ==========================================
INPUT_FILENAME = "benchmark_results_0001.json" 

MODEL_JUDGE_NAME = "Qwen/Qwen2.5-7B-Instruct"
# MODEL_JUDGE_NAME = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"

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
        raise FileNotFoundError(f"Le fichier cible est introuvable au chemin : {file_path}\nVeuillez vérifier qu'il est bien dans le dossier '.in/'.")
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def setup_llm_judge(model_name: str):
    """Initialise le modèle LLM sur le GPU en bfloat16."""
    print(f"Chargement du modèle juge ({model_name}) sur le GPU...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
    )
    return pipeline("text-generation", model=model, tokenizer=tokenizer)


def get_llm_score(judge_pipeline, prompt_text: str, chunks: list) -> int:
    """Demande à DeepSeek-R1 d'attribuer une note de 0 à 5."""
    context = "\n---\n".join([f"- Extrait : {c}" for c in chunks])

    user_content = (
        "Évalue si les extraits fournis contiennent la réponse exacte à la question.\n"
        "Attribue une note stricte de 0 à 5 selon ce barème :\n"
        "5 : Réponse exacte, complète et explicite.\n"
        "4 : Indices majeurs.\n"
        "3 : Indices partiels mais utiles.\n"
        "2 : Indices très vagues, potentiellement tirés d'un hors-sujet.\n"
        "1 : Réponse très vague ou non pertinente.\n"
        "0 : Totalement hors-sujet ou vide.\n"
        "Donne le chiffre de ta note à la toute fin de ta réponse.\n\n"
        f"Question : {prompt_text}\n\n"
        f"Extraits RAG :\n{context}\n\n"
    )

    messages = [
        {"role": "user", "content": user_content}
    ]

    outputs = judge_pipeline(
        messages,
        max_new_tokens=4096,
        temperature=0.6,
        top_p=0.95,
        do_sample=True,
    )

    response_text = outputs[0]["generated_text"][-1]["content"].strip()
    
    if "</think>" in response_text:
        final_answer = response_text.split("</think>")[-1].strip()
    else:
        final_answer = response_text

    match = re.search(r"\d", final_answer)
    if match:
        score = int(match.group(0))
        return min(max(score, 0), 5)
    return 0


def process_data(json_data: dict, judge_pipeline) -> tuple:
    """Parcourt le JSON, extrait les durées et interroge l'IA pour les scores avec une barre de progression."""
    all_times = []
    all_scores = []

    # 1. Pré-calcul du nombre total de prompts valides pour initialiser la barre de progression globale
    total_prompts = 0
    for model_repo, model_data in json_data.items():
        if model_data.get("success", True):
            for doc_name, doc_data in model_data.get("documents", {}).items():
                if doc_data.get("success", True):
                    total_prompts += len([p for p, p_data in doc_data.get("prompts", {}).items() if p_data.get("success", True)])

    # 2. Lancement de la boucle avec la barre de progression tqdm
    with tqdm(total=total_prompts, desc="Évaluation globale", unit="prompt") as pbar:
        for model_repo, model_data in json_data.items():
            if not model_data.get("success", True):
                tqdm.write(f"⚠️ Modèle sauté (échec détecté dans le JSON) : {model_repo}")
                continue

            model_short_name = model_repo.split("/")[-1]
            # Met à jour le texte à droite de la barre pour savoir quel modèle tourne
            pbar.set_postfix_str(f"Modèle actuel: {model_short_name}")
            tqdm.write(f"\n🚀 Évaluation du modèle d'embedding : {model_repo}")
            
            model_init_time = model_data.get("time_ms", 0)
            doc_times = []
            prompt_times = []

            documents_dict = model_data.get("documents", {})
            for doc_name, doc_data in documents_dict.items():
                if not doc_data.get("success", True):
                    continue

                doc_times.append(doc_data.get("time_ms", 0))
                prompts_dict = doc_data.get("prompts", {})
                
                for prompt_text, prompt_data in prompts_dict.items():
                    if not prompt_data.get("success", True):
                        continue

                    prompt_times.append(prompt_data.get("time_ms", 0))
                    chunks = prompt_data.get("retrieved_chunks", [])
                    
                    # Évaluation LLM répétée (les sous-étapes individuelles n'inondent plus le terminal)
                    scores = [
                        get_llm_score(judge_pipeline, prompt_text, chunks)
                        for _ in range(REPEAT_EVALUATIONS)
                    ]
                    
                    avg_score = sum(scores) // len(scores)
                    all_scores.append({
                        "Modèle": model_short_name,
                        "Document": doc_name,
                        "Prompt": prompt_text,
                        "Score": avg_score
                    })
                    
                    # Un log propre qui s'affiche au-dessus de la barre sans la casser
                    tqdm.write(f"  ✅ [Note: {avg_score}/5] {prompt_text[:60]}...")
                    
                    # Avancement de la barre d'un prompt complet (les 3 sous-itérations étant terminées)
                    pbar.update(1)

            all_times.append({
                "Modèle": model_short_name,
                "Initialisation Modèle": model_init_time,
                "Indexation Document": np.mean(doc_times) if doc_times else 0,
                "Requête RAG": np.mean(prompt_times) if prompt_times else 0
            })

    return pd.DataFrame(all_times), pd.DataFrame(all_scores)


def generate_time_charts(df_times: pd.DataFrame, output_dir: str):
    """Génère 3 sous-graphiques verticaux dans une seule image pour gérer les échelles."""
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
                ax.annotate(f"{height:,.1f} ms",
                            (p.get_x() + p.get_width() / 2., height),
                            ha='center', va='center',
                            xytext=(0, 8),
                            textcoords='offset points', fontsize=10, fontweight="semibold")

    axes[-1].set_xticklabels(axes[-1].get_xticklabels(), rotation=15, ha="right", fontsize=11)
    axes[-1].set_xlabel("Modèles d'Embedding", fontsize=12, fontweight="bold", labelpad=10)
    
    fig.suptitle("Analyse Multi-Échelle des Temps d'Exécution du RAG", fontsize=18, fontweight="bold", y=0.98)
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, "times.png")
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"-> Graphique des durées sauvegardé : {output_path}")


def generate_score_chart(df_scores: pd.DataFrame, output_dir: str):
    """Génère le graphique des scores moyens de pertinence regroupés par document."""
    sns.set_theme(style="whitegrid")
    df_avg_scores = df_scores.groupby(["Document", "Modèle"])["Score"].mean().unstack()
    
    plt.figure(figsize=(14, 8))
    short_doc_names = [d[:35] + "..." if len(d) > 35 else d for d in df_avg_scores.index]
    
    ax = df_avg_scores.plot(kind="bar", width=0.75, ax=plt.gca(), cmap="plasma")
    plt.title("Mean score of relevance for retrieved chunks by Document (Mistral Large)", fontsize=15, fontweight="bold", pad=15)
    plt.ylabel("Note (0-5)", fontsize=12)
    plt.ylim(0, 5.5)
    
    ax.set_xticklabels(short_doc_names, rotation=15, ha="right", fontsize=11)
    plt.legend(title="Evaluated Models", bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=10)
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, "scores.png")
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"-> Graphique des scores sauvegardé : {output_path}")


def main():
    input_path, output_dir = get_paths(INPUT_FILENAME)
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        json_data = load_benchmark_json(input_path)
        print(f"Fichier détecté avec succès dans : {input_path}")

        judge_pipeline = setup_llm_judge(MODEL_JUDGE_NAME)

        print("\nDébut de l'analyse automatisée par l'IA...")
        df_times, df_scores = process_data(json_data, judge_pipeline)

        print("\nGénération et mise en page des rendus graphiques...")
        generate_time_charts(df_times, output_dir)
        generate_score_chart(df_scores, output_dir)

        print(f"\n[Succès] Traitement terminé ! Retrouvez vos deux graphiques dans le dossier : {output_dir}")

    except Exception as e:
        print(f"\n[Erreur] Le processus a été interrompu : {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()