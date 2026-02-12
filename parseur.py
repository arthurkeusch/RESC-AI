import os

# Dossiers totalement ignorés
IGNORED_DIRS = {
    ".git", "__pycache__", "node_modules",
    ".idea", ".gradle", ".cxx", ".venv",
    "build", "dist", "out", ".android",
    "inspectionProfiles", "caches"
}

# Fichiers inutiles à afficher
IGNORED_FILES = {
    ".DS_Store", "Thumbs.db"
}

# Extensions à ignorer
IGNORED_EXTENSIONS = {
}

MAX_ITEMS = 100


def should_ignore_file(name: str) -> bool:
    return (
        name in IGNORED_FILES
        or any(name.endswith(ext) for ext in IGNORED_EXTENSIONS)
    )


def generate_tree(root_dir, output_file, indent=""):
    try:
        items = sorted(os.listdir(root_dir))
    except (PermissionError, FileNotFoundError):
        return

    # filtrage
    filtered_items = []
    for item in items:
        path = os.path.join(root_dir, item)
        if os.path.isdir(path) and item in IGNORED_DIRS:
            continue
        if os.path.isfile(path) and should_ignore_file(item):
            continue
        filtered_items.append(item)

    # trop volumineux → on coupe
    if len(filtered_items) > MAX_ITEMS:
        output_file.write(
            f"{indent}└── [contenu ignoré : {len(filtered_items)} éléments]\n"
        )
        return

    for index, item in enumerate(filtered_items):
        path = os.path.join(root_dir, item)
        is_last = index == len(filtered_items) - 1

        branch = "└── " if is_last else "├── "
        output_file.write(f"{indent}{branch}{item}\n")

        if os.path.isdir(path):
            new_indent = indent + ("    " if is_last else "│   ")
            generate_tree(path, output_file, new_indent)


def export_directory_tree(root_dir, output_path="arborescence.txt"):
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(f"{os.path.basename(os.path.abspath(root_dir))}\n")
        generate_tree(root_dir, f)


if __name__ == "__main__":
    dossier_a_analyser = "./"
    export_directory_tree(dossier_a_analyser)
