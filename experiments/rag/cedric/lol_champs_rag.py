import os
from api import prompt_str, select_model
from rag import RAG
import requests
from bs4 import BeautifulSoup
from bs4.element import Tag

BASE_URL = "https://wiki.leagueoflegends.com"
RAG_MODEL_NAME = "intfloat/e5-large"
OUT_FILE = "out/lol_champs_rag.pkl"


def create_rag():
    """
    Create and save a RAG instance for League of Legends champions' skills.
    """

    def remove_attributes(tag):
        # Remove attributes from the current tag
        if hasattr(tag, "attrs"):
            tag.attrs = {}
        
        # Recursively remove attributes or images from child tags
        for child in tag.children:
            if isinstance(child, Tag):
                if child.name == "img":
                    child.decompose()
                else:
                    remove_attributes(child)
        
        return tag

    # Retrieve the list of League of Legends champions from the official wiki
    page = requests.get(BASE_URL + "/en-us/List_of_champions")
    soup = BeautifulSoup(page.content, "html.parser")

    # Find the champion names
    champions_urls = []
    table = soup.find("table", {"class": "article-table"})
    tbody = table.find("tbody")
    rows = tbody.find_all("tr")
    for row in rows:
        try:
            td = row.find("td")
            span = td.find("span")
            a = span.find("a")
            href = a["href"]
            champions_urls.append(href)
        except Exception:
            continue

    # Retrieve pages
    content = []
    for i, url in enumerate(champions_urls):
        print(f"Fetching: '{url}'  ({i+1}/{len(champions_urls)})")
        page = requests.get(BASE_URL + url)
        soup = BeautifulSoup(page.content, "html.parser")

        # Retrieve each skills
        content_div = soup.find("div", {"id": "content"})
        skill_divs = content_div.find_all("div", {"class": "skill"})
        for skill_div in skill_divs:

            # Retrieve the skill descriptions
            skill_descriptions = skill_div.find_all("div", {"class": "ability-info-description"})
            if len(skill_descriptions) == 0: continue  # Skip empty descriptions

            # Build the content
            text = f"{url.split('/')[-1].replace('_', ' ').replace('%27', '\'')}  :  "
            for description in skill_descriptions:
                description = remove_attributes(description)
                text += description.get_text(separator=" ", strip=True)
            content.append(text)

            print(f"  -> Skill added. Total length: {len(text)} characters.")
        print("")

    # Create the RAG instance
    rag = RAG(content, model_name=RAG_MODEL_NAME)

    # Save the RAG instance
    os.makedirs("out", exist_ok=True)
    rag.save(OUT_FILE)


def main():
    """
    Main function to run the RAG-based assistant for League of Legends champions.
    """

    if not os.path.exists(OUT_FILE):
        create_rag()
    
    # Initialize the instance
    rag = RAG.load(OUT_FILE)
    model = select_model()

    # Process user queries
    while True:
        prompt = input("[User]  ")

        response = prompt_str(model,
            "You are an assistant that provides information about League of Legends champions and their skills based on the provided context.\n\n" + \
            "<context>\n" + \
            "\n\n".join(rag.search_texts(prompt, k=5)) + \
            "\n</context>\n\n" +
            prompt
        )
        print(f"\n[Assistant]  {response}\n")

if __name__ == "__main__":
    main()
