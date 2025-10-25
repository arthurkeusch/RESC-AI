from api import main as api_main
import requests
from bs4 import BeautifulSoup
from bs4.element import Tag

BASE_URL = "https://wiki.leagueoflegends.com"


def main():

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

    # Load the API's main with the champion pages as texts
    api_main(content)


if __name__ == "__main__":
    main()
