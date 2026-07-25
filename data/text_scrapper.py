import random
import requests
from bs4 import BeautifulSoup
from core import normalize
import os

WIKI_BASE = "https://fr.wikipedia.org"
name = "scrapped"


def get_article_text(url: str):
    response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")
    content = soup.find("div", {"id": "mw-content-text"})
    if content is None:
        return "", []

    paragraphs = content.find_all("p")
    text = " ".join(p.get_text(" ", strip=True) for p in paragraphs)
    links: list[str] = []
    for a in content.find_all("a", href=True):
        href: str = str(a["href"])

        if href.startswith("/wiki/") and ":" not in href and "#" not in href:
            links.append(WIKI_BASE + href)

    return text, links


def scrap_text(length: int, offset: int = 0, must_normalize: bool = False, filename: str | None = None) -> str:
    directory = os.path.join("data", "text")
    if filename is None:
        filename = name
    path = os.path.join(directory, filename + ".txt")

    try:
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
        words = text.split(" ")
        if len(words) >= length + offset:
            return " ".join(words[offset : offset + length])
    except FileNotFoundError:
        print("No text file found")
    except Exception as e:
        print(f"Error while reading text: {e}")

    words: list[str] = []

    current_url = "https://fr.wikipedia.org/wiki/Sp%C3%A9cial:Page_au_hasard"
    visited = set()
    while len(words) < length + offset:
        try:
            text, links = get_article_text(current_url)
            if must_normalize:
                text = normalize(text)
            if text:
                words.extend(text.split())
            visited.add(current_url)
            links = [link for link in links if link not in visited]
            if links:
                current_url = random.choice(links)
            else:
                current_url = "https://fr.wikipedia.org/wiki/Sp%C3%A9cial:Page_au_hasard"
            print(f"{len(words)}/{length + offset} mots collectés", end="\r")
        except Exception:
            current_url = "https://fr.wikipedia.org/wiki/Sp%C3%A9cial:Page_au_hasard"

    final_text = " ".join(words[offset : offset + length])
    try:
        os.mkdir(directory)
    except FileExistsError:
        print(directory, "already exists")
    with open(path, "w", encoding="utf-8") as f:
        f.write(final_text)
    print("text saved as", path)
    return final_text
