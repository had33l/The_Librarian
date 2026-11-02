
from pathlib import Path
from bs4 import BeautifulSoup
import json

root = Path("python-3.12-docs-html")
if not root.exists():
    raise RuntimeError(f"Documentation folder not found: {root}")

output = Path("data/processed/documents.json")
output.parent.mkdir(parents=True, exist_ok=True)

documents = []
excluded_names = {
    "contents.html",
    "genindex.html",
    "genindex-all.html",
    "py-modindex.html",
    "search.html",
}

for path in root.rglob("*.html"):
    if any(part in {"_static", "_sources"} for part in path.parts):
        continue

    if path.name in excluded_names or path.name.startswith("genindex-"):
        continue

    try:
        soup = BeautifulSoup(
            path.read_text(encoding="utf-8"),
            "html.parser"
        )

        # Remove page elements that are not the actual content
        for tag in soup([
            "script", "style", "nav", "footer", "header"
        ]):
            tag.decompose()

        main = soup.find("main") or soup.find("div", role="main")
        if main:
            content = main
        else:
            content = soup.body or soup

        text = " ".join(content.get_text(" ", strip=True).split())

        if len(text) < 200:
            continue

        title_tag = soup.find("title")
        title = (
            title_tag.get_text(" ", strip=True)
            if title_tag else path.stem
        )

        documents.append({
            "doc_id": path.relative_to(root).as_posix(),
            "title": title,
            "url": (
                "https://docs.python.org/3.12/"
                + path.relative_to(root).as_posix()
            ),
            "text": text
        })

    except Exception as error:
        print(f"Skipping {path}: {error}")

output.write_text(
    json.dumps(documents, indent=2, ensure_ascii=False),
    encoding="utf-8"
)

print("Usable documents:", len(documents))
print("Total extracted characters:",
      sum(len(doc["text"]) for doc in documents))
print("Saved to", output)
