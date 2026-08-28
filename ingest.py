from langchain_openai import OpenAI
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

import json
from pathlib import Path

from langchain_core.documents import Document
from langchain_chroma import Chroma

load_dotenv() 

MASTER_FILE = Path("data/anime_master.json")
PERSIST_DIR = "chroma_db"
COLLECTION_NAME = "anime"

# Batch size for adding documents to Chroma. Keeps memory use predictable
# and gives you visible progress instead of one giant silent call.
BATCH_SIZE = 100


def sanitize_metadata(value):
    """
    Chroma metadata values must be str, int, float, or bool -- no None,
    no lists. Convert anything else into a safe scalar.
    """
    if value is None:
        return ""
    if isinstance(value, list):
        return ", ".join(str(v) for v in value) if value else ""
    if isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def build_documents(records: list[dict]) -> list[Document]:
    """Convert cleaned anime records into LangChain Document objects."""
    documents = []

    for record in records:
        synopsis = record.get("synopsis", "").strip()
        if not synopsis:
            continue  # should already be filtered out by fetch step, but double-check

        title = record.get("title") or record.get("title_romaji") or "Unknown title"

        metadata = {
            "id": record.get("anilist_id"),
            "title": title,
            "title_romaji": record.get("title_romaji"),
            "year": record.get("year"),
            "episodes": record.get("episodes"),
            "score": record.get("score"),
            "status": record.get("status"),
            "genres": record.get("genres", []),
            "tags": record.get("tags", []),
            "studios": record.get("studios", []),
            "url": record.get("url"),
        }
        # Sanitize every value so Chroma never rejects the batch over one bad field
        metadata = {k: sanitize_metadata(v) for k, v in metadata.items()}

        # Prefix the content with the title -- helps semantic search match
        # queries that mention the anime by name, not just plot descriptions
        page_content = f"{title}\n\n{synopsis}"

        documents.append(Document(page_content=page_content, metadata=metadata))
        
    return documents


def main():
    if not MASTER_FILE.exists():
        raise FileNotFoundError(
            f"{MASTER_FILE} not found. Run fetch_anime_data.py first."
        )

    print(f"Loading {MASTER_FILE}...")
    records = json.loads(MASTER_FILE.read_text(encoding="utf-8"))
    print(f"Loaded {len(records)} anime records.")

    print("Building Document objects...")
    documents = build_documents(records)
    print(f"Built {len(documents)} documents (dropped {len(records) - len(documents)} with no synopsis).")

    if documents:
        print("\n--- Preview: first document ---")
        print("page_content:")
        print(documents[0].page_content)
        print("\nmetadata:")
        for key, value in documents[0].metadata.items():
            print(f"  {key}: {value}")
        print("--- end preview ---\n")

    print("Setting up OpenAI embeddings + Chroma...")
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    vectorstore = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=PERSIST_DIR,
    )

    print(f"Embedding and storing {len(documents)} documents in batches of {BATCH_SIZE}...")
    for i in range(0, len(documents), BATCH_SIZE):
        batch = documents[i:i + BATCH_SIZE]
        vectorstore.add_documents(batch)
        done = min(i + BATCH_SIZE, len(documents))
        print(f"  {done}/{len(documents)} documents embedded and stored")

    print(f"\nDone. Vector store persisted to ./{PERSIST_DIR}/")
    print(f"Collection '{COLLECTION_NAME}' now contains {len(documents)} anime.")


if __name__ == "__main__":
    main()