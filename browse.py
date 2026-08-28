"""
browse.py

Direct, exact lookups against anime_master.json -- NOT RAG, NOT semantic
search, NOT the LLM. Use this whenever the user wants a complete, accurate
list (e.g. "what anime exist from 2016-2018") rather than a relevance-ranked
answer to a specific question.

Why this is separate from rag_chain.py:
Semantic retrieval (the vector store) always returns the top-k closest
matches to a query -- it was never designed to answer "list everything."
For exhaustive/browsable listings, a plain filter over the source data is
faster, free (no LLM call), and 100% accurate -- no risk of the model only
showing a partial or arbitrary subset.
"""

import json
from pathlib import Path

MASTER_FILE = Path("data/anime_master.json")


def load_all_anime() -> list[dict]:
    return json.loads(MASTER_FILE.read_text(encoding="utf-8"))


def list_titles_by_year(start_year: int | None = None, end_year: int | None = None) -> list[str]:
    """
    Returns every anime title in the dataset within [start_year, end_year]
    (inclusive). Pass None for either bound to leave it open-ended.
    """
    records = load_all_anime()
    titles = []

    for r in records:
        year = r.get("year")
        if year is None:
            continue
        if start_year is not None and year < start_year:
            continue
        if end_year is not None and year > end_year:
            continue
        titles.append((year, r.get("title") or r.get("title_romaji")))

    titles.sort(key=lambda t: (t[0], t[1] or ""))
    return [f"{title} ({year})" for year, title in titles]


def list_titles_by_genre(genre: str) -> list[str]:
    """Returns every anime title whose genre list contains the given genre (case-insensitive)."""
    records = load_all_anime()
    genre_lower = genre.lower()
    matches = [
        r.get("title") or r.get("title_romaji")
        for r in records
        if any(g.lower() == genre_lower for g in r.get("genres", []))
    ]
    return sorted(matches)


def dataset_summary() -> dict:
    """Quick stats about what's actually in the dataset -- useful for a new user."""
    records = load_all_anime()
    years = sorted({r["year"] for r in records if r.get("year")})
    all_genres = sorted({g for r in records for g in r.get("genres", [])})

    return {
        "total_anime": len(records),
        "year_range": f"{years[0]}-{years[-1]}" if years else "unknown",
        "available_genres": all_genres,
    }


if __name__ == "__main__":
    # Quick manual test
    summary = dataset_summary()
    print(f"Total anime: {summary['total_anime']}")
    print(f"Year range: {summary['year_range']}")
    print(f"Genres available: {', '.join(summary['available_genres'])}")

    print("\nFirst 10 titles from 2016-2018:")
    for title in list_titles_by_year(2016, 2018)[:10]:
        print(f"  {title}")
