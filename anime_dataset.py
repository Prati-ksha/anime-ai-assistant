"""
fetch_anime_data.py

Pulls anime data (2010-2020) from the AniList GraphQL API and caches raw
responses to local JSON files. Replaces the earlier Jikan-based version,
which was hitting frequent 504 Gateway Timeout errors.

AniList advantages for this use case:
- GraphQL means we filter by seasonYear directly in the query (no client-side
  date filtering needed)
- No API key required
- Generally more stable uptime than Jikan's public proxy

Usage:
    python fetch_anime_data.py

Output:
    data/raw/anime_<year>.json   -> one file per year, list of anime dicts
    data/anime_master.json        -> combined, deduped, cleaned dataset

AniList API docs: https://docs.anilist.co/
GraphQL playground: https://anilist.co/graphiql
"""

import json
import time
from pathlib import Path

import requests

# ---- Config ----
START_YEAR = 2015
END_YEAR = 2025 # inclusive
API_URL = "https://graphql.anilist.co"
RAW_DIR = Path("data/raw")
MASTER_FILE = Path("data/anime_master.json")

PER_PAGE = 50  # AniList max per page is 50
REQUEST_DELAY_SECONDS = 1.5  # AniList's public rate limit is modest; stay well under it
MAX_RETRIES = 5

# Only TV series by default, to match the earlier Jikan setup. Set to None
# to include all formats (movies, OVAs, specials, etc.)
FORMAT_FILTER = "TV"

QUERY = """
query ($page: Int, $perPage: Int, $year: Int, $format: MediaFormat) {
  Page(page: $page, perPage: $perPage) {
    pageInfo {
      hasNextPage
    }
    media(seasonYear: $year, type: ANIME, format: $format, sort: POPULARITY_DESC) {
      id
      title {
        romaji
        english
      }
      description(asHtml: false)
      genres
      tags {
        name
      }
      episodes
      seasonYear
      averageScore
      status
      studios(isMain: true) {
        nodes {
          name
        }
      }
      siteUrl
      coverImage {
        large
      }
    }
  }
}
"""


def fetch_year(year: int) -> list[dict]:
    """Fetch all anime for a given season year across paginated results."""
    cache_path = RAW_DIR / f"anime_{year}.json"
    if cache_path.exists():
        print(f"[cache] {year} already fetched, skipping API calls.")
        return json.loads(cache_path.read_text(), encoding="utf-8")

    print(f"[fetch] Pulling anime for {year}...")
    all_entries = []
    page = 1

    while True:
        variables = {
            "page": page,
            "perPage": PER_PAGE,
            "year": year,
            "format": FORMAT_FILTER,
        }

        retries = 0
        resp = None
        while retries < MAX_RETRIES:
            resp = requests.post(
                API_URL,
                json={"query": QUERY, "variables": variables},
                timeout=15,
            )

            if resp.status_code == 429:
                # AniList sends a Retry-After header when rate limited
                wait = int(resp.headers.get("Retry-After", 10))
                print(f"  Rate limited, backing off {wait}s...")
                time.sleep(wait)
                retries += 1
                continue

            if resp.status_code >= 500:
                wait = 5 
                print(f"  Server error {resp.status_code}, retry {retries+1}/{MAX_RETRIES} in {wait}s...")
                time.sleep(wait)
                retries += 1
                continue

            break
        else:
            print(f"  Gave up on {year} page {page} after {MAX_RETRIES} retries. Stopping this year.")
            break

        resp.raise_for_status()
        payload = resp.json()

        if "errors" in payload:
            print(f"  GraphQL error on {year} page {page}: {payload['errors']}")
            break

        page_data = payload["data"]["Page"]
        batch = page_data["media"]
        all_entries.extend(batch)

        has_next = page_data["pageInfo"]["hasNextPage"]
        print(f"  page {page}: +{len(batch)} entries (total so far: {len(all_entries)})")

        if not has_next:
            break

        page += 1
        time.sleep(REQUEST_DELAY_SECONDS)

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(all_entries, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[saved] {year}: {len(all_entries)} entries -> {cache_path}")
    return all_entries


def clean_entry(entry: dict) -> dict | None:
    """Extract only the fields we need for the RAG app. Skip entries with no synopsis."""
    synopsis = entry.get("description")
    if not synopsis or synopsis.strip() == "":
        return None

    # AniList descriptions sometimes contain leftover HTML tags like <br> or <i>
    # even when asHtml: false is set for some fields; strip the common ones.
    synopsis = (
        synopsis.replace("<br>", " ")
        .replace("<br/>", " ")
        .replace("<i>", "")
        .replace("</i>", "")
        .strip()
    )

    title_obj = entry.get("title") or {}
    title = title_obj.get("english") or title_obj.get("romaji")

    studios = entry.get("studios", {}).get("nodes", [])
    tags = entry.get("tags", [])

    return {
        "anilist_id": entry.get("id"),
        "title": title,
        "title_romaji": title_obj.get("romaji"),
        "synopsis": synopsis,
        "genres": entry.get("genres", []),
        "tags": [t["name"] for t in tags[:10]],  # top 10 tags, avoids noisy long tail
        "episodes": entry.get("episodes"),
        "year": entry.get("seasonYear"),
        "score": entry.get("averageScore"),
        "status": entry.get("status"),
        "studios": [s["name"] for s in studios],
        "url": entry.get("siteUrl"),
        "image_url": (entry.get("coverImage") or {}).get("large"),
    }


def main():
    all_years_data = []

    for year in range(START_YEAR, END_YEAR + 1):
        year_data = fetch_year(year)
        all_years_data.extend(year_data)

    print(f"\nTotal raw entries fetched: {len(all_years_data)}")

    # Clean + dedupe by anilist_id
    seen_ids = set()
    cleaned = []
    for entry in all_years_data:
        clean = clean_entry(entry)
        if clean is None:
            continue
        if clean["anilist_id"] in seen_ids:
            continue
        seen_ids.add(clean["anilist_id"])
        cleaned.append(clean)

    MASTER_FILE.parent.mkdir(parents=True, exist_ok=True)
    MASTER_FILE.write_text(json.dumps(cleaned, ensure_ascii=False, indent=2), encoding= "utf-8")

    print(f"Cleaned + deduped entries (with synopsis): {len(cleaned)}")
    print(f"Saved master dataset -> {MASTER_FILE}")


if __name__ == "__main__":
    main()