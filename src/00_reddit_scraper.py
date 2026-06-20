"""
00_reddit_scraper.py — Reddit API data collection via praw.

Collects posts and their top-level comments from financial subreddits.
Designed to be resilient: supports checkpointing, rate limiting, and
idempotent re-runs (skips already-scraped post IDs).

Output:
    data/raw/reddit/scraped_posts.csv
    data/raw/reddit/scraped_comments.csv

Usage:
    python src/00_reddit_scraper.py
"""

import os
import time
import logging
from pathlib import Path

import pandas as pd
import praw

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SUBREDDITS: list[str] = [
    "personalfinance",
    "investing",
    "stocks",
]

POSTS_LIMIT: int = 500          # Max posts to fetch per subreddit
COMMENTS_PER_POST: int = 10     # Top-level comments to collect per post
CHECKPOINT_EVERY: int = 100     # Save progress every N posts
RATE_LIMIT_SLEEP: float = 2.0   # Seconds to sleep between API calls

RAW_DIR = Path("data/raw/reddit")
POSTS_PATH = RAW_DIR / "scraped_posts.csv"
COMMENTS_PATH = RAW_DIR / "scraped_comments.csv"

# Columns for output DataFrames
POST_COLS: list[str] = [
    "id", "subreddit", "author", "title", "text",
    "score", "upvote_ratio", "num_comments", "created_utc", "url",
]
COMMENT_COLS: list[str] = [
    "comment_id", "post_id", "post_author", "comment_author",
    "comment_text", "comment_score", "created_utc",
]

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("scraper")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_existing_ids(path: Path) -> set[str]:
    """
    Load already-scraped post IDs from checkpoint CSV.

    Returns an empty set if no checkpoint exists yet.
    """
    if path.exists():
        df = pd.read_csv(path, usecols=["id"])
        ids = set(df["id"].astype(str))
        log.info("Checkpoint found — skipping %d already-scraped posts.", len(ids))
        return ids
    return set()


def _save_checkpoint(records: list[dict], path: Path, cols: list[str]) -> None:
    """
    Append a batch of records to the checkpoint CSV.

    Creates the file with headers on the first write; appends without
    headers on subsequent writes.
    """
    if not records:
        return
    df = pd.DataFrame(records, columns=cols)
    write_header = not path.exists()
    df.to_csv(path, mode="a", header=write_header, index=False)
    log.info("Checkpoint saved — %d new records → %s", len(records), path)


def _build_reddit_client() -> praw.Reddit:
    """
    Instantiate a read-only Reddit client.

    Reads credentials from praw.ini or environment variables:
        REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET, REDDIT_USER_AGENT
    """
    client_id = os.getenv("REDDIT_CLIENT_ID")
    client_secret = os.getenv("REDDIT_CLIENT_SECRET")
    user_agent = os.getenv("REDDIT_USER_AGENT", "FinTrustSearch/1.0")

    if client_id and client_secret:
        log.info("Using Reddit credentials from environment variables.")
        return praw.Reddit(
            client_id=client_id,
            client_secret=client_secret,
            user_agent=user_agent,
        )

    log.info("Using Reddit credentials from praw.ini.")
    return praw.Reddit("DEFAULT")  # reads from praw.ini [DEFAULT] section


# ---------------------------------------------------------------------------
# Core scraping functions
# ---------------------------------------------------------------------------

def scrape_posts(
    reddit: praw.Reddit,
    subreddit_name: str,
    limit: int = POSTS_LIMIT,
    existing_ids: set[str] | None = None,
) -> tuple[list[dict], list[dict]]:
    """
    Scrape hot posts and their top-level comments from one subreddit.

    Args:
        reddit:         Authenticated praw.Reddit instance.
        subreddit_name: Name of the subreddit (without r/).
        limit:          Maximum number of posts to attempt to fetch.
        existing_ids:   Set of post IDs to skip (idempotency).

    Returns:
        Tuple of (post_records, comment_records) as lists of dicts.
    """
    if existing_ids is None:
        existing_ids = set()

    post_batch: list[dict] = []
    comment_batch: list[dict] = []
    batch_count = 0

    log.info("Scraping r/%s (limit=%d)...", subreddit_name, limit)
    subreddit = reddit.subreddit(subreddit_name)

    for post in subreddit.hot(limit=limit):
        if post.id in existing_ids:
            continue

        try:
            # --- Post record ---
            post_record: dict = {
                "id": post.id,
                "subreddit": subreddit_name,
                "author": str(post.author) if post.author else "[deleted]",
                "title": post.title,
                "text": post.selftext or "",
                "score": post.score,
                "upvote_ratio": post.upvote_ratio,
                "num_comments": post.num_comments,
                "created_utc": int(post.created_utc),
                "url": post.url,
            }
            post_batch.append(post_record)

            # --- Comment records ---
            post.comments.replace_more(limit=0)  # avoid extra API calls
            for comment in post.comments[:COMMENTS_PER_POST]:
                comment_record: dict = {
                    "comment_id": comment.id,
                    "post_id": post.id,
                    "post_author": post_record["author"],
                    "comment_author": (
                        str(comment.author) if comment.author else "[deleted]"
                    ),
                    "comment_text": comment.body or "",
                    "comment_score": comment.score,
                    "created_utc": int(comment.created_utc),
                }
                comment_batch.append(comment_record)

            existing_ids.add(post.id)
            batch_count += 1

            # --- Checkpoint ---
            if batch_count % CHECKPOINT_EVERY == 0:
                _save_checkpoint(post_batch, POSTS_PATH, POST_COLS)
                _save_checkpoint(comment_batch, COMMENTS_PATH, COMMENT_COLS)
                post_batch.clear()
                comment_batch.clear()
                log.info("Progress: %d posts scraped from r/%s.",
                         batch_count, subreddit_name)

            time.sleep(RATE_LIMIT_SLEEP)

        except Exception as exc:  # noqa: BLE001
            log.warning("[SKIP] post %s — %s", post.id, exc)
            continue

    # Save any remaining records
    _save_checkpoint(post_batch, POSTS_PATH, POST_COLS)
    _save_checkpoint(comment_batch, COMMENTS_PATH, COMMENT_COLS)

    return post_batch, comment_batch


def run_scraper(
    subreddits: list[str] = SUBREDDITS,
    limit: int = POSTS_LIMIT,
) -> None:
    """
    Main entry point — scrape all configured subreddits.

    Resumes automatically from the last checkpoint if the output CSV
    already exists (idempotent re-runs).

    Args:
        subreddits: List of subreddit names to scrape.
        limit:      Max posts per subreddit.
    """
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    reddit = _build_reddit_client()
    existing_ids = _load_existing_ids(POSTS_PATH)

    for sub in subreddits:
        scrape_posts(reddit, sub, limit=limit, existing_ids=existing_ids)

    # --- Final summary ---
    if POSTS_PATH.exists():
        total_posts = len(pd.read_csv(POSTS_PATH))
        log.info("Done. Total posts saved: %d → %s", total_posts, POSTS_PATH)
    if COMMENTS_PATH.exists():
        total_comments = len(pd.read_csv(COMMENTS_PATH))
        log.info("Done. Total comments saved: %d → %s",
                 total_comments, COMMENTS_PATH)


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    log.info("=== Smoke test: scrape 5 posts from r/investing ===")
    log.info(
        "NOTE: This requires valid Reddit API credentials in praw.ini "
        "or environment variables REDDIT_CLIENT_ID / REDDIT_CLIENT_SECRET."
    )

    try:
        reddit_client = _build_reddit_client()

        # Scrape a tiny sample — 5 posts, 3 comments each
        COMMENTS_PER_POST = 3
        posts, comments = scrape_posts(
            reddit_client,
            subreddit_name="investing",
            limit=5,
        )

        log.info("Posts scraped in this batch: %d", len(posts))
        log.info("Comments scraped in this batch: %d", len(comments))

        if posts:
            log.info("Sample post: id=%s author=%s score=%s",
                     posts[0]["id"], posts[0]["author"], posts[0]["score"])

        log.info("Smoke test PASSED.")

    except Exception as e:
        log.error(
            "Smoke test failed — likely missing credentials: %s\n"
            "To run the full scraper you need a praw.ini file or env vars.\n"
            "The Kaggle CSV dataset will be used as the primary data source.",
            e,
        )
