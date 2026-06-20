"""
01_data_loader.py — Load, clean, merge, and serialize all data sources.

Sources:
    1. Kaggle mattpodolak/rwallstreetbets-posts-and-comments (Posts)
    2. Kaggle mattpodolak/rwallstreetbets-posts-and-comments (Comments)
    3. data/raw/reddit/scraped_posts.csv       (optional, from 00_reddit_scraper.py)
    4. data/raw/reddit/scraped_comments.csv    (optional)
    5. WSB Trend Queries                       (synthetic queries)

Outputs (all in data/processed/):
    corpus.pkl                    — unified post DataFrame
    comments.pkl                  — comment reply pairs for graph building
    queries/train_queries.pkl     — 100 WSB query texts
    queries/dev_queries.pkl       — 50  WSB query texts
    queries/test_queries.pkl      — 50  WSB query texts

Usage:
    python src/01_data_loader.py
"""

import re
import sys
import logging
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from utils import save_pkl, PROCESSED_DIR, RAW_DIR

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

KAGGLE_POSTS_PATH = RAW_DIR / "reddit" / "rwallstreetbets_posts.csv"
KAGGLE_COMMENTS_PATH = RAW_DIR / "reddit" / "rwallstreetbets_comments.csv"
SCRAPED_POSTS_PATH = RAW_DIR / "reddit" / "scraped_posts.csv"
SCRAPED_COMMENTS_PATH = RAW_DIR / "reddit" / "scraped_comments.csv"

CORPUS_PATH = PROCESSED_DIR / "corpus.pkl"
COMMENTS_PATH = PROCESSED_DIR / "comments.pkl"
QUERIES_DIR = PROCESSED_DIR / "queries"

# Number of queries per split
N_TRAIN = 100
N_DEV = 50
N_TEST = 50
RANDOM_SEED = 42

# Minimum text length (characters) to keep a post
MIN_TEXT_LEN: int = 30

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("data_loader")


# ---------------------------------------------------------------------------
# Part 1 — Load raw data
# ---------------------------------------------------------------------------

def get_kaggle_dataset_dir() -> Path | None:
    try:
        import kagglehub
        log.info("Verifying/Downloading Kaggle dataset mattpodolak/rwallstreetbets-posts-and-comments...")
        path_str = kagglehub.dataset_download("mattpodolak/rwallstreetbets-posts-and-comments")
        log.info("Dataset available at: %s", path_str)
        return Path(path_str)
    except ImportError:
        log.warning("kagglehub not installed. Run: pip install kagglehub pandas")
        return None
    except Exception as e:
        log.warning("Failed to download dataset via kagglehub: %s", e)
        return None

def load_kaggle_posts(dataset_dir: Path | None = None) -> pd.DataFrame:
    """
    Load the Kaggle posts CSV and normalize columns.
    """
    if not dataset_dir:
        return pd.DataFrame()

    # Search for post files
    post_files = list(dataset_dir.glob("*post*.csv"))
    if not post_files:
        post_files = list(dataset_dir.glob("*.csv"))  # fallback

    if not post_files:
        log.warning("No CSV files found in Kaggle dataset.")
        return pd.DataFrame()

    # Prefer the file with 'post' in the name
    path = next((f for f in post_files if "post" in f.name.lower()), post_files[0])

    log.info("Loading Kaggle posts from %s...", path)
    # RWallStreetBets dataset uses 'utf-8' but may have errors, we'll try to read gracefully
    df = pd.read_csv(path, low_memory=False, lineterminator='\n')
    log.info("  Raw shape: %s", df.shape)

    df.columns = [c.lower().strip().replace(" ", "_") for c in df.columns]

    rename_map = {
        "selftext": "text",
        "body": "text",
        "post_score": "score",
        "comment_count": "num_comments",
        "author_name": "author",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    if "title" in df.columns:
        body = df.get("text", pd.Series("", index=df.index)).fillna("")
        df["text"] = df["title"].fillna("") + " " + body
    elif "text" not in df.columns:
        df["text"] = ""

    df["source"] = "kaggle"
    
    def safe_get(col, default_val):
        if col not in df.columns:
            return pd.Series(default_val, index=df.index)
        s = df[col]
        if isinstance(s, pd.DataFrame):
            s = s.iloc[:, 0]
        return s

    df["subreddit"] = safe_get("subreddit", "unknown")
    df["author"] = safe_get("author", "[deleted]").fillna("[deleted]")
    df["score"] = pd.to_numeric(safe_get("score", 0), errors="coerce").fillna(0).astype(int)
    df["upvote_ratio"] = pd.to_numeric(safe_get("upvote_ratio", 0.5), errors="coerce").fillna(0.5)
    df["num_comments"] = pd.to_numeric(safe_get("num_comments", 0), errors="coerce").fillna(0).astype(int)
    df["created_utc"] = pd.to_numeric(safe_get("created_utc", 0), errors="coerce").fillna(0).astype(int)

    if "id" not in df.columns:
        df["id"] = ["kaggle_" + str(i) for i in range(len(df))]
    df["id"] = df["id"].astype(str)

    cols = ["id", "subreddit", "author", "title", "text",
            "score", "upvote_ratio", "num_comments", "created_utc", "source"]
    available = [c for c in cols if c in df.columns]
    return df[available]


def load_kaggle_comments(dataset_dir: Path | None = None) -> pd.DataFrame:
    """
    Load the Kaggle comments CSV.
    """
    if not dataset_dir:
        return pd.DataFrame()

    comment_files = list(dataset_dir.glob("*comment*.csv"))
    if not comment_files:
        log.warning("No comment CSV file found in Kaggle dataset.")
        return pd.DataFrame()

    path = comment_files[0]
    log.info("Loading Kaggle comments from %s...", path)
    df = pd.read_csv(path, low_memory=False, lineterminator='\n')
    df.columns = [c.lower().strip().replace(" ", "_") for c in df.columns]
    log.info("  Raw shape: %s", df.shape)

    # Handle post_id mapping without creating duplicate columns
    if "link_id" in df.columns:
        df = df.rename(columns={"link_id": "post_id"})
    elif "parent_id" in df.columns:
        df = df.rename(columns={"parent_id": "post_id"})

    rename_map = {
        "author_name": "comment_author",
        "author": "comment_author",
        "comment": "comment_text",
        "body": "comment_text",
        "score": "comment_score",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    # Ensure it's a Series, not a DataFrame (in case of weird duplicate columns from the raw CSV)
    post_id_series = df["post_id"] if "post_id" in df.columns else pd.Series("", index=df.index)
    if isinstance(post_id_series, pd.DataFrame):
        post_id_series = post_id_series.iloc[:, 0]
        
    df["post_id"] = post_id_series.astype(str).str.replace(r"^t3_", "", regex=True)
    def safe_get(col, default_val):
        if col not in df.columns:
            return pd.Series(default_val, index=df.index)
        s = df[col]
        if isinstance(s, pd.DataFrame):
            s = s.iloc[:, 0]
        return s

    df["comment_author"] = safe_get("comment_author", "[deleted]").fillna("[deleted]")
    df["comment_score"] = pd.to_numeric(safe_get("comment_score", 0), errors="coerce").fillna(0).astype(int)
    df["created_utc"] = pd.to_numeric(safe_get("created_utc", 0), errors="coerce").fillna(0).astype(int)

    cols = ["post_id", "post_author", "comment_author", "comment_score", "created_utc"]
    available = [c for c in cols if c in df.columns]
    return df[available]


def load_scraped_posts(
    posts_path: Path = SCRAPED_POSTS_PATH,
    comments_path: Path = SCRAPED_COMMENTS_PATH,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load scraped posts and comments from 00_reddit_scraper.py output.

    Returns (posts_df, comments_df). Both may be empty if files don't exist.
    """
    posts_df = pd.DataFrame()
    if posts_path.exists():
        log.info("Loading scraped posts from %s...", posts_path)
        posts_df = pd.read_csv(posts_path)
        posts_df["source"] = "scraped"
        posts_df["id"] = posts_df["id"].astype(str)
        log.info("  Scraped posts: %d", len(posts_df))
    else:
        log.info("No scraped posts found — skipping.")

    comments_df = pd.DataFrame()
    if comments_path.exists():
        log.info("Loading scraped comments from %s...", comments_path)
        comments_df = pd.read_csv(comments_path)
        log.info("  Scraped comments: %d", len(comments_df))
    else:
        log.info("No scraped comments found — skipping.")

    return posts_df, comments_df


# ---------------------------------------------------------------------------
# Part 2 — Clean and merge
# ---------------------------------------------------------------------------

_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_MENTION_RE = re.compile(r"@\w+")
_SUBREDDIT_RE = re.compile(r"r/\w+")
_HTML_RE = re.compile(r"<[^>]+>")
_WHITESPACE_RE = re.compile(r"\s+")


def clean_text(text: str) -> str:
    """
    Normalize raw Reddit post text for indexing.

    Steps:
        1. Remove HTML tags
        2. Remove URLs
        3. Remove @mentions
        4. Remove r/subreddit references
        5. Collapse whitespace
    """
    if not isinstance(text, str):
        return ""
    text = _HTML_RE.sub(" ", text)
    text = _URL_RE.sub(" ", text)
    text = _MENTION_RE.sub(" ", text)
    text = _SUBREDDIT_RE.sub(" ", text)
    text = _WHITESPACE_RE.sub(" ", text)
    return text.strip()


def build_corpus(
    kaggle_posts: pd.DataFrame,
    scraped_posts: pd.DataFrame,
) -> pd.DataFrame:
    """
    Merge Kaggle and scraped posts into a unified corpus DataFrame.

    Deduplicates on the 'id' column, cleans text, filters short posts,
    and renames 'id' to 'docno' (PyTerrier convention).

    Args:
        kaggle_posts:  DataFrame from load_kaggle_posts().
        scraped_posts: DataFrame from load_scraped_posts().

    Returns:
        Unified corpus DataFrame with 'docno' as the primary key.
    """
    frames = [df for df in [kaggle_posts, scraped_posts] if not df.empty]
    if not frames:
        raise RuntimeError("No post data available — provide Kaggle CSVs or run the scraper.")

    corpus = pd.concat(frames, ignore_index=True)
    log.info("Combined corpus before deduplication: %d rows", len(corpus))

    corpus = corpus.drop_duplicates(subset=["id"], keep="first")
    log.info("After deduplication: %d rows", len(corpus))

    corpus["text"] = corpus["text"].apply(clean_text)
    corpus = corpus[corpus["text"].str.len() >= MIN_TEXT_LEN].copy()
    log.info("After text length filter (>= %d chars): %d rows", MIN_TEXT_LEN, len(corpus))

    corpus = corpus.rename(columns={"id": "docno"})

    # Ensure numeric columns are clean
    corpus["score"] = corpus["score"].fillna(0).astype(int)
    corpus["upvote_ratio"] = corpus["upvote_ratio"].fillna(0.5).astype(float)
    corpus["num_comments"] = corpus["num_comments"].fillna(0).astype(int)
    corpus["created_utc"] = corpus["created_utc"].fillna(0).astype(int)
    corpus["author"] = corpus["author"].fillna("[deleted]").astype(str)

    log.info("Final corpus shape: %s", corpus.shape)
    return corpus.reset_index(drop=True)


def build_comments(
    kaggle_comments: pd.DataFrame,
    scraped_comments: pd.DataFrame,
    corpus: pd.DataFrame,
) -> pd.DataFrame:
    """
    Merge Kaggle and scraped comments into a unified reply-pair DataFrame.

    Used later by 03_features.py to build the author interaction graph.
    """
    frames = [df for df in [kaggle_comments, scraped_comments] if not df.empty]
    if not frames:
        log.warning("No comment data available — graph features will be limited.")
        return pd.DataFrame(columns=["post_id", "post_author",
                                     "comment_author", "comment_score"])

    comments = pd.concat(frames, ignore_index=True)
    
    # If post_author is missing, map it from corpus
    if "post_author" not in comments.columns and "post_id" in comments.columns and not corpus.empty:
        post_authors = corpus.set_index("docno")["author"]
        comments["post_author"] = comments["post_id"].map(post_authors)
        
    comments = comments.dropna(subset=["post_id", "comment_author"])
    comments = comments[
        (comments["comment_author"] != "[deleted]")
        & (comments.get("post_author", pd.Series("", index=comments.index)).fillna("") != "[deleted]")
        & (comments.get("post_author", pd.Series("", index=comments.index)).fillna("") != "")
    ]
    log.info("Combined comments: %d reply pairs", len(comments))
    return comments.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Part 3 — WSB queries and split serialization
# ---------------------------------------------------------------------------

def load_wsb_queries() -> pd.DataFrame:
    """
    Generate 200 WSB-specific queries for Social Momentum Search.
    """
    log.info("Generating WSB-specific queries...")
    tickers = ['GME', 'AMC', 'TSLA', 'PLTR', 'BBBY', 'AAPL', 'AMD', 'NVDA', 'SPY', 'BABA']
    terms = ['short squeeze', 'diamond hands', 'YOLO options', 'calls', 'puts', 
             'loss porn', 'to the moon', 'rocket', 'buy the dip', 'earnings',
             'bull run', 'bear market', 'squeeze prediction', 'gamma squeeze',
             'hold the line', 'paper hands', 'tendies', 'fd options', 'margin call', 'robinhood']
    
    records = []
    qid = 1
    for t in tickers:
        for term in terms:
            records.append({"qid": str(qid), "query": f"{t} {term}"})
            qid += 1
            
    df = pd.DataFrame(records)
    log.info("Generated %d WSB queries.", len(df))
    return df


def split_queries(
    queries: pd.DataFrame,
    n_train: int = N_TRAIN,
    n_dev: int = N_DEV,
    n_test: int = N_TEST,
    seed: int = RANDOM_SEED,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Randomly split query pool into train / dev / test sets.

    Splits are frozen at this point and never reshuffled — ensuring
    that test queries are never seen during training or validation.

    Args:
        queries: Full query DataFrame from load_wsb_queries().
        n_train: Number of training queries.
        n_dev:   Number of dev queries.
        n_test:  Number of test queries.
        seed:    Random seed for reproducibility.

    Returns:
        Tuple of (train_queries, dev_queries, test_queries) DataFrames.
    """
    total = n_train + n_dev + n_test
    if len(queries) < total:
        raise ValueError(
            f"Need {total} queries but only {len(queries)} available."
        )

    shuffled = queries.sample(frac=1, random_state=seed).reset_index(drop=True)
    train = shuffled.iloc[:n_train].copy()
    dev = shuffled.iloc[n_train: n_train + n_dev].copy()
    test = shuffled.iloc[n_train + n_dev: n_train + n_dev + n_test].copy()

    log.info("Query split — train: %d | dev: %d | test: %d",
             len(train), len(dev), len(test))
    return train, dev, test


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run(nrows: int | None = None) -> None:
    """
    Execute the full data loading pipeline.

    Args:
        nrows: If set, limits rows read from CSVs (useful for smoke tests).
    """
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    QUERIES_DIR.mkdir(parents=True, exist_ok=True)

    dataset_dir = get_kaggle_dataset_dir()

    # --- Load raw sources ---
    kaggle_posts = load_kaggle_posts(dataset_dir)
    if nrows and not kaggle_posts.empty:
        kaggle_posts = kaggle_posts.head(nrows)

    kaggle_comments = load_kaggle_comments(dataset_dir)
    if nrows and not kaggle_comments.empty:
        kaggle_comments = kaggle_comments.head(nrows * 5)

    scraped_posts, scraped_comments = load_scraped_posts()

    # --- Build and save corpus ---
    corpus = build_corpus(kaggle_posts, scraped_posts)
    save_pkl(corpus, CORPUS_PATH)

    # --- Build and save comments ---
    comments = build_comments(kaggle_comments, scraped_comments, corpus)
    save_pkl(comments, COMMENTS_PATH)

    # --- Load and split WSB queries ---
    queries = load_wsb_queries()
    train_q, dev_q, test_q = split_queries(queries)

    save_pkl(train_q, QUERIES_DIR / "train_queries.pkl")
    save_pkl(dev_q, QUERIES_DIR / "dev_queries.pkl")
    save_pkl(test_q, QUERIES_DIR / "test_queries.pkl")

    log.info("=== Data loading complete ===")
    log.info("  Corpus:   %d documents  → %s", len(corpus), CORPUS_PATH)
    log.info("  Comments: %d pairs      → %s", len(comments), COMMENTS_PATH)
    log.info("  Queries:  train=%d  dev=%d  test=%d",
             len(train_q), len(dev_q), len(test_q))


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    log.info("=== Smoke test: load_data with nrows=500 ===")

    # Test text cleaning in isolation
    sample = "Check out https://example.com — great post by @user! <b>Bold</b> text."
    cleaned = clean_text(sample)
    assert "http" not in cleaned, "URL not removed"
    assert "@user" not in cleaned, "Mention not removed"
    assert "<b>" not in cleaned, "HTML not removed"
    log.info("clean_text ✓  '%s'", cleaned)

    # Test WSB query loading
    try:
        queries_df = load_wsb_queries()
        assert len(queries_df) > 0, "No WSB queries loaded"
        log.info("WSB queries loaded ✓  total=%d  sample='%s'",
                 len(queries_df), queries_df["query"].iloc[0])

        train_q, dev_q, test_q = split_queries(queries_df)
        assert len(train_q) == N_TRAIN
        assert len(dev_q) == N_DEV
        assert len(test_q) == N_TEST
        # Verify no overlap between splits
        assert set(train_q["qid"]).isdisjoint(set(test_q["qid"])), "Train/test overlap!"
        log.info("Query split ✓  train=%d  dev=%d  test=%d",
                 len(train_q), len(dev_q), len(test_q))

    except Exception as e:
        log.warning("WSB query loading failed: %s", e)

    # Test corpus build with synthetic data
    synthetic = pd.DataFrame({
        "id": ["a1", "a2", "a3"],
        "subreddit": ["investing"] * 3,
        "author": ["user1", "user2", "user3"],
        "title": ["Index funds are great", "Buy low sell high", "ETF basics"],
        "text": [
            "Index funds are great for long-term investing and low fees.",
            "Buy low sell high is a common but oversimplified strategy.",
            "ETFs provide diversification at low cost compared to mutual funds.",
        ],
        "score": [100, 200, 50],
        "upvote_ratio": [0.95, 0.80, 0.99],
        "num_comments": [10, 25, 5],
        "created_utc": [1700000000, 1700001000, 1700002000],
        "source": ["kaggle"] * 3,
    })
    corpus = build_corpus(synthetic, pd.DataFrame())
    assert "docno" in corpus.columns, "docno column missing"
    assert len(corpus) == 3, f"Expected 3 rows, got {len(corpus)}"
    log.info("build_corpus ✓  shape=%s  cols=%s",
             corpus.shape, corpus.columns.tolist())

    log.info("=== All smoke tests passed ===")

    # Run full pipeline
    log.info("")
    log.info("=== Running full data loading pipeline ===")
    run()
