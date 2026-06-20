"""
02_indexer.py — Build a PyTerrier inverted index from the Reddit corpus.

Reads corpus.pkl produced by 01_data_loader.py and builds a Terrier
IterDictIndexer index with Stopwords and PorterStemmer term pipelines.

Prerequisites:
    - Java 11+ installed and on PATH  (verify: java -version)
    - corpus.pkl must exist in data/processed/

Output:
    index/   — PyTerrier inverted index directory

Usage:
    python src/02_indexer.py
"""

import os
import sys
import logging
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from utils import load_pkl, get_logger, PROCESSED_DIR, INDEX_DIR

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CORPUS_PATH = PROCESSED_DIR / "corpus.pkl"
INDEX_PATH  = INDEX_DIR.resolve()         # absolute path — required by Terrier

# Terrier term pipeline: stopword removal + Porter stemming
TERM_PIPELINES = "Stopwords,PorterStemmer"

# BM25 parameters (explicit for reproducibility)
BM25_K1 = 1.2
BM25_B  = 0.75

# Columns required for indexing
REQUIRED_COLS = ["docno", "text"]

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

log = get_logger("indexer")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _init_pyterrier() -> None:
    """
    Initialize PyTerrier JVM (downloads Terrier JAR on first run).
    Compatible with PyTerrier >= 0.11.
    """

    import pyterrier as pt
    if not pt.java.started():
        log.info("Initializing PyTerrier...")
        pt.java.init()
    log.info("PyTerrier ready — Terrier version: %s", pt.terrier.version())


def print_index_stats(index) -> None:
    """Log collection statistics for the built index."""
    stats = index.getCollectionStatistics()
    log.info("=== Index Statistics ===")
    log.info("  Documents   : %d", stats.getNumberOfDocuments())
    log.info("  Tokens      : %d", stats.getNumberOfTokens())
    log.info("  Unique terms: %d", stats.getNumberOfUniqueTerms())
    log.info("  Pointers    : %d", stats.getNumberOfPointers())


# ---------------------------------------------------------------------------
# Core indexing function
# ---------------------------------------------------------------------------

def build_index(
    corpus: pd.DataFrame,
    index_path: Path = INDEX_PATH,
    term_pipelines: str = TERM_PIPELINES,
    overwrite: bool = True,
) -> object:
    """
    Build a PyTerrier IterDictIndexer index from the corpus DataFrame.

    Args:
        corpus:         DataFrame with at minimum 'docno' and 'text' columns.
        index_path:     Absolute directory path to store the Terrier index.
        term_pipelines: Comma-separated Terrier term pipeline components.
        overwrite:      If True, rebuilds the index even if it already exists.

    Returns:
        Loaded Terrier index object.

    Raises:
        ValueError: If required columns are missing from the corpus.
    """
    import pyterrier as pt

    missing = [c for c in REQUIRED_COLS if c not in corpus.columns]
    if missing:
        raise ValueError(
            f"Corpus is missing required columns: {missing}. "
            f"Available: {corpus.columns.tolist()}"
        )

    index_path = Path(index_path).resolve()
    
    if overwrite and index_path.exists():
        import shutil
        log.info("Removing existing index directory to prevent Windows locking issues: %s", index_path)
        shutil.rmtree(index_path, ignore_errors=True)
        
    index_path.mkdir(parents=True, exist_ok=True)

    log.info("Building index from %d documents → %s", len(corpus), index_path)

    # Prepare list of dicts — required format for IterDictIndexer
    records = [
        {"docno": str(row["docno"]), "text": str(row["text"]) if row["text"] else ""}
        for _, row in corpus[["docno", "text"]].iterrows()
    ]

    indexer = pt.terrier.IterDictIndexer(
        str(index_path),
        overwrite=overwrite,
        meta={"docno": 20},
    )
    indexer.setProperty("termpipelines", term_pipelines)

    index_ref = indexer.index(records)
    index = pt.IndexFactory.of(index_ref)

    log.info("Index built successfully.")
    print_index_stats(index)
    return index


def load_index(index_path: Path = INDEX_PATH) -> object:
    """
    Load an existing PyTerrier index from disk.

    Args:
        index_path: Directory containing the Terrier index.

    Returns:
        Loaded Terrier index object.

    Raises:
        FileNotFoundError: If the index does not exist.
    """
    import pyterrier as pt

    index_path = Path(index_path).resolve()
    data_props = index_path / "data.properties"
    if not data_props.exists():
        raise FileNotFoundError(
            f"No index found at '{index_path}'. "
            "Run 02_indexer.py first to build the index."
        )
    log.info("Loading existing index from %s...", index_path)
    index = pt.IndexFactory.of(str(data_props))
    print_index_stats(index)
    return index


def get_bm25_retriever(index, num_results: int = 1000) -> object:
    """
    Return a configured BM25 BatchRetrieve transformer.

    Args:
        index:       Loaded Terrier index object.
        num_results: Maximum number of results to return per query.

    Returns:
        pt.BatchRetrieve instance with BM25 weighting model.
    """
    import pyterrier as pt
    return pt.BatchRetrieve(
        index,
        wmodel="BM25",
        controls={"c": BM25_B, "bm25.k_1": BM25_K1},
        num_results=num_results,
        verbose=False,
    )


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run() -> None:
    """Load corpus and build the full PyTerrier index."""
    _init_pyterrier()

    log.info("Loading corpus from %s...", CORPUS_PATH)
    corpus = load_pkl(CORPUS_PATH)
    log.info("Corpus loaded — %d documents.", len(corpus))

    build_index(corpus)
    log.info("=== Indexing complete. Index ready at: %s ===", INDEX_PATH)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    run()

