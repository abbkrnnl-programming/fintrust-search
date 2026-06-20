"""
search_service.py — Facade class for FinTrust Search pipeline.

Provides a single clean interface that wraps the full retrieval and
re-ranking pipeline. Designed for future UI integration (Gradio, Flask,
or any frontend) without exposing internal pipeline details.

Usage:
    from src.search_service import SearchService

    service = SearchService()
    service.load()
    results = service.search("how to invest in index funds", top_k=10)
    print(results)
"""

import os
import sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.utils import load_pkl, get_logger, PROCESSED_DIR

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

ALL_FEATURES_PATH = PROCESSED_DIR / "all_features.pkl"
CORPUS_PATH       = PROCESSED_DIR / "corpus.pkl"
BASE_DIR          = Path(__file__).resolve().parent.parent
MODELS_DIR        = BASE_DIR / "models"
MODEL_PATH        = MODELS_DIR / "ltr_model.pkl"
SCALER_PATH       = MODELS_DIR / "ltr_scaler.pkl"
INDEX_DIR         = BASE_DIR / "index"

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

log = get_logger("search_service")


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class SearchResult:
    """
    A single search result with retrieval and document metadata.

    Attributes:
        rank:        Position in the ranked list (1-based).
        docno:       Document identifier in the corpus.
        score:       Final retrieval score (higher = more relevant).
        text:        Document body (truncated preview).
        author:      Reddit post author username.
        subreddit:   Subreddit the post was published in.
        post_score:  Raw Reddit post score (upvotes - downvotes).
        upvote_ratio: Fraction of upvotes over total votes.
        num_comments: Number of comments on the post.
    """
    rank:         int
    docno:        str
    score:        float
    text:         str
    author:       str      = "unknown"
    subreddit:    str      = "unknown"
    post_score:   int      = 0
    upvote_ratio: float    = 0.5
    num_comments: int      = 0


# ---------------------------------------------------------------------------
# SearchService facade
# ---------------------------------------------------------------------------

class SearchService:
    """
    Facade class wrapping the full FinTrust Search retrieval pipeline.

    Encapsulates:
        - PyTerrier BM25 index
        - Document feature matrix (social + graph + content)
        - Trained Logistic Regression LTR model
        - Corpus metadata for result enrichment

    Example:
        >>> service = SearchService()
        >>> service.load()
        >>> results = service.search("index funds vs ETFs", top_k=5)
        >>> for r in results:
        ...     print(r.rank, r.author, r.score)
    """

    def __init__(self) -> None:
        self._index        = None
        self._bm25         = None
        self._ltr_model    = None
        self._ltr_scaler   = None
        self._doc_features: Optional[pd.DataFrame] = None
        self._corpus:       Optional[pd.DataFrame] = None
        self._corpus_lookup: dict = {}
        self._loaded = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def load(self) -> "SearchService":
        """
        Load all pipeline artifacts into memory.

        Must be called before search(). Safe to call multiple times
        (subsequent calls are no-ops if already loaded).

        Returns:
            Self for fluent chaining: service = SearchService().load()

        Raises:
            FileNotFoundError: If the index or model files are missing.
            RuntimeError:      If PyTerrier / Java initialization fails.
        """
        if self._loaded:
            return self

        self._init_java()

        import pyterrier as pt
        if not pt.java.started():
            pt.java.init()

        import importlib
        indexer_mod = importlib.import_module("src.02_indexer")
        load_index = indexer_mod.load_index
        get_bm25_retriever = indexer_mod.get_bm25_retriever
        reranker_mod = importlib.import_module("src.05_reranker")
        load_model = reranker_mod.load_model

        self._index        = load_index(INDEX_DIR)
        self._bm25         = get_bm25_retriever(self._index, num_results=100)
        self._ltr_model, self._ltr_scaler = load_model()
        self._doc_features = load_pkl(ALL_FEATURES_PATH)
        self._corpus       = load_pkl(CORPUS_PATH)
        self._corpus_lookup = (
            self._corpus.set_index("docno").to_dict(orient="index")
        )

        self._loaded = True
        log.info("SearchService loaded — corpus=%d docs", len(self._corpus))
        return self

    @staticmethod
    def _init_java() -> None:
        """Initialize Java environment if needed."""
        pass

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(
        self,
        query: str,
        top_k: int = 10,
        mode: str = "ltr",
        alpha: float = 0.3,
    ) -> list[SearchResult]:
        """
        Run a search query and return ranked results.

        Args:
            query: Natural-language search query string.
            top_k: Maximum number of results to return.
            mode:  Retrieval mode — 'bm25' for baseline,
                   'ltr' for full re-ranked pipeline.
            alpha: BM25 weight in the LTR blend
                   (0 = pure model signal, 1 = pure BM25).

        Returns:
            Ordered list of SearchResult objects, best first.

        Raises:
            RuntimeError: If load() has not been called.
        """
        if not self._loaded:
            raise RuntimeError("Call service.load() before search().")

        query = query.strip()
        if not query:
            return []

        query_df = pd.DataFrame([{"qid": "q0", "query": query}])
        bm25_results = self._bm25.transform(query_df)

        if bm25_results.empty:
            log.warning("BM25 returned no results for query: %s", query)
            return []

        bm25_results["rank"] = (
            bm25_results.groupby("qid")["score"]
            .rank(ascending=False, method="first")
            .astype(int)
        )

        if mode == "ltr":
            import importlib
            reranker_mod = importlib.import_module("src.05_reranker")
            rerank = reranker_mod.rerank
            ranked = rerank(
                query_df, bm25_results, self._doc_features,
                self._ltr_model, self._ltr_scaler,
                alpha=alpha, num_results=top_k,
            )
        else:
            ranked = bm25_results.sort_values("rank").head(top_k)

        return self._to_results(ranked, top_k)

    def _to_results(self, ranked: pd.DataFrame, top_k: int) -> list[SearchResult]:
        """
        Convert a ranked results DataFrame into a list of SearchResult objects.

        Args:
            ranked: DataFrame with docno, score, rank columns.
            top_k:  Maximum number of results.

        Returns:
            List of SearchResult dataclasses.
        """
        results = []
        for i, (_, row) in enumerate(ranked.sort_values("rank").head(top_k).iterrows()):
            docno = str(row["docno"])
            doc   = self._corpus_lookup.get(docno, {})
            text  = str(doc.get("text", "")).strip()
            if len(text) > 350:
                text = text[:350] + "…"

            results.append(SearchResult(
                rank         = i + 1,
                docno        = docno,
                score        = float(row["score"]),
                text         = text,
                author       = str(doc.get("author",       "unknown")),
                subreddit    = str(doc.get("subreddit",    "unknown")),
                post_score   = int(doc.get("score",        0)),
                upvote_ratio = float(doc.get("upvote_ratio", 0.5)),
                num_comments = int(doc.get("num_comments", 0)),
            ))
        return results

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        status = "loaded" if self._loaded else "not loaded"
        corpus_size = len(self._corpus) if self._corpus is not None else 0
        return f"SearchService(status={status}, corpus_size={corpus_size})"


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    log.info("=== Smoke test: SearchService interface (no artifacts needed) ===")

    # Test SearchResult dataclass
    r = SearchResult(
        rank=1, docno="d1", score=0.92, text="Investing in index funds is smart.",
        author="alice", subreddit="investing", post_score=250,
        upvote_ratio=0.97, num_comments=30,
    )
    assert r.rank == 1
    assert r.score == 0.92
    assert r.author == "alice"
    log.info("SearchResult dataclass ✓  rank=%d  author=%s  score=%.2f",
             r.rank, r.author, r.score)

    # Test that SearchService raises before load()
    service = SearchService()
    try:
        service.search("test")
        assert False, "Should have raised RuntimeError"
    except RuntimeError as e:
        log.info("Pre-load guard ✓  error='%s'", e)

    # Test __repr__
    assert "not loaded" in repr(service)
    log.info("__repr__ ✓  '%s'", service)

    log.info("=== All smoke tests passed ===")
    log.info("Full pipeline usage:")
    log.info("  service = SearchService().load()")
    log.info("  results = service.search('how to invest in index funds', top_k=10)")
