"""
05_reranker.py — Learning-to-Rank reranker trained on ground truth relevance labels.

Pipeline:
    1. (Commit #15) Assemble feature vectors: BM25 score + 9 document features.
    2. (Commit #16) Train Logistic Regression on gold qrels (train split).
    3. (Commit #17) Rerank BM25 candidates using trained model probabilities.

The trained model combines retrieval signal (BM25) with social trust, author
authority, and content quality signals into a single relevance score.

Inputs:
    data/processed/corpus.pkl
    data/processed/all_features.pkl
    data/processed/qrels/train_gold.pkl
    index/ (built by 02_indexer.py)

Outputs:
    models/ltr_model.pkl   — trained LogisticRegression
    models/ltr_scaler.pkl  — fitted StandardScaler

Usage:
    python src/05_reranker.py
"""

import os
import sys
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.utils import load_pkl, save_pkl, get_logger, PROCESSED_DIR, min_max_normalize

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

ALL_FEATURES_PATH = PROCESSED_DIR / "all_features.pkl"
TRAIN_QRELS_PATH  = PROCESSED_DIR / "qrels" / "train_gold.pkl"
CORPUS_PATH       = PROCESSED_DIR / "corpus.pkl"

BASE_DIR          = Path(__file__).resolve().parent.parent
MODELS_DIR        = BASE_DIR / "models"
MODEL_PATH        = MODELS_DIR / "ltr_model.pkl"
SCALER_PATH       = MODELS_DIR / "ltr_scaler.pkl"

# ---------------------------------------------------------------------------
# LTR constants
# ---------------------------------------------------------------------------

# Weight of BM25 score vs model probability in the final reranking score
ALPHA_BM25 = 0.3          # final = α·bm25_norm + (1-α)·P(relevant)

# LogisticRegression hyperparameters
LR_C         = 1.0
LR_MAX_ITER  = 1_000
LR_SOLVER    = "lbfgs"

# All document feature columns (produced by 03_features.py)
FEATURE_COLS = [
    "F_SOC_SCORE", "F_SOC_UPVOTE", "F_SOC_ENGAGE",
    "F_GRAPH_DEG", "F_GRAPH_BET", "F_GRAPH_COMM",
    "F_CONT_SENT", "F_CONT_NER",  "F_CONT_MONEY",
]

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

log = get_logger("reranker")


# ===========================================================================
# Commit #15 — Assemble feature vectors
# ===========================================================================

def build_feature_matrix(
    qrels: pd.DataFrame,
    bm25_results: pd.DataFrame,
    doc_features: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Build the (X, y) training feature matrix for LTR.

    For each (query, doc) pair in qrels, the feature vector contains:
        - BM25 score (min-max normalized per query)
        - 9 document-level features from all_features.pkl

    Args:
        qrels:       DataFrame with columns [qid, docno, label].
        bm25_results: DataFrame with columns [qid, docno, score].
        doc_features: DataFrame indexed by docno with FEATURE_COLS columns.

    Returns:
        X:    Feature matrix of shape (N, 10).
        y:    Binary label array of shape (N,).
        meta: DataFrame with qid/docno for traceability.
    """
    log.info("Assembling feature matrix from %d qrel pairs...", len(qrels))

    # Normalize BM25 score per query (min-max within each query group)
    bm25 = bm25_results[["qid", "docno", "score"]].copy()
    bm25["bm25_norm"] = (
        bm25.groupby("qid")["score"]
        .transform(lambda s: min_max_normalize(s))
    )

    # Merge qrels ← BM25 scores ← document features
    df = qrels.merge(
        bm25[["qid", "docno", "bm25_norm"]],
        on=["qid", "docno"],
        how="left",
    )
    df = df.merge(
        doc_features[FEATURE_COLS],
        left_on="docno",
        right_index=True,
        how="left",
    )

    # Fill missing values (docs not in index or features)
    df["bm25_norm"] = df["bm25_norm"].fillna(0.0)
    for col in FEATURE_COLS:
        df[col] = df[col].fillna(0.0)

    all_cols = ["bm25_norm"] + FEATURE_COLS
    X = df[all_cols].values.astype(np.float32)
    y = df["label"].values.astype(int)

    log.info(
        "Feature matrix ready — shape=%s  positive=%d  negative=%d",
        X.shape, y.sum(), (y == 0).sum(),
    )
    return X, y, df[["qid", "docno", "label"]]


# ===========================================================================
# Commit #16 — Train LTR model
# ===========================================================================

def train_ltr_model(
    X: np.ndarray,
    y: np.ndarray,
    C: float = LR_C,
) -> tuple[LogisticRegression, StandardScaler]:
    """
    Train a Logistic Regression Learning-to-Rank model.

    Uses class_weight='balanced' to handle the 10/90 positive/negative ratio
    produced by the silver labeling threshold.

    Args:
        X: Feature matrix of shape (N, num_features).
        y: Binary label array of shape (N,).
        C: Inverse regularization strength (smaller = stronger regularization).

    Returns:
        Trained LogisticRegression model and fitted StandardScaler.
    """
    log.info("Training LTR model — samples=%d  features=%d", *X.shape)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = LogisticRegression(
        C=C,
        class_weight="balanced",
        solver=LR_SOLVER,
        max_iter=LR_MAX_ITER,
        random_state=42,
    )
    model.fit(X_scaled, y)

    # Training metrics
    y_pred = model.predict(X_scaled)
    y_prob = model.predict_proba(X_scaled)[:, 1]
    auc    = roc_auc_score(y, y_prob)

    log.info("Training complete — AUC=%.4f", auc)
    log.info(
        "\n%s",
        classification_report(y, y_pred, target_names=["irrelevant", "relevant"]),
    )

    # Log feature importances (LR coefficients)
    feature_names = ["bm25_norm"] + FEATURE_COLS
    coef_pairs = sorted(
        zip(feature_names, model.coef_[0]),
        key=lambda x: abs(x[1]),
        reverse=True,
    )
    log.info("Feature importances (|coef|):")
    for name, coef in coef_pairs:
        log.info("  %-20s  %+.4f", name, coef)

    return model, scaler


# ===========================================================================
# Commit #17 — Rerank using trained model
# ===========================================================================

def rerank(
    queries: pd.DataFrame,
    bm25_results: pd.DataFrame,
    doc_features: pd.DataFrame,
    model: LogisticRegression,
    scaler: StandardScaler,
    alpha: float = ALPHA_BM25,
    num_results: int = 1000,
) -> pd.DataFrame:
    """
    Rerank BM25 candidates using the trained LTR model.

    Final score combines BM25 and model probability:
        score = α · bm25_norm + (1 - α) · P(relevant | features)

    Args:
        queries:      DataFrame with 'qid' and 'query' columns.
        bm25_results: BM25 candidate DataFrame with [qid, docno, score].
        doc_features: Feature DataFrame indexed by docno.
        model:        Trained LogisticRegression model.
        scaler:       Fitted StandardScaler from training.
        alpha:        Weight for BM25 score in the final blend.
        num_results:  Maximum number of results to return per query.

    Returns:
        Reranked DataFrame with [qid, query, docno, score, rank] columns,
        sorted by descending score per query.
    """
    log.info("Reranking %d (query, doc) pairs...", len(bm25_results))

    df = bm25_results[["qid", "docno", "score"]].copy()

    # Normalize BM25 score per query
    df["bm25_norm"] = (
        df.groupby("qid")["score"]
        .transform(lambda s: min_max_normalize(s))
    )

    # Attach document features
    df = df.merge(
        doc_features[FEATURE_COLS],
        left_on="docno",
        right_index=True,
        how="left",
    )
    for col in FEATURE_COLS:
        df[col] = df[col].fillna(0.0)
    df["bm25_norm"] = df["bm25_norm"].fillna(0.0)

    # Build feature matrix and predict
    all_cols = ["bm25_norm"] + FEATURE_COLS
    X = df[all_cols].values.astype(np.float32)
    X_scaled = scaler.transform(X)
    p_relevant = model.predict_proba(X_scaled)[:, 1]

    # Blend scores
    df["score"] = alpha * df["bm25_norm"] + (1 - alpha) * p_relevant

    # Merge query text back
    df = df.merge(queries[["qid", "query"]], on="qid", how="left")

    # Sort, clip, and add rank
    df = df.sort_values(["qid", "score"], ascending=[True, False])
    df = df.groupby("qid").head(num_results).copy()
    df["rank"] = df.groupby("qid").cumcount() + 1

    log.info("Reranking complete — %d results total.", len(df))
    return df[["qid", "query", "docno", "score", "rank"]].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Persistence helpers
# ---------------------------------------------------------------------------

def save_model(
    model: LogisticRegression,
    scaler: StandardScaler,
    model_path: Path = MODEL_PATH,
    scaler_path: Path = SCALER_PATH,
) -> None:
    """Save trained model and scaler to disk."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    save_pkl(model,  model_path)
    save_pkl(scaler, scaler_path)
    log.info("Model saved → %s", model_path)
    log.info("Scaler saved → %s", scaler_path)


def load_model(
    model_path: Path = MODEL_PATH,
    scaler_path: Path = SCALER_PATH,
) -> tuple[LogisticRegression, StandardScaler]:
    """Load trained model and scaler from disk."""
    model  = load_pkl(model_path)
    scaler = load_pkl(scaler_path)
    log.info("Loaded model from %s", model_path)
    return model, scaler


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run() -> None:
    """Full LTR training pipeline: load data → train → save model."""
    import pyterrier as pt
    if not pt.java.started():
        pt.java.init()

    import importlib
    indexer_mod = importlib.import_module("src.02_indexer")
    load_index = indexer_mod.load_index
    get_bm25_retriever = indexer_mod.get_bm25_retriever
    corpus       = load_pkl(CORPUS_PATH)
    doc_features = load_pkl(ALL_FEATURES_PATH)
    train_qrels  = load_pkl(TRAIN_QRELS_PATH)

    index = load_index(Path("index").resolve())
    bm25  = get_bm25_retriever(index)

    # Get BM25 scores for all qids in train qrels
    train_queries = (
        train_qrels[["qid"]].drop_duplicates()
        .assign(query=lambda d: d["qid"])  # qid is already the query text
    )
    bm25_results = bm25.transform(train_queries)

    X, y, _ = build_feature_matrix(train_qrels, bm25_results, doc_features)
    model, scaler = train_ltr_model(X, y)
    save_model(model, scaler)

    log.info("=== LTR training complete ===")


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    log.info("=== Smoke test: LTR feature assembly, training, and reranking ===")

    np.random.seed(42)
    N_DOCS    = 50
    N_QUERIES = 5

    # Synthetic document features
    docnos = [f"d{i}" for i in range(N_DOCS)]
    doc_features = pd.DataFrame(
        np.random.rand(N_DOCS, len(FEATURE_COLS)),
        columns=FEATURE_COLS,
        index=pd.Index(docnos, name="docno"),
    )

    # Synthetic BM25 results (5 queries × 10 docs each)
    bm25_results = pd.concat([
        pd.DataFrame({
            "qid":   [f"q{q}"] * 10,
            "docno": [f"d{i}" for i in range(q * 10, q * 10 + 10)],
            "score": np.random.rand(10),
            "query": [f"query about finance topic {q}"] * 10,
        })
        for q in range(N_QUERIES)
    ], ignore_index=True)

    # Synthetic gold qrels (top 3 per query = relevant)
    qrels = pd.concat([
        pd.DataFrame({
            "qid":   [f"q{q}"] * 10,
            "docno": [f"d{i}" for i in range(q * 10, q * 10 + 10)],
            "label": [1] * 3 + [0] * 7,
        })
        for q in range(N_QUERIES)
    ], ignore_index=True)

    # --- Test Commit #15: feature assembly ---
    X, y, meta = build_feature_matrix(qrels, bm25_results, doc_features)
    assert X.shape == (N_QUERIES * 10, 10), f"Wrong X shape: {X.shape}"
    assert y.sum() == N_QUERIES * 3,         f"Wrong positives: {y.sum()}"
    assert not np.isnan(X).any(),            "NaNs in feature matrix"
    log.info("Feature assembly ✓  X=%s  positives=%d", X.shape, y.sum())

    # --- Test Commit #16: model training ---
    model, scaler = train_ltr_model(X, y)
    assert hasattr(model, "predict_proba"), "Model has no predict_proba"
    probs = model.predict_proba(scaler.transform(X))[:, 1]
    assert probs.min() >= 0 and probs.max() <= 1, "Probs out of [0,1]"
    log.info("Model training ✓  coef_shape=%s  AUC computed above", model.coef_.shape)

    # --- Test Commit #17: reranking ---
    queries = pd.DataFrame({
        "qid":   [f"q{q}" for q in range(N_QUERIES)],
        "query": [f"query about finance topic {q}" for q in range(N_QUERIES)],
    })
    reranked = rerank(queries, bm25_results, doc_features, model, scaler)
    assert "rank" in reranked.columns,  "Missing 'rank' column"
    assert "score" in reranked.columns, "Missing 'score' column"
    assert reranked.groupby("qid")["rank"].min().eq(1).all(), "Rank doesn't start at 1"
    # Check scores are descending within each query
    for qid, group in reranked.groupby("qid"):
        scores = group["score"].values
        assert (scores[:-1] >= scores[1:]).all(), f"Scores not descending for {qid}"
    log.info("Reranking ✓  total_results=%d  ranks_OK=True", len(reranked))

    log.info("=== All smoke tests passed ===")

    # Run full pipeline
    log.info("")
    log.info("=== Running full LTR training pipeline ===")
    run()

