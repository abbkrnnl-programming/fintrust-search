"""
03_features.py — Compute all relevance features for the FinTrust Search corpus.

Dimensions:
    1. Social Trust  — Reddit engagement signals (score, upvote ratio, comments).
    2. Author Graph  — Centrality metrics from the author interaction network.
    3. Content Quality — Sentiment neutrality, NER density, money mentions.

All features are stored per-document (indexed by 'docno') and normalized
to [0, 1] via Min-Max scaling.

Inputs:
    data/processed/corpus.pkl
    data/processed/comments.pkl

Outputs:
    data/processed/social_features.pkl
    data/processed/graph_features.pkl
    data/processed/content_features.pkl
    data/processed/all_features.pkl
    data/processed/author_graph.graphml

Usage:
    python src/03_features.py
"""

import re
import sys
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import networkx as nx
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from utils import (
    load_pkl, save_pkl, get_logger,
    min_max_normalize, log1p_normalize,
    PROCESSED_DIR,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CORPUS_PATH   = PROCESSED_DIR / "corpus.pkl"
COMMENTS_PATH = PROCESSED_DIR / "comments.pkl"

SOCIAL_FEATURES_PATH  = PROCESSED_DIR / "social_features.pkl"
GRAPH_FEATURES_PATH   = PROCESSED_DIR / "graph_features.pkl"
CONTENT_FEATURES_PATH = PROCESSED_DIR / "content_features.pkl"
ALL_FEATURES_PATH     = PROCESSED_DIR / "all_features.pkl"
GRAPH_PATH            = PROCESSED_DIR / "author_graph.graphml"

# Regex: detect monetary values ($100, $10k, 500$, 500 USD, 0.5 BTC, etc.)
_MONEY_RE = re.compile(
    r"\$\s*[\d,]+(?:\.\d+)?(?:[kKmMbB])?"                                # $500, $1k, $ 1.5M
    r"|[\d,]+(?:\.\d+)?\s*\$"                                            # 500$, 500 $
    r"|\b(?:USD|EUR|GBP|CAD|AUD|CHF|JPY|BTC|ETH|DOGE|SOL)\s*[\d,]+(?:\.\d+)?"  # USD 500
    r"|\b[\d,]+(?:\.\d+)?\s*(?:[kKmMbB]\s*)?(?:USD|EUR|GBP|CAD|AUD|CHF|JPY|BTC|ETH|DOGE|SOL)\b" # 500 USD, 10k USD
)

# spaCy entity labels considered "financial"
_FINANCIAL_ENT_LABELS = {"ORG", "MONEY", "PRODUCT", "GPE", "PERCENT", "PERSON"}

# Minimum node degree to include in betweenness calculation (performance)
_MIN_DEGREE_FOR_BETWEENNESS = 2

# Approximate betweenness: sample k nodes (set None to compute exactly)
_BETWEENNESS_K: int | None = 500

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

log = get_logger("features")


# ===========================================================================
# Part 1 — Social Trust Features
# ===========================================================================

def compute_social_features(corpus: pd.DataFrame) -> pd.DataFrame:
    """
    Compute three social trust signals from Reddit post metadata.

    Features:
        F_SOC_SCORE   — log1p(score) normalized to [0, 1].
        F_SOC_UPVOTE  — upvote_ratio (already in [0, 1]).
        F_SOC_ENGAGE  — log1p(num_comments) normalized to [0, 1].

    Args:
        corpus: DataFrame with columns 'docno', 'score',
                'upvote_ratio', 'num_comments'.

    Returns:
        DataFrame indexed by 'docno' with the three feature columns.
    """
    log.info("Computing social trust features for %d documents...", len(corpus))

    df = corpus[["docno"]].copy()

    score = corpus.get("score", pd.Series(0, index=corpus.index)).fillna(0)
    df["F_SOC_SCORE"] = log1p_normalize(score).values

    upvote = corpus.get("upvote_ratio", pd.Series(0.5, index=corpus.index)).fillna(0.5)
    df["F_SOC_UPVOTE"] = upvote.clip(0.0, 1.0).values

    comments = corpus.get("num_comments", pd.Series(0, index=corpus.index)).fillna(0)
    df["F_SOC_ENGAGE"] = log1p_normalize(comments).values

    df = df.set_index("docno")
    log.info("Social features computed. Shape: %s", df.shape)
    return df


# ===========================================================================
# Part 2 — Author Graph Features
# ===========================================================================

def build_author_graph(comments: pd.DataFrame) -> nx.DiGraph:
    """
    Build a directed author interaction graph from comment reply pairs.

    An edge (comment_author → post_author) represents a reply event.
    Self-loops and isolates are removed.

    Args:
        comments: DataFrame with 'comment_author' and 'post_author' columns.

    Returns:
        Directed NetworkX graph representing author interactions.
    """
    log.info("Building author interaction graph from %d comment pairs...", len(comments))

    G = nx.DiGraph()

    for _, row in tqdm(comments.iterrows(), total=len(comments),
                       desc="Building graph", unit="edges"):
        src = str(row.get("comment_author", ""))
        dst = str(row.get("post_author", ""))

        # Skip deleted/unknown authors and self-loops
        if not src or not dst or src == dst:
            continue
        if src in ("[deleted]", "AutoModerator") or dst in ("[deleted]", "AutoModerator"):
            continue

        if G.has_edge(src, dst):
            G[src][dst]["weight"] += 1
        else:
            G.add_edge(src, dst, weight=1)

    log.info("Raw graph: %d nodes, %d edges", G.number_of_nodes(), G.number_of_edges())

    # Keep only the Giant Weakly Connected Component
    if G.number_of_nodes() > 0:
        gcc_nodes = max(nx.weakly_connected_components(G), key=len)
        G = G.subgraph(gcc_nodes).copy()
        log.info("Giant component: %d nodes, %d edges",
                 G.number_of_nodes(), G.number_of_edges())

    return G


def _detect_community_hubs(G: nx.DiGraph) -> dict[str, float]:
    """
    Detect community hub authors using Louvain community detection.

    An author is a hub (score=1) if they have edges to nodes in at least
    2 distinct Louvain communities. Requires nx >= 3.0 with louvain support.

    Args:
        G: Directed author interaction graph.

    Returns:
        Dict mapping author → hub_score (0.0 or 1.0).
    """
    G_undirected = G.to_undirected()
    try:
        communities = nx.community.louvain_communities(G_undirected, seed=42)
    except AttributeError:
        # Fallback: use greedy modularity if louvain is unavailable
        communities = list(nx.community.greedy_modularity_communities(G_undirected))

    node_to_community: dict[str, int] = {}
    for comm_id, community in enumerate(communities):
        for node in community:
            node_to_community[node] = comm_id

    hub_scores: dict[str, float] = {}
    for node in G.nodes():
        neighbor_communities = {
            node_to_community[nbr]
            for nbr in nx.all_neighbors(G, node)
            if nbr in node_to_community
        }
        hub_scores[node] = 1.0 if len(neighbor_communities) >= 2 else 0.0

    return hub_scores


def compute_graph_features(
    corpus: pd.DataFrame,
    G: nx.DiGraph,
) -> pd.DataFrame:
    """
    Compute author authority features from the interaction graph.

    Features:
        F_GRAPH_DEG   — Author in-degree centrality (normalized by nx).
        F_GRAPH_BET   — Author betweenness centrality (approximate if large).
        F_GRAPH_COMM  — Community hub score (binary: spans ≥ 2 communities).

    Missing authors (not in graph) are imputed with the median of each feature.

    Args:
        corpus: DataFrame with 'docno' and 'author' columns.
        G:      Directed author interaction graph from build_author_graph().

    Returns:
        DataFrame indexed by 'docno' with three feature columns.
    """
    log.info("Computing graph centrality features...")

    in_degree = nx.in_degree_centrality(G)

    k = _BETWEENNESS_K if G.number_of_nodes() > (_BETWEENNESS_K or 0) else None
    betweenness = nx.betweenness_centrality(G, k=k, normalized=True, seed=42)
    log.info("Betweenness centrality computed (k=%s).", k)

    hub_scores = _detect_community_hubs(G)

    # Map author → document
    df = corpus[["docno", "author"]].copy()
    df["F_GRAPH_DEG"]  = df["author"].map(in_degree)
    df["F_GRAPH_BET"]  = df["author"].map(betweenness)
    df["F_GRAPH_COMM"] = df["author"].map(hub_scores)

    # Impute missing values with median
    for col in ["F_GRAPH_DEG", "F_GRAPH_BET", "F_GRAPH_COMM"]:
        median_val = df[col].median()
        df[col] = df[col].fillna(median_val if not np.isnan(median_val) else 0.0)
        df[col] = min_max_normalize(df[col])

    df = df.set_index("docno")[["F_GRAPH_DEG", "F_GRAPH_BET", "F_GRAPH_COMM"]]
    log.info("Graph features computed. Shape: %s", df.shape)
    return df


# ===========================================================================
# Part 3 — Content Quality Features
# ===========================================================================

def _load_spacy():
    """Load spaCy model (lazy, once per session)."""
    import spacy
    try:
        return spacy.load("en_core_web_sm", disable=["parser", "ner"])
    except OSError:
        log.warning("spaCy model not found. Run: python -m spacy download en_core_web_sm")
        raise


def compute_content_features(
    corpus: pd.DataFrame,
    batch_size: int = 512,
) -> pd.DataFrame:
    """
    Compute NLP-based content quality features.

    Features:
        F_CONT_SENT  — Sentiment neutrality: 1 - |VADER compound|.
                       Neutral, analytical posts score highest.
        F_CONT_NER   — Financial entity density:
                       count(ORG/MONEY/PRODUCT/GPE entities) / token count.
        F_CONT_MONEY — Binary: 1 if text contains a money amount pattern.

    Args:
        corpus:     DataFrame with 'docno' and 'text' columns.
        batch_size: Batch size for spaCy NLP processing.

    Returns:
        DataFrame indexed by 'docno' with three feature columns.
    """
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    import spacy

    log.info("Computing content quality features for %d documents...", len(corpus))

    analyzer = SentimentIntensityAnalyzer()

    try:
        nlp = spacy.load("en_core_web_sm", disable=["parser"])
    except OSError:
        log.warning("spaCy 'en_core_web_sm' not found. Falling back to no-NER mode.")
        nlp = None

    texts  = corpus["text"].fillna("").tolist()
    docnos = corpus["docno"].tolist()

    sent_scores  = []
    ner_scores   = []
    money_scores = []

    # --- VADER sentiment (fast, row-by-row) ---
    for text in tqdm(texts, desc="VADER sentiment", unit="docs"):
        compound = analyzer.polarity_scores(str(text))["compound"]
        sent_scores.append(1.0 - abs(compound))

    # --- spaCy NER + money regex (batched) ---
    if nlp is not None:
        for doc in tqdm(
            nlp.pipe(texts, batch_size=batch_size),
            total=len(texts),
            desc="spaCy NER",
            unit="docs",
        ):
            token_count = max(len(doc), 1)
            fin_ents = sum(
                1 for ent in doc.ents if ent.label_ in _FINANCIAL_ENT_LABELS
            )
            ner_scores.append(fin_ents / token_count)
    else:
        ner_scores = [0.0] * len(texts)

    for text in texts:
        money_scores.append(1.0 if _MONEY_RE.search(str(text)) else 0.0)

    df = pd.DataFrame({
        "docno":        docnos,
        "F_CONT_SENT":  sent_scores,
        "F_CONT_NER":   ner_scores,
        "F_CONT_MONEY": money_scores,
    })

    # Normalize continuous features
    df["F_CONT_SENT"] = min_max_normalize(df["F_CONT_SENT"])
    df["F_CONT_NER"]  = min_max_normalize(df["F_CONT_NER"])
    # F_CONT_MONEY is already binary [0, 1]

    df = df.set_index("docno")
    log.info("Content features computed. Shape: %s", df.shape)
    return df


# ===========================================================================
# Part 4 — Merge all features (Commit #13)
# ===========================================================================

def merge_all_features(
    social: pd.DataFrame,
    graph: pd.DataFrame,
    content: pd.DataFrame,
) -> pd.DataFrame:
    """
    Merge social, graph, and content feature DataFrames on 'docno' index.

    All three DataFrames must be indexed by 'docno'. Missing values after
    merge (due to alignment issues) are filled with 0.

    Args:
        social:  DataFrame from compute_social_features().
        graph:   DataFrame from compute_graph_features().
        content: DataFrame from compute_content_features().

    Returns:
        Unified feature matrix indexed by 'docno' with 9 feature columns.
    """
    log.info("Merging all feature matrices...")
    merged = social.join(graph, how="left").join(content, how="left")
    merged = merged.fillna(0.0)

    expected_cols = [
        "F_SOC_SCORE", "F_SOC_UPVOTE", "F_SOC_ENGAGE",
        "F_GRAPH_DEG", "F_GRAPH_BET", "F_GRAPH_COMM",
        "F_CONT_SENT", "F_CONT_NER", "F_CONT_MONEY",
    ]
    missing_cols = [c for c in expected_cols if c not in merged.columns]
    if missing_cols:
        log.warning("Missing feature columns: %s — filling with 0.", missing_cols)
        for col in missing_cols:
            merged[col] = 0.0

    merged = merged[expected_cols]
    log.info("Merged feature matrix. Shape: %s  NaN count: %d",
             merged.shape, merged.isna().sum().sum())
    return merged


# ===========================================================================
# Main pipeline
# ===========================================================================

def run() -> None:
    """Execute the full feature engineering pipeline."""
    corpus   = load_pkl(CORPUS_PATH)
    comments = load_pkl(COMMENTS_PATH)

    # --- Social ---
    social = compute_social_features(corpus)
    save_pkl(social, SOCIAL_FEATURES_PATH)

    # --- Graph ---
    G = build_author_graph(comments)
    nx.write_graphml(G, GRAPH_PATH)
    log.info("Graph saved → %s", GRAPH_PATH)

    graph = compute_graph_features(corpus, G)
    save_pkl(graph, GRAPH_FEATURES_PATH)

    # --- Content ---
    content = compute_content_features(corpus)
    save_pkl(content, CONTENT_FEATURES_PATH)

    # --- Merge ---
    all_features = merge_all_features(social, graph, content)
    save_pkl(all_features, ALL_FEATURES_PATH)

    log.info("=== Feature engineering complete ===")
    log.info("  Social:  %s → %s", social.shape,  SOCIAL_FEATURES_PATH)
    log.info("  Graph:   %s → %s", graph.shape,   GRAPH_FEATURES_PATH)
    log.info("  Content: %s → %s", content.shape, CONTENT_FEATURES_PATH)
    log.info("  All:     %s → %s", all_features.shape, ALL_FEATURES_PATH)


# ===========================================================================
# Smoke test
# ===========================================================================

if __name__ == "__main__":
    log.info("=== Smoke test: feature engineering on synthetic data ===")

    # --- Synthetic corpus (5 posts) ---
    corpus = pd.DataFrame({
        "docno":        ["d1", "d2", "d3", "d4", "d5"],
        "author":       ["alice", "bob", "alice", "carol", "dave"],
        "text": [
            "Investing $500 monthly in SPY ETF is a solid strategy for retirement.",
            "YOLO on GME options. To the moon! This will definitely make me rich.",
            "Dollar cost averaging into index funds reduces volatility risk over time.",
            "I recommend consulting a financial advisor before making major decisions.",
            "Buy low sell high is obvious but timing the market is impossible.",
        ],
        "score":        [250, 10, 180, 90, 45],
        "upvote_ratio": [0.97, 0.61, 0.94, 0.89, 0.72],
        "num_comments": [30, 5, 20, 12, 8],
    })

    # --- Synthetic comments (reply pairs for graph) ---
    comments = pd.DataFrame({
        "post_id":        ["d1", "d1", "d3", "d4", "d2"],
        "post_author":    ["alice", "alice", "alice", "carol", "bob"],
        "comment_author": ["bob",   "carol", "dave",  "alice", "dave"],
        "comment_score":  [15, 8, 5, 20, 2],
        "created_utc":    [1700000000] * 5,
    })

    # --- Test social features ---
    social = compute_social_features(corpus)
    assert social.shape == (5, 3), f"Social shape wrong: {social.shape}"
    assert (social >= 0).all().all() and (social <= 1).all().all(), "Social out of [0,1]"
    log.info("Social features ✓  shape=%s", social.shape)

    # --- Test graph ---
    G = build_author_graph(comments)
    assert G.number_of_nodes() > 0, "Graph is empty"
    log.info("Author graph ✓  nodes=%d  edges=%d",
             G.number_of_nodes(), G.number_of_edges())

    graph = compute_graph_features(corpus, G)
    assert graph.shape == (5, 3), f"Graph shape wrong: {graph.shape}"
    assert (graph >= 0).all().all() and (graph <= 1).all().all(), "Graph out of [0,1]"
    log.info("Graph features ✓  shape=%s", graph.shape)

    # --- Test content features ---
    content = compute_content_features(corpus, batch_size=5)
    assert content.shape == (5, 3), f"Content shape wrong: {content.shape}"
    # Check binary MONEY feature
    assert content.loc["d1", "F_CONT_MONEY"] == 1.0, "Money mention not detected in d1"
    log.info("Content features ✓  shape=%s", content.shape)
    log.info("  Sample — d1: SENT=%.3f  NER=%.3f  MONEY=%.1f",
             content.loc["d1", "F_CONT_SENT"],
             content.loc["d1", "F_CONT_NER"],
             content.loc["d1", "F_CONT_MONEY"])

    # --- Test merge ---
    all_feats = merge_all_features(social, graph, content)
    assert all_feats.shape == (5, 9), f"Merged shape wrong: {all_feats.shape}"
    assert all_feats.isna().sum().sum() == 0, "NaNs in merged features"
    log.info("Merged features ✓  shape=%s  cols=%s",
             all_feats.shape, all_feats.columns.tolist())

    log.info("=== All smoke tests passed ===")

    # Run full pipeline
    log.info("")
    log.info("=== Running full feature engineering pipeline ===")
    run()

