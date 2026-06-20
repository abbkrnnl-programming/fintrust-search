# FinTrust Search

> **Social search engine for financial advice** — ranks Reddit posts using
> textual relevance, social trust signals, author graph authority, and content quality.

---

## Idea

We built a specialized search engine for the Reddit platform (specifically the **r/wallstreetbets** subreddit).

## The Problem with Standard Search

If you search for something on Reddit using standard text-based search (BM25), it will return hundreds of random, useless comments and posts simply because the keywords happen to match. Standard keyword matching is unable to distinguish between a high-quality analysis and a random noisy comment.

## Our Solution: Social Search

We claim that to find high-quality posts on Reddit, text alone is not enough. You must absolutely consider:
1. **WHO** wrote the post (**Graph** Analysis / Centrality).
2. **HOW** the crowd reacted to it (**Likes** / Social Metrics).
3. **WHAT** is the quality of the content (Sentiment, Entities).

**FinTrust Search** addresses this by combining classical Information Retrieval (PyTerrier BM25) with Graph Analysis, Social Metrics, and NLP into a single Learning-to-Rank (LTR) pipeline using Logistic Regression.

---

## Core Hypothesis

> A re-ranking model that incorporates social trust signals (likes/engagement), author graph centrality (who wrote it), and content quality features **statistically significantly outperforms** a text-only BM25 baseline on financial question-answering tasks.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        DATA INGESTION                           │
│  WSB Queries  ──► 200 financial queries (train/dev/test splits) │
│  kagglehub    ──► Automatically downloads WSB dataset           │
│  praw API     ──► Live Reddit sample (optional)                 │
│             ↓  01_data_loader.py                                │
│         corpus.pkl  +  comments.pkl  +  query splits           │
└────────────────────────────┬────────────────────────────────────┘
                             │
           ┌─────────────────┼──────────────────┐
           ↓                 ↓                  ↓
    02_indexer.py     03_features.py      03_features.py
    PyTerrier BM25    Social Trust        Author Graph
    inverted index    (score, upvote,     (in-degree,
                      comments)           betweenness,
                                          Louvain hub)
                                               ↓
                                       03_features.py
                                       Content Quality
                                       (VADER, NER, $)
                                               │
           ┌───────────────────────────────────┘
           ↓
    04_ground_truth.py
    Text Heuristic Teacher → query-dependent labels → train_gold.pkl
           ↓
    05_reranker.py
    LogisticRegression LTR on train_gold.pkl → ltr_model.pkl
           ↓
    06_evaluate.py
    pt.Experiment on 50 Test queries
    [BM25 | BM25+RM3 | BM25+Social | BM25+Graph | BM25+Full]
```

---

## Project Structure

```text
fintrust-search/
├── src/
│   ├── utils.py                  # Shared normalization + I/O helpers
│   ├── 00_reddit_scraper.py      # Reddit API collection (praw)
│   ├── 01_data_loader.py         # Auto-download via kagglehub + merge
│   ├── 02_indexer.py             # Build PyTerrier inverted index
│   ├── 03_features.py            # Social, Graph, and Content features
│   ├── 04_ground_truth.py        # Text heuristic Ground Truth labels
│   ├── 05_reranker.py            # FeatureReRanker + LTR training
│   ├── 06_evaluate.py            # pt.Experiment on test set
│   └── search_service.py         # Facade class for UI integration
├── notebooks/
│   ├── 01_EDA_and_Graph.ipynb    # Corpus stats + graph visualization
│   └── 02_Evaluation_Results.ipynb  # Results, ablation, case studies
├── data/                         # Raw and processed datasets (gitignored)
├── index/                        # PyTerrier inverted index (gitignored)
├── models/                       # Trained LTR model + scaler (gitignored)
├── results/                      # Evaluation CSVs + figures
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Feature Specification

| Feature ID | Dimension | Description |
|---|---|---|
| `F_TEXT_BM25` | Textual | BM25 score (normalized per query) |
| `F_SOC_SCORE` | Social Trust | `log1p(score)` → Min-Max |
| `F_SOC_UPVOTE` | Social Trust | Upvote ratio ∈ [0, 1] |
| `F_SOC_ENGAGE` | Social Trust | `log1p(num_comments)` → Min-Max |
| `F_GRAPH_DEG` | Author Authority | In-degree centrality (NetworkX) |
| `F_GRAPH_BET` | Author Authority | Betweenness centrality (approx.) |
| `F_GRAPH_COMM` | Author Authority | Binary: bridges ≥ 2 Louvain communities |
| `F_CONT_SENT` | Content Quality | Sentiment neutrality: `1 - |VADER compound|` |
| `F_CONT_NER` | Content Quality | Financial entity density (spaCy) |
| `F_CONT_MONEY` | Content Quality | Binary: regex money mention (`$`, `USD`, `BTC` …) |

---

## Setup

### Prerequisites

- **Python** 3.10+
- **Java 11+** — required by PyTerrier (`java -version` to verify)

### Installation

```bash
# 1. Clone the repository
git clone <repo-url>
cd fintrust-search

# 2. Create and activate a virtual environment
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS / Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download spaCy language model
python -m spacy download en_core_web_sm
```

### Data Download
**You do not need to download Kaggle CSVs manually.** 
The script `src/01_data_loader.py` will automatically download the dataset `mattpodolak/rwallstreetbets-posts-and-comments` via the `kagglehub` library on its first run.

---

## Pipeline Execution

Run scripts **in order**:

```bash
python src/01_data_loader.py       # auto-download data, build corpus + query splits
python src/02_indexer.py           # build PyTerrier BM25 index
python src/03_features.py          # compute all 9 document features
python src/04_ground_truth.py      # generate gold labels via text heuristic
python src/05_reranker.py          # train Logistic Regression LTR model
```

Then for evaluation:
```bash
# Step 1 — run full pt.Experiment
python src/06_evaluate.py experiment

# Step 2 — ablation study
python src/06_evaluate.py ablation
```

---

## Evaluation Results

*Results based on the `mattpodolak/rwallstreetbets-posts-and-comments` Kaggle dataset (510k posts).*

| System | NDCG@10 | MAP@1000 | P@5 | MRR@10 |
|---|---|---|---|---|
| BM25 (baseline) | 0.0719 | 0.0708 | 0.056 | 0.1787 |
| BM25 + RM3 | 0.0641 | 0.0634 | 0.048 | 0.1690 |
| BM25 + Social | 0.0750 | 0.0788 | 0.072 | 0.1836 |
| BM25 + Graph | 0.0704 | 0.0694 | 0.056 | 0.1792 |
| BM25 + Content | 0.1799 | 0.1536 | 0.148 | - |
| **BM25 + Full LTR** | **0.2270** | **0.1767** | **0.212** | **0.4085** |

*Note: Absolute metrics appear low because manual annotation was generated via heuristics for a subset of queries during testing. However, the relative improvements hold true, confirming that adding social, graph, and content features improves upon the text-only BM25 baseline.*

---

## Tech Stack

| Component | Library |
|---|---|
| Inverted index + BM25 | [PyTerrier](https://github.com/terrier-org/pyterrier) |
| Graph analysis | [NetworkX](https://networkx.org/) |
| Sentiment analysis | [VADER](https://github.com/cjhutto/vaderSentiment) |
| Named entity recognition | [spaCy](https://spacy.io/) `en_core_web_sm` |
| LTR model (Student) | [scikit-learn](https://scikit-learn.org/) `LogisticRegression` |
| Data Loading | [kagglehub](https://github.com/Kaggle/kagglehub) |
| Reddit data collection | [PRAW](https://praw.readthedocs.io/) |

---

## License

This project was developed as a university course assignment.
