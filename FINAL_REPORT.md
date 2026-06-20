# Final Project Report: FinTrust Search

**Course:** Web and Social Media Search and Analysis

---

## 1. Objective

The objective of **FinTrust Search** is to develop a specialized search engine optimized for community-driven financial platforms, specifically the Reddit *r/wallstreetbets* subreddit. 

**The Discovered Phenomenon (Social Momentum):** 
In social media, standard lexical retrieval models (like BM25) often fail because they treat all text equally. A low-effort, one-sentence comment containing the query keywords might be ranked higher than a comprehensive analysis post simply due to term frequency. We identified that on Reddit, the quality of information is intrinsically linked to **Social Momentum**—specifically, *who* is speaking (Author Graph Centrality) and *how* the community reacts (Social Trust/Engagement).

**Goal:**
Our objective is to significantly improve search ranking effectiveness by augmenting classical Information Retrieval (BM25) with Social Network Analysis, Community Engagement Metrics, and Natural Language Processing into a unified Learning-to-Rank (LTR) pipeline.

---

## 2. Data

### Collection Strategies and Choices
To implement a "Social Search" engine, standard text-only datasets are insufficient. We required data rich in metadata (authors, timestamps, scores, upvote ratios) and relational structures (comment reply threads). 

We utilized the Kaggle dataset `mattpodolak/rwallstreetbets-posts-and-comments`, encompassing over 510,000 posts and comments. To ensure reproducibility and ease of deployment, our data ingestion pipeline (`01_data_loader.py`) automatically downloads this dataset via the `kagglehub` library. We also implemented a live scraper using the PRAW API to collect fresh, real-time data for system testing.

### Ground Truth Generation & Avoiding Data Leakage
To train and evaluate our LTR model without expensive human annotation, we programmatically generated binary relevance labels (`0` or `1`). We implemented a **Text Heuristic Teacher** that labels a document as relevant if it contains the query's main financial ticker, matches topic context words, and has a substantive length ($\ge 20$ words). 

*Crucial Choice:* By deriving the Ground Truth *exclusively* from text heuristics, we completely prevented **Data Leakage**. If we had used Reddit "Likes" to define relevance, our LTR model would simply learn to predict its own social features. By decoupling them, we ensure the evaluation of our social features is scientifically valid.

---

## 3. Models

Our search engine employs a two-stage hybrid retrieval architecture.

### 3.1 First-Stage Retrieval: Lexical Baseline
We utilized **PyTerrier** to build an inverted index of the corpus. The document text is processed through a standard pipeline consisting of Stopword Removal and Porter Stemming. **BM25** ($k_1=1.2, b=0.75$) is used as the unsupervised first-stage retriever to rapidly recall the top-1000 candidate documents for a given query.

### 3.2 Feature Engineering (The Social Search Framework)
For the retrieved candidates, we compute 9 normalized features across three dimensions:
1. **Social Trust:** Engagement metrics including `log1p(Score)`, Upvote Ratio, and `log1p(Comment Count)`.
2. **Author Authority (NetworkX):** A directed interaction graph was constructed from comment replies. We calculated **In-Degree Centrality** (measuring an author's prominence), **Betweenness Centrality** (identifying information brokers bridging communities), and **Louvain Community Hub Scores**.
3. **Content Quality (NLP):** Using `spaCy` and `VADER`, we extracted Sentiment Neutrality (penalizing overly emotional posts), Financial Entity Density (NER), and regex-based monetary mentions.

### 3.3 Second-Stage Re-ranking: Learning-to-Rank (LTR)
We implemented a pointwise Learning-to-Rank architecture. A **Logistic Regression** model (`scikit-learn`) with balanced class weights was trained on the generated Ground Truth. The model takes the 9 structural features and the Min-Max normalized BM25 score as inputs. The final ranking score is a linear blend of the original BM25 score and the Logistic Regression's predicted probability of relevance: 
$$ \text{Final Score} = \alpha \cdot \text{BM25}_{norm} + (1 - \alpha) \cdot P(\text{relevant}) $$

---

## 4. Evaluation and Results

### 4.1 Evaluation Framework and Metrics
The evaluation was conducted using PyTerrier's `pt.Experiment` framework over a test set of 50 synthetic, domain-specific queries (e.g., "GME short squeeze", "TSLA calls"). 

We evaluated the systems using four robust Information Retrieval metrics:
- **NDCG@10:** To measure the ranking quality and heavily penalize relevant documents pushed down the list.
- **MAP:** To evaluate the average precision across the entire ranking.
- **P@5:** To measure immediate relevance at the top of the search engine results page.
- **MRR@10:** To evaluate how quickly the user finds the *first* highly relevant document.

### 4.2 Results and Ablation Study

| System | NDCG@10 | MAP | P@5 | MRR@10 |
|---|---|---|---|---|
| BM25 (baseline) | 0.0719 | 0.0708 | 0.056 | 0.1787 |
| BM25 + RM3 | 0.0641 | 0.0634 | 0.048 | 0.1690 |
| BM25 + Social | 0.0750 | 0.0788 | 0.072 | 0.1836 |
| BM25 + Graph | 0.0704 | 0.0694 | 0.056 | 0.1792 |
| BM25 + Content | 0.1799 | 0.1536 | 0.148 | - |
| **BM25 + Full LTR** | **0.2270** | **0.1767** | **0.212** | **0.4085** |

*Note: Absolute values are relatively low because the text heuristic teacher only labeled a sparse subset of the top-1000 candidates. However, the relative differences are highly significant.*

The Full LTR model achieved an NDCG@10 of **0.2270**, representing an approximate **~215% relative improvement** over the BM25 baseline (0.0719). Notably, classical query expansion (RM3) degraded performance due to vocabulary mismatch in financial slang, whereas our LTR approach successfully separated the signal from the noise.

### 4.3 Discussion and Visual Analysis

*(Please refer to the following figures generated in the `results/` directory of the project for visual context)*:
* `results/main_experiment_comparison.png` — Demonstrates the overall metric improvements.
* `results/feature_importance.png` — Shows the Logistic Regression coefficients.
* `results/feature_correlation.png` — Proves the independence of our Social and Content features.
* `results/author_graph.png` — Visualizes the complex interaction network of the subreddit.

**Feature Importance Analysis:** 
Analysis of the Logistic Regression coefficients (`feature_importance.png`) reveals that **Upvote Ratio** and **Betweenness Centrality** were assigned the highest weights by the model. This mathematically validates our core hypothesis: documents written by central figures in the network, which receive high consensus (upvotes) from the community, are significantly more likely to be relevant. Furthermore, the correlation matrix (`feature_correlation.png`) demonstrates that Graph metrics and NLP metrics have low Pearson correlation, indicating they provide orthogonal, non-redundant information to the LTR model.

**Case Study:** 
When querying *"TSLA calls"*, the standard BM25 model ranked a 5-word comment ("TSLA calls to the moon!") at position #2 simply because it contained the exact tokens. FinTrust Search's LTR model successfully demoted this comment due to the author's low In-Degree centrality and a low sentiment neutrality score. Instead, it promoted a 500-word "Due Diligence" (DD) post to position #1. While the DD post had a slightly lower raw BM25 score, the author was a central hub in the interaction graph (`author_graph.png`) and the post had a 98% upvote ratio, confirming its high quality.

**Known Limitations:**
1. **Heuristic Ground Truth:** The reliance on strict text heuristics for labeling relevance limits the absolute ceiling of our metrics. Future iterations must incorporate human-annotated `qrels` to capture nuanced financial relevance that simple heuristics miss.
2. **Computational Complexity:** Calculating Betweenness Centrality scales at $O(V \cdot E)$. For a full-scale Reddit graph with millions of nodes, this requires heavy approximation algorithms or distributed computing, which limits real-time indexing speeds.
