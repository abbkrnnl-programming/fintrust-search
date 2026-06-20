import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.utils import load_pkl, save_pkl, PROCESSED_DIR

def relabel():
    print("Loading corpus and initializing PyTerrier...")
    import pyterrier as pt
    if not pt.java.started():
        pt.java.init()
        
    corpus = load_pkl(PROCESSED_DIR / "corpus.pkl")
    # Quick lookup dictionary for text
    text_lookup = corpus.set_index('docno')['text'].to_dict()
    
    index = pt.IndexFactory.of(str(Path(__file__).parent.parent / "index"))
    bm25 = pt.BatchRetrieve(index, wmodel="BM25", num_results=1000)
    
    # Text-based query-dependent heuristic for relevance (Ground Truth Y)
    def get_label(row):
        qid = row['qid']
        docno = row['docno']
        query_text = query_lookup.get(qid, "").lower().split()
        if not query_text:
            return 0
            
        doc_text = str(text_lookup.get(docno, "")).lower()
        
        # 1. Main entity/ticker (usually the first word)
        ticker = query_text[0]
        if ticker not in doc_text:
            return 0
            
        # 2. Topic words (the rest of the query)
        topic_words = query_text[1:]
        if topic_words:
            topic_match = any(w in doc_text for w in topic_words)
            if not topic_match:
                return 0
                
        # 3. Substantive content (at least 20 words) to mimic "good" answers
        if len(doc_text.split()) < 20:
            return 0
            
        return 1

    # 1. Train Set
    print("Generating Train Gold (Query-dependent heuristic)...")
    train_queries = load_pkl(PROCESSED_DIR / "queries" / "train_queries.pkl")
    global query_lookup
    query_lookup = train_queries.set_index('qid')['query'].to_dict()
    
    # Get top 1000 BM25 candidates to label
    train_res = bm25.transform(train_queries)
    train_res['label'] = train_res.apply(get_label, axis=1)
    
    # Save train_gold.pkl
    save_pkl(train_res[['qid', 'docno', 'label']], PROCESSED_DIR / "qrels" / "train_gold.pkl")
    print(f"Train Positives: {train_res['label'].sum()} / {len(train_res)}")
    
    # 2. Test Set
    print("Generating Test Gold (Query-dependent heuristic)...")
    test_queries = load_pkl(PROCESSED_DIR / "queries" / "test_queries.pkl")
    query_lookup = test_queries.set_index('qid')['query'].to_dict()
    
    test_res = bm25.transform(test_queries)
    test_res['label'] = test_res.apply(get_label, axis=1)
    
    # Save test_gold.csv
    test_res[['qid', 'docno', 'label']].to_csv(PROCESSED_DIR / "qrels" / "test_gold.csv", index=False)
    print(f"Test Positives: {test_res['label'].sum()} / {len(test_res)}")
    print("Relabeling complete! Valid query-dependent Ground Truth generated.")

if __name__ == '__main__':
    relabel()
