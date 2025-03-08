import pandas as pd
import numpy as np
import os
import json
from rank_bm25 import BM25Okapi
from datasets import load_dataset

from tqdm.auto import tqdm
import logging
import pickle

logger = logging.getLogger(__name__)

BM25_PICKLE_PATH = "cuad/bm25_model.pkl"
BM25_DATASET_PATH = "cuad/bm25_data.json"

def load_documents(csv_file):
    """
    Load documents from a CSV file.
    Assumes that the CSV has a column named "text" containing the document content.
    """
    df = pd.read_csv(csv_file)
    documents = df['text'].tolist()
    return documents, df

def tokenize(text):
    """
    Simple tokenization: lowercases and splits on whitespace.
    """
    return text.lower().split()

def build_bm25_index(documents):
    """
    Build and return a BM25 index from the provided documents.
    """
    tokenized_docs = [tokenize(doc) for doc in tqdm(documents, desc="Tokenizing Documents")]
    bm25 = BM25Okapi(tokenized_docs)
    return bm25

def retrieve(query, bm25, documents, k=5):
    """
    Retrieve top-k documents for a given query using BM25 scores.
    Returns a list of tuples (doc_index, document_text, score).
    """
    tokenized_query = tokenize(query)
    scores = bm25.get_scores(tokenized_query)
    # Get indices of documents sorted by score (highest first)
    top_k_indices = np.argsort(scores)[-k:][::-1]
    return [(idx, documents[idx], scores[idx]) for idx in top_k_indices]

def download_cuad_dataset(retrieval_dataset):
    """
    Download the retrieval dataset from the Hugging Face Hub.
    """
    logger.info(f"Downloading retrieval dataset from Hugging Face: {retrieval_dataset}...")
    hf_dataset = load_dataset(retrieval_dataset)
    return hf_dataset["train"]["context"]

def setup_bm25_index(retrieval_dataset, dataset_dir="bm25_data"):
    os.makedirs(dataset_dir, exist_ok=True)
    dataset_path = os.path.join(dataset_dir, f"dataset.json")
    pickle_path = os.path.join(dataset_dir, f"bm25_index.pkl")

    if os.path.exists(dataset_path) and os.path.exists(pickle_path):
        logger.info("Loading precomputed BM25 index and dataset...")

        with open(dataset_path, "r") as f:
            retrieval_texts = json.load(f)
        
        with open(pickle_path, "rb") as f:
            bm25 = pickle.load(f)

        logger.info("BM25 index successfully loaded.")
    else:
        retrieval_texts = download_cuad_dataset(retrieval_dataset)

        with open(dataset_path, "w") as f:
            json.dump(retrieval_texts, f)
        logger.info("BM25 dataset saved for future use.")

        logger.info("Building BM25 index...")
        bm25 = build_bm25_index(retrieval_texts)

        with open(pickle_path, "wb") as f:
            pickle.dump(bm25, f)
        logger.info("BM25 index successfully built and saved.")

    return bm25, retrieval_texts