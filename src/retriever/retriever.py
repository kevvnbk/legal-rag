import pandas as pd
import numpy as np
import os
import json
from tqdm.auto import tqdm
import logging
import pickle

from rank_bm25 import BM25Okapi
from datasets import load_dataset
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

logger = logging.getLogger('legal-rag')

def build_documents(texts, chunk_size=2000, chunk_overlap=200):
    """
    Split long documents into smaller chunks.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["Page -", "\n\n", "\n", " ", ""],
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    documents = []
    for text in tqdm(texts, desc="Chunking Documents"):
        chunks = text_splitter.split_text(text)
        documents.extend([Document(page_content=chunk) for chunk in chunks])
    return documents

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

def setup_bm25_index(retrieval_dataset, dataset_dir="bm25_data", dataset_id="cuad", chunk_size=1000, chunk_overlap=200):
    os.makedirs(dataset_dir, exist_ok=True)
    dataset_path = os.path.join(dataset_dir, "chunked_dataset.json")
    pickle_path = os.path.join(dataset_dir, "bm25_index.pkl")

    if os.path.exists(dataset_path) and os.path.exists(pickle_path):
        logger.info("Loading precomputed BM25 index and chunked dataset...")

        with open(dataset_path, "r") as f:
            chunked_texts = json.load(f)
        
        with open(pickle_path, "rb") as f:
            bm25 = pickle.load(f)

        logger.info("BM25 index successfully loaded.")
    else:
        raw_texts = download_cuad_dataset(retrieval_dataset)
        document_objects = build_documents(raw_texts, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        chunked_texts = [doc.page_content for doc in document_objects]

        with open(dataset_path, "w") as f:
            json.dump(chunked_texts, f)
        logger.info("Chunked dataset saved for future use.")

        logger.info("Building BM25 index...")
        bm25 = build_bm25_index(chunked_texts)

        with open(pickle_path, "wb") as f:
            pickle.dump(bm25, f)
        logger.info("BM25 index successfully built and saved.")

    return bm25, chunked_texts