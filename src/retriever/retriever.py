import pandas as pd
import numpy as np
import os
import json
import re
import textwrap
from datasets import load_dataset

from tqdm.auto import tqdm
import logging
import pickle
import faiss
from sentence_transformers import SentenceTransformer
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

logger = logging.getLogger('legal-rag')

def load_chunked_json_files(file_paths):
    """
    Load pre-chunked JSON files into Document objects with metadata.
    """
    documents = []
    for file_path in file_paths:
        with open(file_path, "r") as f:
            lines = [line for line in f if line.strip()]
            records = [
                json.loads(line)
                for line in tqdm(
                    lines,
                    desc=f"Parsing {os.path.basename(file_path)}",
                    leave=False
                )
            ]
        for rec in records:
            documents.append(Document(
                page_content=rec["contents"],
                metadata={
                    "url": rec.get("url"),
                    "created_timestamp": rec.get("created_timestamp"),
                    "downloaded_timestamp": rec.get("downloaded_timestamp"),
                    "id": rec.get("id"),
                }
            ))
    return documents


def build_documents(texts, chunk_size, chunk_overlap):
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

def build_faiss_index(documents, model_name="all-MiniLM-L6-v2", nlist=100, batch_size=128):
    """
    Build and return a FAISS approximate index using IndexIVFFlat.
    Documents should be a list of Document objects
    """
    model = SentenceTransformer(model_name)

    texts = [doc.page_content for doc in documents]
    
    # Compute embeddings in batches
    all_embeddings = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Encoding documents"):
        batch_texts = texts[i:i+batch_size]
        batch_embeddings = model.encode(batch_texts, convert_to_numpy=True)
        all_embeddings.append(batch_embeddings)
    embeddings = np.vstack(all_embeddings)
    
    d = embeddings.shape[1]  # dimensionality of embeddings
    quantizer = faiss.IndexFlatL2(d)  # base index for clustering
    index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_L2)
    
    # Train the index on the full set of embeddings
    index.train(embeddings)
    
    # Add embeddings in batches and show progress
    for i in tqdm(range(0, embeddings.shape[0], batch_size), desc="Adding embeddings to index"):
        index.add(embeddings[i:i+batch_size])

    return index, model

def retrieve(query, faiss_index, documents, model, top_k):
    query_embedding = model.encode([query], convert_to_numpy=True)
    distances, indices = faiss_index.search(query_embedding, top_k)
    results = []
    for i, idx in enumerate(indices[0]):
        doc = documents[idx]
        results.append((doc, float(distances[0][i])))
    return results

def download_cuad_dataset(retrieval_dataset):
    """
    Download the retrieval dataset from the Hugging Face Hub.
    """
    logger.info(f"Downloading retrieval dataset from Hugging Face: {retrieval_dataset}...")
    hf_dataset = load_dataset(retrieval_dataset)
    return hf_dataset["train"]["context"]

def setup_faiss_index(json_files, dataset_dir="faiss_data",
                      index_filename="faiss_index.bin", docs_filename="documents.pkl",
                      model_name="all-MiniLM-L6-v2", nlist=100, batch_size=128):
    """
    Load or build a FAISS index from pre-chunked JSON files.
    """
    os.makedirs(dataset_dir, exist_ok=True)
    index_path = os.path.join(dataset_dir, index_filename)
    docs_path = os.path.join(dataset_dir, docs_filename)
    model_path = os.path.join(dataset_dir, "model_name.txt")

    if os.path.exists(index_path) and os.path.exists(docs_path) and os.path.exists(model_path):
        logger.info("Loading precomputed FAISS index and documents...")
        index = faiss.read_index(index_path)
        saved_model_name = open(model_path, "r").read().strip()
        model = SentenceTransformer(saved_model_name)
        with open(docs_path, "rb") as f:
            documents = pickle.load(f)
    else:
        logger.info("Building FAISS index from scratch...")
        documents = load_chunked_json_files(json_files)
        index, model = build_faiss_index(
            documents, model_name=model_name, nlist=nlist, batch_size=batch_size
        )
        faiss.write_index(index, index_path)

        with open(model_path, "w") as f:
            f.write(model_name)
        with open(docs_path, "wb") as f:
            pickle.dump(documents, f)
        logger.info("FAISS index and documents saved.")

    return index, documents, model


# Helper function to clean document context for LLM
def clean_document(doc: str) -> str:
    """
    Remove extra indentation and collapse whitespace so the LLM receives
    a compact, uniform string.
    """
    doc = textwrap.dedent(doc)
    doc = re.sub(r'\s+', ' ', doc)
    return doc.strip()