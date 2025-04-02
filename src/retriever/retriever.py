import pandas as pd
import numpy as np
import os
import json
from datasets import load_dataset

from tqdm.auto import tqdm
import logging
import pickle
import faiss
from sentence_transformers import SentenceTransformer
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

logger = logging.getLogger('legal-rag')

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
    Processes documents in batches to visualize progress with tqdm.
    """
    model = SentenceTransformer(model_name)
    
    # Compute embeddings in batches
    all_embeddings = []
    for i in tqdm(range(0, len(documents), batch_size), desc="Encoding documents"):
        batch = documents[i:i+batch_size]
        batch_embeddings = model.encode(batch, convert_to_numpy=True)
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
    return [(idx, documents[idx], distances[0][i]) for i, idx in enumerate(indices[0])]

def download_cuad_dataset(retrieval_dataset):
    """
    Download the retrieval dataset from the Hugging Face Hub.
    """
    logger.info(f"Downloading retrieval dataset from Hugging Face: {retrieval_dataset}...")
    hf_dataset = load_dataset(retrieval_dataset)
    return hf_dataset["train"]["context"]

def setup_faiss_index(retrieval_dataset, dataset_dir="faiss_data", dataset_id="cuad", chunk_size=2000, chunk_overlap=200):
    os.makedirs(dataset_dir, exist_ok=True)
    dataset_path = os.path.join(dataset_dir, "chunked_dataset_no_duplicates.json")
    index_path = os.path.join(dataset_dir, "faiss_index.bin")
    model_path = os.path.join(dataset_dir, "model_name.txt")

    if os.path.exists(index_path):
        logger.info("Loading precomputed FAISS index...")
        index = faiss.read_index(index_path)
        model_name = open(model_path, "r").read().strip()
        model = SentenceTransformer(model_name)
        logger.info("FAISS index successfully loaded.")

        with open(dataset_path, "r") as f:
            chunked_texts = json.load(f)

    elif os.path.exists(dataset_path):
        logger.info("FAISS index not found, but dataset exists. Building FAISS index from saved dataset...")
        with open(dataset_path, "r") as f:
            chunked_texts = json.load(f)

        index, model = build_faiss_index(chunked_texts)
        faiss.write_index(index, index_path)
        
        with open(model_path, "w") as f:
            f.write("all-MiniLM-L6-v2")
        logger.info("FAISS index successfully built and saved.")
    else:
        logger.info("No existing FAISS index or saved dataset. Downloading and processing dataset...")
        raw_texts = download_cuad_dataset(retrieval_dataset)
        document_objects = build_documents(raw_texts, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        chunked_texts = [doc.page_content for doc in document_objects]

        with open(dataset_path, "w") as f:
            json.dump(chunked_texts, f)
        logger.info("Chunked dataset saved for future use.")

        logger.info("Building FAISS index...")
        index, model = build_faiss_index(chunked_texts)
        faiss.write_index(index, index_path)

        with open(model_path, "w") as f:
            f.write("all-MiniLM-L6-v2")
        logger.info("FAISS index successfully built and saved.")

    return index, chunked_texts, model
