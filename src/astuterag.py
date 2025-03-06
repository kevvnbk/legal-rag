import os
import json
import logging
import re
from typing import List, Tuple
import torch
from tqdm import tqdm

def prompt_generate_passages(query: str, max_passages: int = 5) -> str:
    """
    Generates a prompt for the LLM to produce up to `max_passages` passages 
    relevant to the query.
    """
    return f"""You are a helpful assistant. 
Your task is to generate up to {max_passages} passages that provide accurate and relevant information 
to answer the following question. If the information is unclear or uncertain, explicitly state 
"I don't know" to avoid any hallucinations.

Question: {query}

Instructions:
- Do not invent facts.
- Provide at most {max_passages} concise passages or knowledge statements.
- If unsure, say "I don't know."
- Return them as a list of passages clearly delimited.
"""

def prompt_consolidate(
    query: str, 
    docs: List[str], 
    prev_summary: str
) -> str:
    """
    Generates a prompt for the LLM to consolidate multiple documents into two sections:
    CONSOLIDATED_CONTENT and CONSOLIDATED_SUMMARY.
    """
    docs_str = "\n\n".join([f"Document {i+1}:\n{doc}" for i, doc in enumerate(docs)])
    return f"""You are an expert system that consolidates information from multiple sources 
to answer a user query.

Query: {query}

Below are the documents to consolidate:
{docs_str}

Previous Consolidation Summary (if any):
{prev_summary}

Instructions:
1. Combine documents that provide consistent information into a single consolidated view.
2. If documents conflict, identify the conflict clearly.
3. Summarize the key points in a concise, well-organized way.
4. Exclude any irrelevant information.
5. Return two sections:
   - CONSOLIDATED_CONTENT: A single consolidated text of the combined knowledge.
   - CONSOLIDATED_SUMMARY: A short summary of the combined knowledge.
"""

def prompt_answer_final(
    query: str, 
    consolidated_content: str, 
    consolidated_summary: str
) -> str:
    """
    Generates a prompt for the LLM to produce the final answer based on the consolidated content.
    """
    return f"""You are a knowledgeable assistant. 
Use the consolidated information to answer the final user query.

Query: {query}

Consolidated Content:
{consolidated_content}

Consolidated Summary:
{consolidated_summary}

Instructions:
1. Provide the best possible answer based on the information above.
2. If the information is unclear or uncertain, say "I don't know" to avoid hallucinations.
3. Support your answer with brief reasoning when possible.

Final Answer (please provide a direct answer below):
"""

def parse_generated_passages(llm_output: str) -> List[str]:
    """
    Parses the LLM output from the passage generation step into a list of passages.
    """
    lines = [line.strip() for line in llm_output.split('\n') if line.strip()]
    return lines

def parse_consolidation_output(llm_output: str) -> Tuple[str, str]:
    """
    Parses the LLM output from the consolidation step into CONSOLIDATED_CONTENT 
    and CONSOLIDATED_SUMMARY.
    """
    content_pattern = r'CONSOLIDATED_CONTENT:\s*(.*?)(?=CONSOLIDATED_SUMMARY:|$)'
    summary_pattern = r'CONSOLIDATED_SUMMARY:\s*(.*)'
    
    content_match = re.search(content_pattern, llm_output, re.DOTALL)
    summary_match = re.search(summary_pattern, llm_output, re.DOTALL)
    
    consolidated_content = content_match.group(1).strip() if content_match else ""
    consolidated_summary = summary_match.group(1).strip() if summary_match else ""
    
    return consolidated_content, consolidated_summary

def astute_rag_pipeline(
    query: str,
    retrieve_passages_fn,   # function: str -> List[str]
    call_llm_fn,            # function: str -> str
    num_iterations: int = 2,
    max_generated_passages: int = 5
) -> str:
    """
    Implements the AstuteRAG pipeline:
    
    1) Retrieves external documents based on the query.
    2) Generates passages using the LLM.
    3) Consolidates the retrieved and generated information iteratively.
    4) Produces the final answer.
    """
    # Step 1: Retrieve external documents
    retrieved_docs = retrieve_passages_fn(query)  # E

    # Step 2: Generate passages using the LLM
    pgen_prompt = prompt_generate_passages(query, max_generated_passages)
    pgen_output = call_llm_fn(pgen_prompt)
    generated_docs = parse_generated_passages(pgen_output)  # I

    # Combine retrieved and generated docs
    D0 = retrieved_docs + generated_docs

    consolidated_content = ""
    consolidated_summary = ""

    if num_iterations > 1:
        D_j = D0
        for iteration in range(num_iterations - 1):
            pcon_prompt = prompt_consolidate(
                query=query, 
                docs=D_j, 
                prev_summary=consolidated_summary
            )
            pcon_output = call_llm_fn(pcon_prompt)
            consolidated_content, consolidated_summary = parse_consolidation_output(pcon_output)
            # For next iteration, use the latest consolidated content
            D_j = [consolidated_content]
        
        # Final Answer step
        pans_prompt = prompt_answer_final(query, consolidated_content, consolidated_summary)
        final_answer = call_llm_fn(pans_prompt)
        return final_answer.strip()
    else:
        # Single iteration consolidation
        pcon_prompt = prompt_consolidate(
            query=query, 
            docs=D0, 
            prev_summary=""
        )
        pcon_output = call_llm_fn(pcon_prompt)
        consolidated_content, consolidated_summary = parse_consolidation_output(pcon_output)
        pans_prompt = prompt_answer_final(query, consolidated_content, consolidated_summary)
        final_answer = call_llm_fn(pans_prompt)
        return final_answer.strip()
