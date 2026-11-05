import gradio as gr
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
import pandas as pd
from prompt import system_prompt
from langchain_google_genai import ChatGoogleGenerativeAI
from typing import List, Tuple
import json 
from dotenv import load_dotenv
import os


load_dotenv()

# Initialize the retrieval system
CHROMA_DB_DIR = "./Database"
embeddings = HuggingFaceEmbeddings(model_name="avichr/heBERT")
db = Chroma(persist_directory=CHROMA_DB_DIR, embedding_function=embeddings)
retriever = db.as_retriever(search_kwargs={"k": 7})
model = ChatGoogleGenerativeAI(model="gemini-2.0-flash-001")

def format_metadata(metadata_dict: dict) -> str:
    """Format metadata into readable key-value pairs"""
    formatted_str = ""
    for key, value in metadata_dict.items():
        formatted_str += f"{key}: {value} \n"
    return formatted_str

def format_context(results_with_scores: List[Tuple]):
    """Format retrieved documents into a context string"""
    context = "\n\n".join([
        f"Document {i+1}:\n{doc.page_content}\nSource: {format_metadata(doc.metadata)}"
        for i, (doc, _) in enumerate(results_with_scores)
    ])
    return context

def generate_response(query: str, context: str) -> str:
    """Generate response using RAG"""
    prompt = system_prompt.format(context=context, question=query)
    response = model.invoke(prompt)
    return response.content

def process_query(query_text: str):
    """Process the query and return both retrieved documents and generated response"""
    if not query_text.strip():
        return pd.DataFrame(), ""
    
    # Get relevant documents with scores
    results_with_scores = db.similarity_search_with_score(query_text, k=10)
    
    # Format context for RAG
    context = format_context(results_with_scores)
    
    # Generate response using RAG
    generated_response = generate_response(query_text, context)
    # generated_response = "This is a placeholder response."  # 
    
    # Prepare data for display
    data = []
    for doc, score in results_with_scores:
        
        data.append({
            'Content': doc.page_content,
            'Score': f"{score:.4f}",
            'Project': doc.metadata.get("project", "N/A"),
            'File Name': doc.metadata.get("file_name", "N/A"),
            'File Path': doc.metadata.get("file_path", "N/A"),
            'Page': doc.metadata.get("page", "N/A"),
            'Keywords': doc.metadata.get("keywords", "N/A")
        })
    
    # Create DataFrame
    df = pd.DataFrame(data)
    return df, generated_response
