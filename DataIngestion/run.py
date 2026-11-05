from langchain_chroma import Chroma
from helper import extract_keywords_with_pos, load_embeddings
import json
import os

CHROMA_DB_DIR = "./Database"
RESULTS_DIR = "./Results"
RESULTS_FILE = "results.json"

# Initialize Chroma database
db = Chroma(persist_directory=CHROMA_DB_DIR, embedding_function=load_embeddings())
retriever = db.as_retriever(search_kwargs={"k": 5})

# Query
query = """when is project “הרצליה קו תכנון פרויקט“ expected to start"""
keywords = extract_keywords_with_pos(query, num_keywords=5)
print("Extracted Keywords:", keywords)

# Retrieve documents
results = retriever.get_relevant_documents(query)

# Prepare serializable data
output_data = {
    "query": query,
    "extracted_keywords": keywords,
    "results": [
        {
            "page_content": doc.page_content,
            "metadata": doc.metadata
        }
        for doc in results
    ]
}

# Ensure output directory exists
os.makedirs(RESULTS_DIR, exist_ok=True)

# Save results to JSON file
output_path = os.path.join(RESULTS_DIR, RESULTS_FILE)
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(output_data, f, ensure_ascii=False, indent=4)

print(f" Results saved successfully to {output_path}")
