# store_index.py
import time
import logging
from dotenv import load_dotenv
from DataIngestion.helper import (
    load_project_documents,
    split_documents,
    load_embeddings,
    add_bm25_keywords_to_metadata,
)
from langchain_chroma import Chroma

#  Setup
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

load_dotenv()

ROOT_DIR = "../Projects"
CHROMA_DB_DIR = "../Database"

#  Step 1: Load Project Documents
logger.info(" Loading project documents...")
try:
    docs = load_project_documents(ROOT_DIR)
    logger.info(f"✅ Loaded {len(docs)} documents from {ROOT_DIR}")
except Exception as e:
    logger.error(f" Error loading documents: {e}")
    raise

# Step 2: Add BM25 Keywords
try:
    docs = add_bm25_keywords_to_metadata(docs, top_n=10)
    logger.info("BM25 keywords successfully merged with POS keywords.")
except Exception as e:
    logger.error(f" Error adding BM25 keywords: {e}")
    raise

# Step 3: Split into Text Chunks
try:
    text_chunks = split_documents(docs)
    logger.info(f" Split documents into {len(text_chunks)} text chunks.")
except Exception as e:
    logger.error(f" Error splitting documents: {e}")
    raise

#  Step 4: Load Embedding Model
try:
    logger.info(" Loading Hebrew embedding model...")
    embeddings = load_embeddings()
    logger.info(" Embedding model loaded successfully.")
except Exception as e:
    logger.error(f" Error loading embeddings: {e}")
    raise

#  Step 5: Store Chunks in ChromaDB
# ---------------------------------
logger.info(" Initializing ChromaDB and storing embeddings...")
start_time = time.time()

try:
    vectorstore = Chroma.from_documents(
        documents=text_chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DB_DIR,

    )

    total_chunks = len(text_chunks)
    logger.info(f" Stored {total_chunks} chunks successfully in {CHROMA_DB_DIR}")

except Exception as e:
    logger.error(f" Error during indexing: {e}")
    raise

end_time = time.time()
logger.info(f" Indexing completed in {(end_time - start_time):.2f} seconds.")
logger.info(" Chroma index build completed successfully.")
