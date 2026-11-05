from pathlib import Path
from langchain_community.document_loaders import PyMuPDFLoader, UnstructuredExcelLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from rank_bm25 import BM25Okapi
from tqdm import tqdm
import re
import logging
import stanza
from collections import Counter

#  Setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Hebrew Stanza model (load once)
try:
    stanza.download("he")
    nlp = stanza.Pipeline("he", processors="tokenize,pos,lemma")
except Exception as e:
    logger.error(f"❌ Failed to load Stanza Hebrew model: {e}")
    nlp = None

#  Hebrew Text Normalization
def normalize_hebrew(text: str):
    """Clean and normalize Hebrew text."""
    text = text.lower()
    text = re.sub(r'[^\u0590-\u05FF\s]', ' ', text)  # Keep only Hebrew characters + spaces
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

#  Hebrew POS-based Keyword Extraction (Stanza)

def extract_keywords_with_pos(text: str, num_keywords=5):
    """Extract top Hebrew keywords using Stanza POS tagging."""
    if not nlp:
        return []

    text = normalize_hebrew(text)
    doc = nlp(text)
    allowed_pos = { "PROPN","NOUN"}
    lemmas = [
        w.lemma for s in doc.sentences for w in s.words
        if w.upos in allowed_pos and len(w.lemma) > 1
    ]

    if not lemmas:
        return []

    counts = Counter(lemmas)
    top_keywords = [word for word, _ in counts.most_common(num_keywords)]
    return top_keywords

#  Load Project Documents
def load_project_documents(root_dir: str):
    """Loads all PDFs/XLSX from subdirectories with cleaned metadata."""
    all_documents = []
    project_dirs = [p for p in Path(root_dir).iterdir() if p.is_dir()]

    for project_path in tqdm(project_dirs, desc="Projects", unit="project"):
        project_name = project_path.name
        files = list(project_path.glob("*"))

        for file_path in tqdm(files, desc=f"Loading {project_name}", unit="file", leave=False):
            try:
                docs = []
                if file_path.suffix.lower() == ".pdf":
                    loader = PyMuPDFLoader(str(file_path))
                    docs = loader.load()
                elif file_path.suffix.lower() in [".xls", ".xlsx"]:
                    loader = UnstructuredExcelLoader(str(file_path))
                    docs = loader.load()
                else:
                    logger.warning(f"⚠️ Unsupported file type, skipping: {file_path.name}")
                    continue

                # Clean irrelevant metadata
                remove_keys = {"author","creationdate","creator","format","author","subject","keywords","trapped",}

                for doc in docs:
                    for key in list(doc.metadata.keys()):
                        if key in remove_keys:
                            del doc.metadata[key]

                # POS keyword extraction
                    pos_keywords = extract_keywords_with_pos(doc.page_content)


                    # Attach metadata
                    doc.metadata.update({
                        "project": project_name,
                        "file_path": str(file_path),
                        "file_name": file_path.name,
                        "keywords": ", ".join(pos_keywords) if pos_keywords else ""
                    })

                all_documents.extend(docs)
            except Exception as e:
                logger.error(f"❌ Error loading {file_path.name}: {e}")

    return all_documents

#  Split into Chunks
def split_documents(documents, chunk_size=500, chunk_overlap=50):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    chunks = []
    for doc in tqdm(documents, desc="Splitting documents", unit="doc"):
        doc_chunks = text_splitter.split_documents([doc])
        for chunk in doc_chunks:
            chunk.metadata.update(doc.metadata)
        chunks.extend(doc_chunks)
    return chunks

#  Load Hebrew Embedding Model
def load_embeddings(model_name="avichr/heBERT"):
    return HuggingFaceEmbeddings(model_name=model_name )

#  BM25 Keyword Scoring (Optional Enhancement)
def add_bm25_keywords_to_metadata(documents, top_n=10):
    """
    Uses BM25 to find top unique keywords by IDF score for each doc
    (works on pre-cleaned Hebrew text).
    Merges BM25 keywords with existing POS-based keywords.
    """
    logger.info(" Building BM25 keyword index...")

    # Step 1: Tokenize all documents
    tokenized_corpus = [normalize_hebrew(doc.page_content).split() for doc in documents]
    bm25 = BM25Okapi(tokenized_corpus)
    idf_scores = bm25.idf

    # Step 2: Add BM25 keywords
    for i, doc in enumerate(tqdm(documents, desc="Adding BM25 keywords")):
        tokens = tokenized_corpus[i]
        # Get IDF score per token
        token_scores = {t: idf_scores.get(t, 0) for t in tokens}

        # Sort by IDF and pick top N
        sorted_tokens = sorted(token_scores.items(), key=lambda x: x[1], reverse=True)
        bm25_keywords = [t for t, _ in sorted_tokens[:top_n]]

        # Step 3: Merge with existing POS-based keywords (if any)
        existing_keywords_str = doc.metadata.get("keywords", "")
        existing_keywords = [
            kw.strip() for kw in existing_keywords_str.split(",") if kw.strip()
        ]

        merged_keywords = sorted(set(existing_keywords + bm25_keywords))

        # Step 4: Update metadata cleanly
        doc.metadata["keywords"] = ", ".join(merged_keywords)

    logger.info("BM25 keywords merged with POS-based keywords.")
    return documents
