# Hebrew Document Intelligence System

A document retrieval and question-answering system optimized for Hebrew text documents, using RAG (Retrieval-Augmented Generation) architecture.

## 🌟 Features

- Hebrew-optimized document processing and retrieval
- Intelligent keyword extraction using both POS tagging and BM25 scoring
- Support for PDF and Excel documents
- Advanced document chunking with metadata preservation
- Interactive query interface using Gradio
- RAG-based response generation using Google's Gemini model

## 🏗️ System Architecture

### Data Ingestion Pipeline
1. **Document Loading** (`DataIngestion/helper.py`)
   - Supports PDF and Excel files
   - Maintains folder-based project structure
   - Preserves document metadata

2. **Text Processing**
   - Hebrew text normalization
   - Keyword extraction using:
     - Stanza POS tagging for Hebrew
     - BM25 scoring for relevance

3. **Document Storage**
   - ChromaDB vector database
   - HuggingFace Hebrew embeddings (avichr/heBERT)

### Query Processing (`process_user_query.py`)
- Semantic search using HuggingFace embeddings
- Context-aware response generation using Gemini model
- Structured output with relevance scores and metadata

## 🚀 Getting Started

### Prerequisites

Create a `.env` file from `.env.example` and add your own Google (Gemini) and Hugging Face keys.

```sh
pip install -r requirements.txt
```


### Data Ingestion
Put your Hebrew PDF and Excel files in `Projects/`, one sub-folder per project (the folder name is kept as metadata). To build or rebuild the index, empty the `Database/` folder and run:

```sh
python -m DataIngestion.store_index
```

### Manual Run for Retrieval Testing
Just  for Testing The Retrieval from Embedding.

```sh
python -m DataIngestion.run
```

### Running the Interface

```sh
python app.py
```

## 📁 Project Structure

```
├── app.py                  # Main application entry point
├── process_user_query.py   # Query processing logic
├── DataIngestion/
│   ├── helper.py          # Core processing utilities
│   ├── run.py            # Test retrieval script
│   └── store_index.py    # Document indexing pipeline
├── Database/              # ChromaDB storage
└── Projects/              # Document source directory (add your own files)
```

## 🛠️ Technical Details

### Embedding Model
- Using `avichr/heBERT` for Hebrew-optimized embeddings

### Text Processing
- Chunk size: 500 characters
- Chunk overlap: 50 characters
- Hebrew-specific text normalization
- Dual keyword extraction (POS + BM25)

### Retrieval
- Similarity search with metadata preservation
- Score-based ranking

## 📝 Notes

- Ensure Hebrew language support is properly configured
- Monitor ChromaDB storage growth
- Regular reindexing recommended for optimal performance

## 🔧 Configuration

Key parameters can be adjusted in respective files:
- Chunk size and overlap in `helper.py`
- Retrieval parameters in `process_user_query.py`
- Keywords extraction settings in `helper.py`