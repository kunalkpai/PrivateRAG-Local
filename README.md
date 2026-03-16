# 🔍 Multi-Document RAG System

> [!IMPORTANT]
> **Privacy First**: This project is configured to keep your data local. Your documents and keys are excluded from git.

A fully **local**, privacy-preserving Retrieval-Augmented Generation (RAG) pipeline built with:

| Component | Technology |
|---|---|
| LLM | [Ollama](https://ollama.com) (local) or Google Gemini (cloud) |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` |
| Vector Store | [ChromaDB](https://www.trychroma.com/) (persisted to disk) |
| RAG Framework | [LangChain](https://python.langchain.com/) |
| Document Loaders| `PyPDFLoader`, `docx2txt`, `unstructured`, `pandas` |

No API keys required — everything runs on your machine.

---

## 📁 Project Structure

```
multi_doc_rag/
├── config.py       # All settings (chunk size, model names, paths…)
├── ingest.py       # Load → Split → Embed → Store to Chroma
├── retriever.py    # Load Chroma and expose a LangChain retriever
├── rag_chain.py    # LangChain LCEL RAG chain (Ollama LLM)
├── main.py         # Interactive CLI
├── utils.py        # Logging + source formatting helpers
├── requirements.txt
├── .env.example
├── data/           ← DROP YOUR PDFs HERE
├── chroma_db/      ← Auto-created by ingest.py
└── tests/
    ├── test_splitter.py
    └── test_ingest_integration.py
```

---

## 🚀 Quick Start

### 1. Prerequisites (Ollama or Gemini)

**If using Ollama (Local)**:
Install [Ollama](https://ollama.com/download) and pull a model:
```bash
ollama serve          # start the Ollama daemon (keep this running)
ollama pull llama3    # ~4 GB — one-time download
```

**If using Gemini (Cloud)**:
Get an API key from [Google AI Studio](https://aistudio.google.com/app/apikey).

### 2. Install Python dependencies

```bash
cd multi_doc_rag
python -m venv .venv && source .venv/bin/activate  # recommended
pip install -r requirements.txt
```

### 3. Add your Documents

Drop any supported document type into the `data/` folder:
- PDF (`.pdf`)
- Word (`.docx`, `.doc`)
- Markdown / Text (`.md`, `.txt`)
- Excel (`.xlsx`, `.xls`)

```bash
cp /path/to/your/document.pdf data/
```

### 4. Ingest → Chat in one command

```bash
python -m app.main --ingest
```

Or run ingestion and Q&A separately:

```bash
python -m app.ingest          # build / update the vector store
python -m app.main            # start the Q&A session
```

---

## 💬 CLI Options

```
python -m app.main [OPTIONS]

Options:
  --ingest         Run ingestion before starting Q&A
  --ingest-only    Ingest and exit (no chat)
  --model MODEL    Override LLM model (e.g. mistral, gemini-1.5-flash)
  --k K            Number of chunks to retrieve per query (default 5)
  --silent         Run in silent mode, suppressing all internal logs

```

---

## ⚙️ Configuration

Copy `.env.example` to `.env` and edit:

```bash
cp .env.example .env
```

| Variable | Default | Description |
|---|---|---|
| `LLM_PROVIDER` | `ollama` | Which LLM backend to use (`ollama` or `gemini`) |
| `OLLAMA_MODEL` | `llama3` | Ollama model to use |
| `GEMINI_MODEL` | `gemini-1.5-pro` | Gemini model to use |
| `GEMINI_API_KEY` | `None` | Automatically picked up if set |
| `CHUNK_SIZE` | `1000` | Characters per chunk |
| `CHUNK_OVERLAP` | `200` | Overlap between chunks |
| `RETRIEVER_K` | `5` | Chunks retrieved per query |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Sentence embedding model |

---

## 🧠 Architecture

```
                  ┌─────────────────────────┐
  PDF files ───►  │        ingest.py        │
                  │  PyPDFLoader            │
                  │  RecursiveCharSplitter  │
                  │  SentenceTransformer    │
                  │  ChromaDB (persist)     │
                  └─────────────────────────┘
                              │
                              ▼ (at query time)
  User query ──► retriever.py (MMR search, k=5)
                              │
                              ▼
               rag_chain.py (LangChain LCEL)
               ┌─────────────────────────────┐
               │  Prompt + Context           │
               │  OllamaLLM (llama3)         │
               │  StrOutputParser            │
               └─────────────────────────────┘
                              │
                              ▼
              Answer + Citations (doc name, page 1-indexed)
```

---

## 🧪 Running Tests

```bash
pytest tests/ -v
```

---

## 📝 Adding New Documents

Simply copy new PDFs to `data/` and run:

```bash
python -m app.ingest
```

The ingestion pipeline **synchronizes** your `data/` folder with ChromaDB: new documents are added, already-indexed documents are skipped, and deleted files are automatically purged from the vector database.
