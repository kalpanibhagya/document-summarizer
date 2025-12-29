# Document Summarizer

A small desktop GUI app to upload files, preview data, generate basic statistics, create embeddings (via Ollama), and ask data-driven questions using retrieval-augmented generation (RAG).

## Features
- Upload and preview files
- Basic dataset summary and numeric statistics
- Generate embeddings for efficient retrieval
- Ask questions or click "Summarize" to get concise AI-driven insights
- Switch models from the UI (e.g., gemma3:1b, llama3.2, etc.)

## Requirements
- Python 3.8+
- Ollama installed and available in PATH (app will attempt to start it)
- Python packages: `customtkinter`, `requests`, `numpy` (install with pip)

## Quick Start
1. Ensure Ollama is installed and accessible (see https://ollama.ai).
2. Install Python packages: `pip install customtkinter requests numpy`
3. Run the app: `python main.py`
4. In the UI: Upload a file → (optional - use for large files) Generate Embeddings → Ask questions or click Summarize

## Important notes
- Generating embeddings enables full-data retrieval for more accurate answers; without them the app uses sample rows.
- The app starts/controls an Ollama server process via `OllamaServer.py` and uses `ollama` for embeddings and chat.

## Files
- `Summarizer.py` — GUI and core logic
- `CSVParser.py` — CSV parsing, chunking, embeddings, simple stats
- `OllamaServer.py` — manages the Ollama server process
