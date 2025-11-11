# CV Comparison & Ranking (English README)

This project extracts text from PDF CVs, generates embeddings using different backends (local SentenceTransformer, OpenAI, Google Gemini or a local API server), and computes cosine similarity against a search query. It supports chunked analysis (compare fragments instead of full documents), returns the top matching fragments, and provides a simple CLI and FastAPI server to run batch or remote comparisons.

---

## Features

- PDF text extraction using PyPDF2
- Local embeddings via SentenceTransformer (Hugging Face models)
- Remote embeddings via OpenAI and Google Gemini
- FastAPI server with endpoints for full-document and chunked comparisons
- Chunking strategies: split-by-sentence (dot), fixed character blocks, or fixed word blocks
- Control parameters: `max_chunks`, `chunk_size`, `chunk_method` and `top_n`
- Outputs: similarity score, relevance percentage, top matching fragments, and processing timings
- CLI and lightweight client to call the API and run batch evaluations

---

## Quick start / Usage examples

1. Run the API server (optional, for `--embeddings api` mode):

```powershell
uvicorn api_server:app --reload
# (optional) change model with cambiar_modelo.py
python cambiar_modelo.py --model "hiiamsid/sentence_similarity_spanish_es"
```

2. Run `vectorizacion.py` (CLI) with different backends:

```powershell
# Local sentence-transformer with chunking
python .\vectorizacion.py --embeddings sentence --chunked true --max-chunks 6

# OpenAI embeddings (requires OPENAI_API_KEY in env)
python .\vectorizacion.py --embeddings openai --chunked true --max-chunks 5

# Gemini embeddings (requires GEMINI_API_KEY in env)
python .\vectorizacion.py --embeddings gemini --chunked true --max-chunks 10

# Use local API server for embedding/compare
python .\vectorizacion.py --embeddings api --chunked true --max-chunks 7
```

3. Client usage (`client_api.py`):

```powershell
python .\client_api.py --pdf path\to\cv.pdf --query "Python developer" --embeddings api
```

---

## API endpoints

- `POST /compare/` — compare whole document (returns `similarity_score`, `relevance_percentage`, `processing_time_seconds`, `model_used`)
- `POST /compare_chunked/` — compare by fragments (supports `max_chunks`, `chunk_method`, `chunk_size`, `top_n`) and returns `chunks_analyzed`, `top_scores`, and `top_chunks`
- `GET /health` — health check

---

## Notes and recommendations

- `util.cos_sim` returns values in `[-1, 1]`. If you want an intuitive percentage (0% = -1, 100% = 1) normalize with `(score + 1) / 2` before formatting.
- Prefer batching remote embedding requests (OpenAI/Gemini) instead of one call per chunk to reduce latency and cost.
- Replace `split('.')` with sentence tokenization (spaCy or NLTK) for better sentence segmentation.
- Consider adding an overlap between chunks to avoid splitting important context.
- Validate embedding dimensions (assert `query_dim == chunk_dim`) before calculating similarity.

---

## Dependencies

See `requirements.txt`. Key packages: `sentence-transformers`, `torch`, `fastapi`, `uvicorn`, `PyPDF2`, `requests`, `openai`, `google-generativeai`.

---

If you want, I can also:
- Normalize similarity output across the codebase (so percentages are intuitive)
- Add support for sending `cv_chunks` (with ids/offsets) from the client to the server for traceability
- Implement sentence tokenization for chunking
