# YT-RAG: YouTube Video Knowledge Assistant

A deployable 45-day industrial-training project in AI/NLP. It retrieves public YouTube captions, builds a local FAISS vector index using Hugging Face sentence-transformer embeddings, and asks Groq to answer questions grounded in retrieved transcript passages. Answers include transcript excerpts and links that jump to the relevant timestamps.

## Features

- Accepts YouTube watch, `youtu.be`, Shorts, embed, and live URLs.
- Retrieves English captions/transcripts when available.
- Uses LangChain text splitting, Hugging Face embeddings, and FAISS similarity search.
- Uses Groq chat completions for grounded answers; asks the model to say when the transcript does not contain the answer.
- Displays source passages with timestamp links into the video.
- Keeps the API key out of source code; use an environment variable, Streamlit secret, or sidebar input.
- Docker-ready and suitable for Streamlit Community Cloud.

## Architecture

```text
YouTube URL
    │
    ├── YouTube Transcript API ──> timestamped caption text
    │                                  │
    │                           chunking (LangChain)
    │                                  │
    │                Hugging Face sentence embeddings
    │                                  │
    │                             FAISS index
    │                                  │
Question ──> semantic retrieval ──> relevant transcript passages
                                       │
                                  Groq LLM
                                       │
                        grounded response + timestamp sources
```

## Run locally

Requirements: Python 3.10 or 3.11 recommended. The first run downloads the embedding model and can take a few minutes; a CPU is sufficient for the default model.

```bash
cd yt-rag
python -m venv .venv
# macOS/Linux
source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
cp .env.example .env
```

Put your Groq API key in `.env` as `GROQ_API_KEY=...`, or enter it in the app sidebar. The app loads `.env` automatically for local development. A convenient cross-platform alternative is to set the variable in your shell:

```bash
# macOS/Linux
export GROQ_API_KEY="your_groq_api_key"
# Windows PowerShell
$env:GROQ_API_KEY="your_groq_api_key"
```

Then run:

```bash
streamlit run app.py
```

Open the URL printed by Streamlit, paste a video URL, build the index, and ask a question. You can obtain a Groq API key from the Groq developer console. Do not commit your key or share it publicly.

## Deploy with Docker

```bash
docker build -t yt-rag .
docker run --rm -p 8501:8501 -e GROQ_API_KEY="your_groq_api_key" yt-rag
```

Open `http://localhost:8501`. For a hosted Docker service, set `GROQ_API_KEY` as a secret environment variable in the hosting provider's dashboard; do not bake it into the image.

## Deploy to Streamlit Community Cloud

1. Push this project to a GitHub repository.
2. In Streamlit Community Cloud, create an app from the repository and select `app.py` as the entry point.
3. Add `GROQ_API_KEY = "your_groq_api_key"` under the app's **Secrets** settings (TOML format).
4. Deploy. Keep the repository public or connect the private repository through Streamlit's supported GitHub access flow.

The first request may take longer while the Hugging Face embedding model downloads. For larger videos or multi-user production traffic, deploy with more memory and consider a persistent vector database, background indexing, authentication, and request limits.

## Configuration

The sidebar allows changing the Groq model and Hugging Face embedding model. Defaults:

- Embeddings: `sentence-transformers/all-MiniLM-L6-v2`
- Groq model: `llama-3.3-70b-versatile`
- Retrieved passages per question: 5
- Chunk size / overlap: 850 / 120 characters

## Limitations and responsible use

- A transcript/captions track must be available to the transcript service. Videos with disabled captions, inaccessible/private videos, and some age-restricted or region-restricted videos may not work.
- This prototype searches English transcripts. It does not download or process video/audio itself.
- Responses are limited to transcript evidence and can still be wrong; verify important claims against the video and cited passages.
- The FAISS index is session-only and is not persisted. Do not put confidential content or API keys into prompts. Follow YouTube's terms and applicable copyright/privacy requirements.
- Groq and Hugging Face have their own service terms, quotas, and data-handling policies. Review them before production deployment.

## Project layout

```text
app.py                 Streamlit user interface
 yt_rag.py             transcript, chunking, embeddings, FAISS, retrieval
requirements.txt       Python dependencies
Dockerfile             container deployment
.streamlit/config.toml Streamlit defaults
tests/                 small unit-test suite
TRAINING_PLAN.md       45-day plan, objectives, milestones, evaluation
```

## Tests

```bash
python -m unittest discover -s tests -v
```

## Abstract

This project develops an intelligent question-answering system that extracts transcripts from YouTube videos, processes them into a searchable vector database, and leverages a Large Language Model (LLM) via the Groq API to answer user queries in natural language. The system is powered by Retrieval Augmented Generation (RAG), combining the precision of semantic search with the fluency of modern LLMs.
