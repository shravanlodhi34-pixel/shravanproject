# 45-Day Industrial Training Plan — YT-RAG

## Project at a glance

| Field | Details |
|---|---|
| Project title | YT-RAG: YouTube Video Knowledge Assistant using RAG |
| Duration | 45 days |
| Domain | Artificial Intelligence / Natural Language Processing |
| Stack | Python, Streamlit, LangChain, FAISS, Hugging Face, Groq API |
| Objective | Build an assistant that answers questions from YouTube video transcripts with evidence and timestamped sources. |

## Abstract

This project develops an intelligent question-answering system that extracts transcripts from YouTube videos, processes them into a searchable vector database, and leverages a Large Language Model (LLM) via the Groq API to answer user queries in natural language. The system is powered by Retrieval Augmented Generation (RAG), combining the precision of semantic search with the fluency of modern LLMs.

## Problem statement

Long educational and informational videos are difficult to search with ordinary keyword search. Viewers need a way to ask natural-language questions and quickly locate the relevant part of a video. The system should retrieve semantically relevant transcript passages, use them as evidence for generation, and make the evidence easy to verify.

## Objectives

1. Accept and validate common YouTube URL formats.
2. Fetch available captions and preserve their time offsets.
3. Normalize and split transcript text into semantically searchable chunks.
4. Create embeddings and index chunks in FAISS.
5. Retrieve relevant context for a user's question.
6. Generate a concise, evidence-grounded answer using Groq.
7. Show transcript excerpts and links that jump to the source timestamps.
8. Deploy a usable web app and evaluate retrieval quality, answer quality, and reliability.

## 45-Day schedule

| Phase / Days | Focus | Deliverable / exit criteria |
|---|---|---|
| Days 1–7 (Week 1) | Python, Git, APIs, NLP & RAG fundamentals; requirements and architecture | Requirements specification, architecture diagram, risk register, initial repository |
| Days 8–15 (Week 2) | YouTube URL handling, transcript extraction, timestamp preservation & cleaning | Tested transcript ingestion module; unit tests for URL parsing & timestamp calculation |
| Days 16–23 (Week 3) | Text chunking, Hugging Face embeddings, FAISS vector indexing & semantic search | Searchable local vector index; retrieval baseline testing |
| Days 24–30 (Week 4) | Groq integration, Whisper voice STT, prompt engineering & multilingual citation logic | Grounded multilingual Q&A with Groq + Whisper microphone voice input |
| Days 31–38 (Week 5) | Streamlit UI/UX, audio read-aloud (gTTS), error handling, unit testing & Docker | Complete web application, interactive speech UI, passing test suite, Dockerfile |
| Days 39–45 (Week 6) | Evaluation, performance benchmarking, documentation, report & presentation | Final deployment, benchmark results, project report, presentation slides & demo |

## Proposed evaluation

Build a manually reviewed test set from several public educational videos. For each video, write questions whose answers are present in the transcript plus a few unanswerable questions. Record the expected evidence timestamp(s), then assess:

- **Retrieval relevance:** whether the retrieved top-5 passages contain the expected evidence.
- **Answer correctness:** whether the response accurately answers the question using the transcript.
- **Faithfulness:** whether every factual claim is supported by retrieved passages.
- **Citation usefulness:** whether source links point near the supporting transcript content.
- **Robustness:** behavior for invalid URLs, missing captions, empty transcripts, and API/network failures.
- **Latency and cost:** index-build time, per-question response time, and API usage under the test workload.

Report the number of examples and scoring rubric. Do not claim a metric without recording the evaluation data and method. Useful future additions include Recall@k, Mean Reciprocal Rank, and a human-rated answer faithfulness score.

## Main modules

- **UI:** Streamlit URL form, chat interface, source citations and settings.
- **Ingestion:** URL parsing and YouTube transcript retrieval.
- **Preprocessing:** whitespace normalization, timestamp-aware grouping, chunking.
- **Retrieval:** Hugging Face sentence embeddings and FAISS similarity search.
- **Generation:** Groq chat completion with transcript-only grounding instructions.
- **Operations:** environment-based secrets, Docker image, health check and deployment docs.

## Risks and mitigations

| Risk | Mitigation |
|---|---|
| Captions unavailable or blocked | Explain the failure clearly; test with caption-enabled public videos. |
| Hallucinated answers | Restrict context, instruct abstention, show the retrieved excerpts, evaluate faithfulness. |
| Inaccurate source time | Preserve caption offsets through grouping and display direct timestamp links. |
| Secret exposure | Read key from secret/environment or masked input; never commit `.env` or Streamlit secrets. |
| Cold start / model memory | Use a compact embedding model; set appropriate hosting memory; consider caching and pre-warming. |
| API limits or downtime | Show recoverable errors, document provider limits, and consider retry/backoff for a production version. |
| Index lost on restart | Current prototype is session-scoped; persistent storage is an explicit future enhancement. |

## Final submission checklist

- [ ] Source code and versioned dependencies
- [ ] README with local setup and deployment instructions
- [ ] Requirements, architecture, and 45-day plan
- [ ] Tests for URL parsing, timestamps, and transcript chunking
- [ ] Evaluation dataset/method and recorded results
- [ ] Deployed app URL and demo video/screenshots
- [ ] Security and limitations section
- [ ] Final report and presentation
