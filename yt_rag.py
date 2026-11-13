from __future__ import annotations

import re
from typing import Any
from urllib.parse import parse_qs, urlparse

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_LLM_MODEL = "openai/gpt-oss-120b"


def parse_video_id(url: str) -> str:
    """Validate a YouTube URL and return its 11-character video ID."""
    if not url or not url.strip():
        raise ValueError("Paste a YouTube video URL to continue.")
    parsed = urlparse(url.strip())
    host = (parsed.hostname or "").lower().removeprefix("www.")
    video_id = ""
    if host in {"youtube.com", "m.youtube.com", "music.youtube.com"}:
        if parsed.path == "/watch":
            video_id = parse_qs(parsed.query).get("v", [""])[0]
        else:
            match = re.match(r"^/(?:embed|shorts|live)/([A-Za-z0-9_-]{11})(?:/|$)", parsed.path)
            if match:
                video_id = match.group(1)
    elif host == "youtu.be":
        video_id = parsed.path.strip("/").split("/")[0]
    if not re.fullmatch(r"[A-Za-z0-9_-]{11}", video_id or ""):
        raise ValueError("That does not look like a valid YouTube video URL. Use a watch, youtu.be, Shorts, or live URL.")
    return video_id


def youtube_url(video_id: str, seconds: int | None = None) -> str:
    base = f"https://www.youtube.com/watch?v={video_id}"
    return f"{base}&t={max(0, int(seconds or 0))}s" if seconds else base


def format_timestamp(seconds: float | int) -> str:
    value = max(0, int(float(seconds)))
    hours, remainder = divmod(value, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours}:{minutes:02d}:{secs:02d}" if hours else f"{minutes}:{secs:02d}"


def _fetch_raw_transcript(video_id: str) -> list[dict[str, Any]]:
    """Smart transcript fetcher: supports manual English, auto-generated English, and any available language with translation fallback."""
    from youtube_transcript_api import YouTubeTranscriptApi

    api = YouTubeTranscriptApi()
    transcript_list = None
    try:
        if hasattr(api, "list"):
            transcript_list = api.list(video_id)
        elif hasattr(YouTubeTranscriptApi, "list_transcripts"):
            transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
    except Exception:
        pass

    if transcript_list:
        # 1. Try manual English
        try:
            t = transcript_list.find_manually_created_transcript(["en", "en-US", "en-GB", "en-IN", "en-CA", "en-AU"])
            fetched = t.fetch()
            if hasattr(fetched, "snippets"):
                return [{"text": p.text, "start": p.start, "duration": p.duration} for p in fetched.snippets]
            return [dict(p) for p in fetched]
        except Exception:
            pass

        # 2. Try auto-generated English
        try:
            t = transcript_list.find_generated_transcript(["en", "en-US", "en-GB", "en-IN", "en-CA", "en-AU"])
            fetched = t.fetch()
            if hasattr(fetched, "snippets"):
                return [{"text": p.text, "start": p.start, "duration": p.duration} for p in fetched.snippets]
            return [dict(p) for p in fetched]
        except Exception:
            pass

        # 3. Try any available transcript, translating to English if supported
        for t in transcript_list:
            try:
                target_track = t
                if getattr(t, "is_translatable", False):
                    try:
                        target_track = t.translate("en")
                    except Exception:
                        target_track = t
                fetched = target_track.fetch()
                if hasattr(fetched, "snippets"):
                    return [{"text": p.text, "start": p.start, "duration": p.duration} for p in fetched.snippets]
                return [dict(p) for p in fetched]
            except Exception:
                continue

    # Fallback to direct fetch
    try:
        fetched = api.fetch(video_id)
        if hasattr(fetched, "snippets"):
            return [{"text": p.text, "start": p.start, "duration": p.duration} for p in fetched.snippets]
        return [dict(p) for p in fetched]
    except Exception:
        pass

    # Compatibility path for older youtube-transcript-api 0.x
    return YouTubeTranscriptApi.get_transcript(video_id)


def get_transcript(video_id: str) -> list[dict[str, Any]]:
    try:
        items = _fetch_raw_transcript(video_id)
    except Exception as exc:
        detail = str(exc)
        if "TranscriptsDisabled" in type(exc).__name__ or "NoTranscriptFound" in type(exc).__name__:
            raise RuntimeError(
                "No captions/subtitles were found for this specific video. The creator has turned captions off, or YouTube has not auto-generated subtitles for it. Please try a video with captions enabled."
            ) from exc
        if "VideoUnavailable" in type(exc).__name__:
            raise RuntimeError("This video is unavailable or private. Check that it is public and try again.") from exc
        raise RuntimeError(f"Could not fetch this video's transcript: {detail}") from exc

    cleaned: list[dict[str, Any]] = []
    for item in items:
        text = re.sub(r"\s+", " ", str(item.get("text", ""))).strip()
        if text:
            cleaned.append({
                "text": text,
                "start": float(item.get("start", 0)),
                "duration": float(item.get("duration", 0)),
            })
    if not cleaned:
        raise RuntimeError("The video returned an empty transcript.")
    return cleaned


def _group_transcript(items: list[dict[str, Any]], target_chars: int = 1100) -> list[Document]:
    """Group adjacent caption lines to retain useful timestamp-level provenance."""
    docs: list[Document] = []
    group: list[str] = []
    char_count = 0
    group_start = 0.0
    group_end = 0.0

    def flush() -> None:
        nonlocal group, char_count
        if group:
            docs.append(Document(
                page_content=" ".join(group),
                metadata={"start_seconds": group_start, "end_seconds": group_end},
            ))
        group = []
        char_count = 0

    for item in items:
        line = item["text"]
        if group and char_count + len(line) > target_chars:
            flush()
        if not group:
            group_start = item["start"]
        group.append(line)
        char_count += len(line) + 1
        group_end = item["start"] + item["duration"]
    flush()
    return docs


def get_embeddings(model_name: str = DEFAULT_EMBEDDING_MODEL) -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True, "batch_size": 64},
    )


def build_index(
    transcript: list[dict[str, Any]], video_id: str, embeddings: HuggingFaceEmbeddings
) -> tuple[FAISS, int]:
    source_docs = _group_transcript(transcript)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=850,
        chunk_overlap=120,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(source_docs)
    for chunk in chunks:
        chunk.metadata["video_id"] = video_id
    if not chunks:
        raise RuntimeError("The transcript could not be split into searchable passages.")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    return vectorstore, len(chunks)


def retrieve_sources(vectorstore: FAISS, question: str, video_id: str, k: int = 5) -> list[dict[str, Any]]:
    matches = vectorstore.similarity_search(question, k=k)
    sources: list[dict[str, Any]] = []
    for doc in matches:
        start = int(doc.metadata.get("start_seconds", 0))
        sources.append({
            "text": doc.page_content,
            "start_seconds": start,
            "timestamp": format_timestamp(start),
            "url": youtube_url(video_id, start),
        })
    return sources
