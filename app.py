from __future__ import annotations

import io
import os
import re
from typing import Any

from gtts import gTTS

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

import streamlit as st
from dotenv import load_dotenv
from groq import Groq

load_dotenv(override=True)

from yt_rag import (
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_LLM_MODEL,
    build_index,
    get_embeddings,
    get_transcript,
    parse_video_id,
    retrieve_sources,
    youtube_url,
)

@st.cache_resource(show_spinner=False)
def load_cached_embeddings(model_name: str):
    return get_embeddings(model_name)

@st.cache_data(show_spinner=False)
def fetch_cached_transcript(video_id: str):
    return get_transcript(video_id)

def get_tts_lang_code(text: str, selected_lang: str) -> str:
    lang_map = {
        "English": "en",
        "Hindi (हिन्दी)": "hi",
        "Hinglish (Hindi in English script)": "en",
        "Spanish (Español)": "es",
        "French (Français)": "fr",
        "German (Deutsch)": "de",
        "Bengali (বাংলা)": "bn",
        "Telugu (తెలుగు)": "te",
        "Tamil (தமிழ்)": "ta",
        "Marathi (मराठी)": "mr",
        "Gujarati (ગુજરાતી)": "gu",
        "Urdu (اردو)": "ur",
        "Arabic (العربية)": "ar",
    }
    if selected_lang in lang_map and selected_lang != "Auto (Match question language)":
        return lang_map[selected_lang]
    if re.search(r"[\u0900-\u097F]", text):
        return "hi"
    if re.search(r"[\u0980-\u09FF]", text):
        return "bn"
    if re.search(r"[\u0B80-\u0BFF]", text):
        return "ta"
    if re.search(r"[\u0C00-\u0C7F]", text):
        return "te"
    if re.search(r"[\u0A80-\u0AFF]", text):
        return "gu"
    if re.search(r"[\u0600-\u06FF]", text):
        return "ar"
    return "en"

st.set_page_config(
    page_title="YT-RAG | Video Knowledge Assistant",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      .block-container {max-width: 1120px; padding-top: 2.3rem; padding-bottom: 3rem;}
      .hero {padding: 1.5rem 1.7rem; border: 1px solid rgba(120,120,120,.22);
             border-radius: 18px; background: linear-gradient(130deg, rgba(99,102,241,.12), rgba(20,184,166,.08));}
      .muted {color: #777; font-size: .94rem;}
      a {text-decoration: none;}
    </style>
    """,
    unsafe_allow_html=True,
)

if "video" not in st.session_state:
    st.session_state.video = None
if "messages" not in st.session_state:
    st.session_state.messages = []


def configured_secret() -> str:
    value = os.getenv("GROQ_API_KEY", "")
    try:
        value = st.secrets.get("GROQ_API_KEY", value)
    except Exception:
        pass
    return str(value).strip()


with st.sidebar:
    st.title("⚙️ Settings")
    api_key = st.text_input(
        "Groq API key",
        value=configured_secret(),
        type="password",
        help="Used only to generate answers. Set GROQ_API_KEY in your environment or Streamlit secrets for deployment.",
    )
    llm_model = st.text_input("Groq model", value=DEFAULT_LLM_MODEL)
    embedding_model = st.text_input(
        "Embedding model", value=DEFAULT_EMBEDDING_MODEL,
        help="Hugging Face sentence-transformer. The model downloads on first use.",
    )
    st.divider()
    st.caption("Your video transcript and vector index stay in this app session. The Groq API key is not saved by this app.")
    st.caption("Training project · AI / NLP · Retrieval Augmented Generation")

st.markdown(
    """
    <div class="hero">
      <div class="muted">INDUSTRIAL TRAINING PROJECT · 45 DAYS</div>
      <h1 style="margin:.35rem 0">🎓 YT-RAG</h1>
      <div style="font-size:1.12rem">YouTube Video Knowledge Assistant</div>
      <p class="muted" style="margin-bottom:0">Ask questions about a video and get grounded answers with timestamped sources.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.write("")
if "video_url_input" not in st.session_state:
    st.session_state["video_url_input"] = "https://www.youtube.com/watch?v=aircAruvnKk"

def set_sample_video(url: str):
    st.session_state["video_url_input"] = url

st.subheader("1. Add a YouTube video")
st.caption("💡 Quick test — click any verified sample video below to load its link:")
col_s1, col_s2, col_s3 = st.columns(3)
with col_s1:
    st.button(
        "🧠 Neural Networks (3Blue1Brown)",
        use_container_width=True,
        on_click=set_sample_video,
        args=("https://www.youtube.com/watch?v=aircAruvnKk",),
    )
with col_s2:
    st.button(
        "🚀 Python in 100s (Fireship)",
        use_container_width=True,
        on_click=set_sample_video,
        args=("https://www.youtube.com/watch?v=x7X9w_GIm1s",),
    )
with col_s3:
    st.button(
        "🤖 What is RAG? (IBM Tech)",
        use_container_width=True,
        on_click=set_sample_video,
        args=("https://www.youtube.com/watch?v=T-D1OfcDW1M",),
    )

with st.form("video_form", clear_on_submit=False):
    video_url = st.text_input(
        "YouTube URL",
        key="video_url_input",
        placeholder="https://www.youtube.com/watch?v=…",
        help="Public videos with available captions/transcripts are supported.",
    )
    # Thumbnail preview of the currently selected video link
    try:
        if video_url and video_url.strip():
            prev_id = parse_video_id(video_url.strip())
            st.image(f"https://img.youtube.com/vi/{prev_id}/hqdefault.jpg", width=260, caption=f"Video Preview: {prev_id}")
    except Exception:
        pass

    index_clicked = st.form_submit_button("Fetch transcript & build knowledge index", type="primary", use_container_width=True)

if index_clicked:
    try:
        video_id = parse_video_id(video_url)
        with st.status("Preparing your video knowledge base…", expanded=True) as status:
            st.write("Fetching the available transcript…")
            transcript = fetch_cached_transcript(video_id)
            total_chars = sum(len(item.get("text", "")) for item in transcript)
            st.write(f"Transcript loaded ({len(transcript):,} captions, ~{total_chars:,} characters). Creating semantic chunks…")
            embeddings = load_cached_embeddings(embedding_model.strip() or DEFAULT_EMBEDDING_MODEL)
            st.write("Generating vector embeddings and building FAISS index…")
            vectorstore, chunk_count = build_index(transcript, video_id, embeddings)
            st.session_state.video = {
                "video_id": video_id,
                "url": youtube_url(video_id),
                "vectorstore": vectorstore,
                "chunk_count": chunk_count,
                "transcript_chars": len(transcript),
                "embedding_model": embedding_model.strip() or DEFAULT_EMBEDDING_MODEL,
            }
            st.session_state.messages = []
            status.update(label="Knowledge base ready", state="complete", expanded=False)
        st.success(f"Indexed {chunk_count} transcript passages. You can now ask questions below.")
    except Exception as exc:
        st.error(str(exc))
        st.info("💡 **Why this happens:** This specific video does not have subtitles/captions enabled on YouTube. Please click one of the verified sample buttons above (e.g. **Neural Networks** or **Python in 100s**), or use any video where the **[CC]** button is active.")

video: dict[str, Any] | None = st.session_state.video

st.subheader("2. Ask the video")

if video:
    thumbnail_url = f"https://img.youtube.com/vi/{video['video_id']}/hqdefault.jpg"
    with st.container():
        col_thumb, col_details = st.columns([1, 2])
        with col_thumb:
            st.image(thumbnail_url, caption="Active Video Thumbnail", use_container_width=True)
        with col_details:
            st.markdown(f"#### 🎬 [{video['video_id']}]({video['url']})")
            st.write(f"📚 **Indexed Passages:** `{video['chunk_count']}` chunks")
            st.write(f"📝 **Transcript Length:** ~`{video.get('transcript_chars', 0):,}` captions")
            with st.expander("▶️ Watch Video Player in App"):
                st.video(video["url"])
    st.divider()

col_lang, col_tts = st.columns([3, 2])
with col_lang:
    selected_language = st.selectbox(
        "🌐 Answer Language",
        options=[
            "Auto (Match question language)",
            "English",
            "Hindi (हिन्दी)",
            "Hinglish (Hindi in English script)",
            "Spanish (Español)",
            "French (Français)",
            "German (Deutsch)",
            "Bengali (বাংলা)",
            "Telugu (తెలుగు)",
            "Tamil (தமிழ்)",
            "Marathi (मराठी)",
            "Gujarati (ગુજરાતી)",
            "Urdu (اردو)",
            "Arabic (العربية)",
        ],
        index=0,
        help="Select a specific language, or keep Auto to respond in whatever language you speak or type.",
    )
with col_tts:
    st.write("")
    enable_voice = st.checkbox(
        "🔊 Read answer aloud (Audio)",
        value=True,
        help="Generate spoken audio of the answer in the selected language.",
    )

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("audio_bytes"):
            st.audio(message["audio_bytes"], format="audio/mp3")
        if message.get("sources"):
            with st.expander("Sources from the transcript"):
                for idx, source in enumerate(message["sources"], start=1):
                    timestamp = source["timestamp"]
                    st.markdown(f"**[{idx}] [Jump to {timestamp}]({source['url']})**")
                    st.caption(source["text"])

tab_text, tab_mic = st.tabs(["✍️ Type Question", "🎙️ Speak Question (Microphone)"])

with tab_text:
    with st.form("ask_form", clear_on_submit=False):
        col_q, col_btn = st.columns([5, 1])
        with col_q:
            form_question = st.text_input(
                "Question input",
                placeholder="Type your question in any language (e.g. What is gradient descent? / न्यूरल नेटवर्क क्या है?)",
                label_visibility="collapsed",
            )
        with col_btn:
            ask_submitted = st.form_submit_button("Ask ➔", type="primary", use_container_width=True)

voice_question = None
with tab_mic:
    st.caption("Click the red record button on the microphone below to speak in your language:")
    recorded_audio = st.audio_input("Record voice question", label_visibility="collapsed")
    if recorded_audio:
        audio_data = recorded_audio.getvalue()
        audio_hash = hash(audio_data)
        if st.session_state.get("last_audio_hash") != audio_hash:
            if not api_key:
                st.error("Groq API key is required to transcribe audio with Whisper.")
            else:
                with st.spinner("🎙️ Transcribing your speech with Groq Whisper..."):
                    try:
                        client_whisper = Groq(api_key=api_key)
                        transcription = client_whisper.audio.transcriptions.create(
                            file=("question.wav", io.BytesIO(audio_data)),
                            model="whisper-large-v3-turbo",
                        )
                        transcribed_text = transcription.text.strip()
                        st.session_state["last_audio_hash"] = audio_hash
                        st.session_state["last_transcribed_text"] = transcribed_text
                    except Exception as exc:
                        st.error(f"Speech transcription failed: {exc}")

        current_transcribed = st.session_state.get("last_transcribed_text", "")
        if current_transcribed:
            st.success(f"🗣️ **Recognized Question:** *\"{current_transcribed}\"*")
            if st.button("Submit This Spoken Question ➔", type="primary", use_container_width=True):
                voice_question = current_transcribed

chat_question = st.chat_input("Ask something covered in the video in any language…")
question = (form_question.strip() if ask_submitted and form_question.strip() else None) or voice_question or chat_question

if question:
    if not video:
        st.warning("⚠️ **Please complete Step 1 first:** Click **'Fetch transcript & build knowledge index'** above so the assistant has the transcript to answer your question!")
    elif not api_key:
        st.error("🔑 **Groq API Key Required:** Please add your Groq API key in the left sidebar `>>` (or set `GROQ_API_KEY` in your environment) to generate an answer.")
    else:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("assistant"):
            try:
                client = Groq(api_key=api_key)
                with st.spinner("Searching transcript and formulating answer in your language…"):
                    # Translate question to English keywords for FAISS retrieval if non-English
                    search_query = question
                    try:
                        trans_res = client.chat.completions.create(
                            model=llm_model.strip() or DEFAULT_LLM_MODEL,
                            temperature=0.0,
                            max_tokens=60,
                            messages=[
                                {
                                    "role": "system",
                                    "content": "Translate the user query into concise English search keywords for semantic search against English video transcripts. Return ONLY the English query without extra explanation.",
                                },
                                {"role": "user", "content": question},
                            ],
                        )
                        search_query = trans_res.choices[0].message.content.strip() or question
                    except Exception:
                        search_query = question

                    sources = retrieve_sources(video["vectorstore"], search_query, video["video_id"], k=5)
                    context = "\n\n".join(
                        f"[Source {i}] ({item['timestamp']}) {item['text']}"
                        for i, item in enumerate(sources, start=1)
                    )

                    lang_directive = (
                        f"You MUST answer strictly in {selected_language}."
                        if selected_language != "Auto (Match question language)"
                        else "Detect the language of the user's question, and answer in that EXACT SAME language (for example: Hindi if the user asks in Hindi, English if in English, Spanish if in Spanish, etc.)."
                    )

                    system_prompt = (
                        "You are YT-RAG, an intelligent video assistant. Answer the user's question accurately using ONLY "
                        "the provided YouTube transcript excerpts.\n"
                        "- If the excerpts do not contain enough information to answer, state clearly in the user's language that the answer is not in this video's transcript.\n"
                        "- Cite evidence by placing source markers like [1], [2] next to facts.\n"
                        f"- LANGUAGE REQUIREMENT: {lang_directive} Even though the transcript excerpts are in English, explain and translate everything fluently and accurately in the target language."
                    )

                    completion = client.chat.completions.create(
                        model=llm_model.strip() or DEFAULT_LLM_MODEL,
                        temperature=0.1,
                        max_tokens=900,
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {
                                "role": "user",
                                "content": f"Transcript excerpts:\n{context}\n\nQuestion: {question}",
                            },
                        ],
                    )
                    answer = completion.choices[0].message.content or "I could not generate an answer."
                    st.markdown(answer)

                    # Optional audio generation in user's language
                    audio_out_bytes = None
                    if enable_voice:
                        try:
                            clean_spoken = re.sub(r"\[\d+\]", "", answer).strip()[:500]
                            tts_lang = get_tts_lang_code(clean_spoken, selected_language)
                            tts = gTTS(text=clean_spoken, lang=tts_lang)
                            fp = io.BytesIO()
                            tts.write_to_fp(fp)
                            audio_out_bytes = fp.getvalue()
                            st.audio(audio_out_bytes, format="audio/mp3")
                        except Exception:
                            pass

                    if sources:
                        with st.expander("Sources from the transcript"):
                            for idx, source in enumerate(sources, start=1):
                                st.markdown(f"**[{idx}] [Jump to {source['timestamp']}]({source['url']})**")
                                st.caption(source["text"])
                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": answer,
                            "sources": sources,
                            "audio_bytes": audio_out_bytes,
                        }
                    )
            except Exception as exc:
                st.error(f"Could not answer this question: {exc}")

if video:
    with st.expander("How this answer is produced"):
        st.markdown(
            "1. Captions are fetched from YouTube.  2. Transcript passages are split and embedded with a Hugging Face model.  "
            "3. FAISS retrieves the most relevant passages.  4. Groq generates an answer grounded in those passages."
        )

st.divider()
st.caption("YT-RAG · Python · LangChain text splitting / FAISS · Hugging Face embeddings · Groq API")
