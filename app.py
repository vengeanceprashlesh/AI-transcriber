import streamlit as st
import whisper
import tempfile
import os
from transformers import pipeline

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="AI-Powered Audio Transcriber & Summarizer",
    page_icon="🎤",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# --- DARK THEME & CLEAN CSS (NO BOXES) ---
st.markdown("""
    <style>
    html, body, [class*="css"]  {
        background-color: #181c23 !important;
        color: #f5f6fa !important;
    }
    .main {
        background-color: #181c23 !important;
    }
    .block-container {
        max-width: 480px;
        margin: auto;
        padding-top: 7vh;
        min-height: 90vh;
        display: flex;
        flex-direction: column;
        justify-content: flex-start;
        align-items: center;
    }
    .big-title {
        font-size: 2.3rem !important;
        font-weight: 900 !important;
        margin-bottom: 0.7rem !important;
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 0.7rem;
        letter-spacing: -1px;
        text-align: center;
    }
    .subtitle {
        color: #b2becd !important;
        font-size: 1.08rem;
        text-align: center;
        margin-bottom: 2.2rem;
        font-weight: 400;
        line-height: 1.5;
    }
    .stFileUploader > label {
        color: #f5f6fa !important;
        font-weight: 600;
        font-size: 1.05rem;
    }
    .stFileUploader .css-1c7y2kd {
        background: transparent !important;
        border-radius: 0 !important;
        border: 1.5px dashed #3a3f4b !important;
        color: #b2becd !important;
    }
    .stButton > button {
        background: linear-gradient(90deg, #3b82f6 0%, #6366f1 100%) !important;
        color: #fff !important;
        border: none;
        border-radius: 10px;
        padding: 0.7rem 2rem;
        font-size: 1.05rem;
        font-weight: 700;
        margin-top: 1.1rem;
        margin-bottom: 0.5rem;
        width: 100%;
        max-width: 320px;
        display: block;
        margin-left: auto;
        margin-right: auto;
    }
    .stButton > button:hover {
        filter: brightness(1.1);
        transform: translateY(-2px) scale(1.03);
    }
    .stTextArea textarea {
        background: #232733 !important;
        color: #f5f6fa !important;
        border-radius: 8px;
        border: 1.5px solid #3a3f4b;
        font-size: 1.05rem;
        min-height: 110px;
        line-height: 1.6;
        padding: 1rem;
    }
    .stAudio audio {
        background: #232733 !important;
        border-radius: 8px;
    }
    .stSuccess {
        background: #232733 !important;
        color: #a3e635 !important;
        border-radius: 8px;
        border-left: 5px solid #a3e635;
        font-size: 1.05rem;
        margin-top: 1.1rem;
        margin-bottom: 1.1rem;
        padding: 1rem 0.9rem;
    }
    .formats {
        color: #fff;
        font-size: 1.1rem;
        font-weight: 700;
        margin-top: 2.2rem;
        text-align: center;
        letter-spacing: 0.5px;
        background: #232733;
        border-radius: 10px;
        padding: 0.9rem 0.5rem 0.9rem 0.5rem;
        width: 100%;
        max-width: 340px;
        margin-left: auto;
        margin-right: auto;
    }
    </style>
    """, unsafe_allow_html=True)

# --- LOAD MODELS ---
@st.cache_resource
def load_models():
    whisper_model = whisper.load_model("tiny")
    summarizer = pipeline("summarization", model="facebook/bart-base")
    return whisper_model, summarizer

# --- HEADER ---
st.markdown('<div class="big-title">🎤 AI-Powered Audio Transcriber & Summarizer</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Upload an audio file and get instant transcription & summary.</div>', unsafe_allow_html=True)

# --- UPLOAD AREA ---
uploaded_file = st.file_uploader(
    "Upload your audio file",
    type=["mp3", "wav", "m4a"],
    help="MP3, WAV, M4A | Max 200MB"
)

# --- MAIN LOGIC ---
if uploaded_file is not None:
    st.audio(uploaded_file, format="audio/wav")
    if st.button("Transcribe & Summarize", use_container_width=True):
        with st.spinner("Processing your audio..."):
            try:
                whisper_model, summarizer = load_models()
                with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_file_path = tmp_file.name
                result = whisper_model.transcribe(tmp_file_path)
                transcription = result["text"].strip()
                st.markdown("### 📝 Transcription", unsafe_allow_html=True)
                st.text_area("", transcription, height=200)
                if len(transcription.split()) > 20:
                    summary = summarizer(transcription, max_length=130, min_length=30, do_sample=False)[0]["summary_text"]
                    st.markdown("### ✨ Summary", unsafe_allow_html=True)
                    st.success(summary)
                col1, col2 = st.columns(2)
                with col1:
                    st.download_button(
                        "Download Transcription",
                        transcription,
                        file_name="transcription.txt",
                        mime="text/plain"
                    )
                with col2:
                    if len(transcription.split()) > 20:
                        st.download_button(
                            "Download Summary",
                            summary,
                            file_name="summary.txt",
                            mime="text/plain"
                        )
                os.remove(tmp_file_path)
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
                st.info("Please try again with a different audio file.")
else:
    st.markdown('<div class="formats">Supported formats: MP3, WAV, M4A</div>', unsafe_allow_html=True)
