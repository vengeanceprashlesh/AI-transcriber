import streamlit as st
import whisper
import tempfile
import os
from transformers import pipeline

st.set_page_config(
    page_title="AI Transcriber & Summarizer",
    page_icon="🎧",
    layout="centered",
)

# Cache models to avoid reloading
@st.cache_resource
def load_whisper_model():
    return whisper.load_model("tiny")  # Using tiny model for faster loading

@st.cache_resource
def load_summarizer():
    return pipeline("summarization", model="facebook/bart-base")  # Using base model

st.title("🎙️ AI-Powered Audio Transcriber & Summarizer")
st.markdown("Upload an audio file and get instant transcription & summary.")

uploaded_file = st.file_uploader("📁 Upload your audio file", type=["mp3", "wav", "m4a"])

if uploaded_file is not None:
    st.success("✅ File uploaded successfully!")
    
    with st.spinner("Transcribing audio using Whisper..."):
        try:
            # Save uploaded file to temporary location
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
                tmp.write(uploaded_file.read())
                tmp_path = tmp.name

            # Load model and transcribe
            model = load_whisper_model()
            result = model.transcribe(tmp_path)
            transcription = result["text"]

            st.markdown("### 📜 Transcription:")
            st.text_area("Transcription", transcription, height=200)

            # Only summarize if transcription is long enough
            if len(transcription.split()) > 10:
                with st.spinner("Summarizing transcription..."):
                    summarizer = load_summarizer()
                    # Truncate text if too long for the model
                    max_length = 1024
                    if len(transcription) > max_length:
                        transcription_truncated = transcription[:max_length]
                    else:
                        transcription_truncated = transcription
                    
                    summary = summarizer(transcription_truncated, 
                                       max_length=130, 
                                       min_length=30, 
                                       do_sample=False)[0]["summary_text"]

                    st.markdown("### ✨ Summary:")
                    st.success(summary)
            else:
                st.info("Transcription too short for summarization.")
                summary = ""

            # Download buttons
            transcription_filename = uploaded_file.name.replace(".", "_") + "_transcription.txt"
            summary_filename = uploaded_file.name.replace(".", "_") + "_summary.txt"

            st.download_button("⬇️ Download Transcription", transcription, file_name=transcription_filename)
            if summary:
                st.download_button("⬇️ Download Summary", summary, file_name=summary_filename)

            # Clean up temporary file
            os.remove(tmp_path)
            
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
            st.info("Try uploading a different audio file or check the file format.")
else:
    st.info("Upload a file above to begin.")
    st.markdown("### Supported formats: MP3, WAV, M4A")
