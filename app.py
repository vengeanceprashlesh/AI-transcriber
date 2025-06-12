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

st.title("🎙️ AI-Powered Audio Transcriber & Summarizer")
st.markdown("Upload an audio file and get instant transcription & summary.")

uploaded_file = st.file_uploader("📁 Upload your audio file", type=["mp3", "wav"])

if uploaded_file is not None:
    st.success("✅ File uploaded successfully!")
    
    with st.spinner("Transcribing audio using Whisper..."):
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
                tmp.write(uploaded_file.read())
                tmp_path = tmp.name

            model = whisper.load_model("base")
            result = model.transcribe(tmp_path)
            transcription = result["text"]

            st.markdown("### 📜 Transcription:")
            st.text_area("Transcription", transcription, height=200)

            with st.spinner("Summarizing transcription..."):
                summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
                summary = summarizer(transcription, max_length=130, min_length=30, do_sample=False)[0]["summary_text"]

                st.markdown("### ✨ Summary:")
                st.success(summary)

            # 🎯 Add Download Buttons
            transcription_filename = uploaded_file.name.replace(".", "_") + "_transcription.txt"
            summary_filename = uploaded_file.name.replace(".", "_") + "_summary.txt"

            st.download_button("⬇️ Download Transcription", transcription, file_name=transcription_filename)
            st.download_button("⬇️ Download Summary", summary, file_name=summary_filename)

            os.remove(tmp_path)
        except Exception as e:
            st.error(f"❌ Error: {e}")
else:
    st.info("Upload a file above to begin.")
