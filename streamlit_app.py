import streamlit as st
import librosa
from model import predict

st.title("🎤 AI Voice Detection Demo")
st.markdown("Upload a **WAV or MP3** file to detect whether the voice is **Human** or **AI-generated**.")

uploaded_file = st.file_uploader("Choose an audio file", type=["wav", "mp3"])

if uploaded_file is not None:
    # Save uploaded file temporarily
    with open("temp.wav", "wb") as f:
        f.write(uploaded_file.read())

    # Load audio with librosa
    audio, sr = librosa.load("temp.wav", sr=16000, mono=True)

    # Make prediction
    result, confidence = predict(audio)

    st.write(f"**Result:** {result}")
    st.write(f"**Confidence:** {confidence}")
