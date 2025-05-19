import os
import re
import streamlit as st
import whisper
import numpy as np
import librosa
import plotly.graph_objects as go
import sys
import torch

# --- FFmpeg Setup ---
ffmpeg_path = os.path.join(os.getcwd(), "ffmpeg", "bin", "ffmpeg.exe")
os.environ["PATH"] += os.pathsep + os.path.dirname(ffmpeg_path)

# --- Prevent Streamlit from trying to "watch" torch internals ---
if 'torch' in sys.modules:
    torch.__path__ = []

# --- Load Whisper Model ---
model = whisper.load_model("base")

# --- Helper Functions ---

def transcribe_audio(audio_path, model):
    return model.transcribe(audio_path, word_timestamps=True)

def analyze_fluency(result):
    timestamps = [word["start"] for word in result["segments"][0]["words"]]
    if len(timestamps) < 2:
        return 0, 0
    pauses = [j - i for i, j in zip(timestamps[:-1], timestamps[1:])]
    return np.mean(pauses), np.std(pauses)

def analyze_pitch(audio_path):
    y, sr = librosa.load(audio_path)
    pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
    pitch_values = pitches[magnitudes > np.median(magnitudes)]
    return np.std(pitch_values) if len(pitch_values) > 0 else 0

def analyze_sentence_complexity(text):
    sentences = re.split(r'[.!?]', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    lengths = [len(sentence.split()) for sentence in sentences]
    return (np.mean(lengths), np.std(lengths)) if lengths else (0, 0)

# --- Streamlit App ---

st.title("VODE Speech Analysis")
audio_file = st.file_uploader("Upload an audio file", type=["mp3", "wav", "m4a"])

if audio_file:
    with open("temp_audio.mp3", "wb") as f:
        f.write(audio_file.read())

    st.audio("temp_audio.mp3")

    progress = st.progress(0, text="Initializing...")

    try:
        progress.progress(10, text="Transcribing audio...")
        result = transcribe_audio("temp_audio.mp3", model)

        progress.progress(40, text="Analyzing fluency...")
        pause_avg, pause_std = analyze_fluency(result)

        progress.progress(60, text="Analyzing pitch...")
        pitch_std = analyze_pitch("temp_audio.mp3")

        progress.progress(80, text="Analyzing sentence complexity...")
        sentence_avg, sentence_std = analyze_sentence_complexity(result["text"])

        progress.progress(100, text="Complete ✅")

        # --- Plotting ---
        st.subheader("📊 Speech Pattern Metrics")
        metrics = ["Pause Duration", "Pitch Variation", "Sentence Complexity"]
        values = [pause_avg, pitch_std, sentence_avg]

        fig = go.Figure([go.Bar(x=metrics, y=values, marker_color=["#FFB703", "#219EBC", "#8ECAE6"])])
        fig.update_layout(title="Speech Pattern Analysis", yaxis_title="Value", height=400)
        st.plotly_chart(fig)

        # --- Prediction ---
        st.subheader("🧠 Speaking Style Prediction")
        is_natural = pause_avg < 0.6 and pitch_std > 30 and sentence_avg > 5

        if is_natural:
            st.success("The speaker appears to be speaking naturally. 🎤")
        else:
            st.warning("The speaker might be reading from a script. 📖")

        # --- Transcript ---
        st.subheader("📝 Transcript")
        st.write(result["text"])

    except Exception as e:
        st.error(f"Something went wrong: {e}")
