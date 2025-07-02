import streamlit as st
import numpy as np
import librosa
import joblib
import matplotlib.pyplot as plt
import soundfile as sf

# Load saved models
model = joblib.load("ser_mlp_model.pkl")
encoder = joblib.load("label_encoder.pkl")

st.title("🎙️ Speech Emotion Recognition")
st.write("Upload an audio file (.wav) to detect the emotion.")

uploaded_file = st.file_uploader("Choose a WAV file", type="wav")

def extract_feature(file, mfcc=True, chroma=True, mel=True):
    # Load only first 3 seconds to keep it consistent with training
    X, sample_rate = librosa.load(file, sr=None, duration=3.0)

    # 🔒 Check for silence or extremely low energy
    if np.max(np.abs(X)) < 0.01:
        st.error("❌ Audio is too quiet or silent. Please upload clear speech.")
        st.stop()

    result = np.array([])

    if chroma or mel:
        stft = np.abs(librosa.stft(X))

    if mfcc:
        mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
        result = np.hstack((result, mfccs))
    if chroma:
        chroma_feat = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
        result = np.hstack((result, chroma_feat))
    if mel:
        mel_feat = np.mean(librosa.feature.melspectrogram(y=X, sr=sample_rate).T, axis=0)
        result = np.hstack((result, mel_feat))

    return result.reshape(1, -1), X  # return waveform too

if uploaded_file is not None:
    st.audio(uploaded_file, format="audio/wav")

    # Extract features and raw waveform
    features, waveform = extract_feature(uploaded_file)

    # 🔍 Show waveform
    fig, ax = plt.subplots()
    ax.plot(waveform)
    ax.set_title("Audio Waveform (First 3 seconds)")
    ax.set_xlabel("Samples")
    ax.set_ylabel("Amplitude")
    st.pyplot(fig)

    # Predict emotion
    prediction = model.predict(features)
    emotion = encoder.inverse_transform(prediction)[0]
    st.success(f"Predicted Emotion: **{emotion}**")

