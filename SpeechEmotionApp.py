import streamlit as st
import numpy as np
import librosa
import joblib
import soundfile as sf

# Load saved models
model = joblib.load("ser_mlp_model.pkl")
encoder = joblib.load("label_encoder.pkl")

st.title("🎙️ Speech Emotion Recognition")
st.write("Upload an audio file (.wav) to detect the emotion.")

uploaded_file = st.file_uploader("Choose a WAV file", type="wav")

def extract_feature(file, mfcc=True, chroma=True, mel=True):
    with sf.SoundFile(file) as sound_file:
        X = sound_file.read(dtype="float32")
        sample_rate = sound_file.samplerate

        if chroma:
            stft = np.abs(librosa.stft(X))

        result = np.array([])
        if mfcc:
            mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
            result = np.hstack((result, mfccs))
        if chroma:
            chroma_feat = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
            result = np.hstack((result, chroma_feat))
        if mel:
            mel_feat = np.mean(librosa.feature.melspectrogram(y=X, sr=sample_rate).T, axis=0)
            result = np.hstack((result, mel_feat))

    return result.reshape(1, -1)

if uploaded_file is not None:
    features = extract_feature(uploaded_file)
    prediction = model.predict(features)
    emotion = encoder.inverse_transform(prediction)[0]
    st.success(f"Predicted Emotion: **{emotion}**")
