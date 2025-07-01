import streamlit as st
import numpy as np
import librosa
import soundfile as sf
import joblib
import matplotlib.pyplot as plt
from streamlit_webrtc import webrtc_streamer, AudioProcessorBase
import av
import queue
import tempfile

# Load model and label encoder
model = joblib.load("ser_mlp_model.pkl")
encoder = joblib.load("label_encoder.pkl")

# Feature extraction
def extract_feature(file, mfcc=True, chroma=True, mel=True):
    X, sample_rate = librosa.load(file, sr=None, duration=3.0)

    if np.max(np.abs(X)) < 0.01:
        st.error("❌ Audio is too quiet. Please speak louder.")
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

    return result.reshape(1, -1), X


# --------------------
# STREAMLIT INTERFACE
# --------------------
st.title("🎙️ Speech Emotion Recognition")
st.write("Upload an audio file or speak to detect emotion.")

# Upload option
uploaded_file = st.file_uploader("Choose a WAV file", type="wav")
if uploaded_file is not None:
    st.audio(uploaded_file, format="audio/wav")
    features, waveform = extract_feature(uploaded_file)
    fig, ax = plt.subplots()
    ax.plot(waveform)
    ax.set_title("Waveform")
    st.pyplot(fig)

    prediction = model.predict(features)
    emotion = encoder.inverse_transform(prediction)[0]
    st.success(f"Predicted Emotion: **{emotion}**")


# Microphone input (WebRTC)
st.header("🎤 Or speak into your microphone")

audio_queue = queue.Queue()

class AudioProcessor(AudioProcessorBase):
    def recv(self, frame: av.AudioFrame) -> av.AudioFrame:
        audio = frame.to_ndarray().flatten().astype(np.float32) / 32768.0
        audio_queue.put(audio)
        return frame

ctx = webrtc_streamer(
    key="speech-mic",
    mode="SENDRECV",
    audio_processor_factory=AudioProcessor,
    media_stream_constraints={"audio": True, "video": False},
    async_processing=True,
)

if st.button("🔍 Predict from Microphone"):
    if audio_queue.empty():
        st.warning("Please speak first — no audio received.")
    else:
        # Aggregate audio data from queue
        audio_data = []
        while not audio_queue.empty():
            audio_data.extend(audio_queue.get())

        # Save as WAV and process
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            sf.write(f.name, np.array(audio_data), 16000)
            st.audio(f.name)

            features, waveform = extract_feature(f.name)
            fig, ax = plt.subplots()
            ax.plot(waveform)
            ax.set_title("Waveform")
            st.pyplot(fig)

            prediction = model.predict(features)
            emotion = encoder.inverse_transform(prediction)[0]
            st.success(f"Predicted Emotion: **{emotion}**")
