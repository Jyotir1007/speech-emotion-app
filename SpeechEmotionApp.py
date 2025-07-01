import streamlit as st
import numpy as np
import librosa
import joblib
import matplotlib.pyplot as plt
import soundfile as sf
from streamlit_webrtc import webrtc_streamer, AudioProcessorBase
import av
import queue
import tempfile

# Queue to collect audio data from microphone
audio_queue = queue.Queue()

# Audio processor to capture audio chunks
class AudioProcessor(AudioProcessorBase):
    def __init__(self):
        self.audio_buffer = []

    def recv(self, frame: av.AudioFrame) -> av.AudioFrame:
        audio_data = frame.to_ndarray().flatten().astype(np.float32) / 32768.0
        audio_queue.put(audio_data)
        return frame

st.header("🎤 Or speak now to detect emotion")

# Streamlit WebRTC audio recorder
ctx = webrtc_streamer(
    key="emotion-speech",
    mode="SENDRECV",
    in_audio=True,
    audio_processor_factory=AudioProcessor,
    media_stream_constraints={"audio": True, "video": False},
    async_processing=True,
)

# Button to process speech after recording
if st.button("🔍 Predict Emotion from Microphone"):
    if audio_queue.empty():
        st.warning("Please speak first. Nothing was recorded.")
    else:
        # Collect all audio chunks
        audio = []
        while not audio_queue.empty():
            audio.extend(audio_queue.get())

        # Save to temporary WAV file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            sf.write(f.name, np.array(audio), 16000)  # Assuming 16kHz sample rate
            st.success("Audio captured successfully.")
            st.audio(f.name)

            # Extract features and predict
            features, waveform = extract_feature(f.name)
            fig, ax = plt.subplots()
            ax.plot(waveform)
            ax.set_title("Audio Waveform")
            st.pyplot(fig)

            prediction = model.predict(features)
            emotion = encoder.inverse_transform(prediction)[0]
            st.success(f"Predicted Emotion: **{emotion}**")


