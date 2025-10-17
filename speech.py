import streamlit as st
from gtts import gTTS
from textblob import TextBlob
from PIL import Image
import speech_recognition as sr
import os
from transformers import pipeline
import plotly.graph_objects as go

# ---------------------------------------------------
# PAGE CONFIGURATION
# ---------------------------------------------------
st.set_page_config(layout="wide", page_title="🧠 AI-Powered Unstructured Data Analyzer")

st.markdown("""
    <style>
        [data-testid="stAppViewContainer"] {
            background: linear-gradient(120deg, #a1c4fd, #c2e9fb);
        }
        h1, h2, h3, h4 {
            color: #0a3d62;
        }
        .stButton>button {
            background-color: #1e3799;
            color: white;
            border-radius: 10px;
            transition: all 0.3s ease;
        }
        .stButton>button:hover {
            background-color: #4a69bd;
            color: #fff;
            transform: scale(1.03);
        }
        .stTextInput>div>div>input, textarea {
            border-radius: 10px;
        }
    </style>
""", unsafe_allow_html=True)

st.title("🧠 AI-Powered Unstructured Data Analyzer")
st.markdown("Analyze **images**, **audio**, and **text** with AI-based insights — all in one place!")

# ---------------------------------------------------
# TABS SETUP
# ---------------------------------------------------
tab1, tab2, tab3 = st.tabs(["🖼 Image Analysis", "🎧 Audio Analysis", "📝 Text Analysis"])

# ---------------------------------------------------
# 🖼 TAB 1: IMAGE ANALYSIS
# ---------------------------------------------------
with tab1:
    st.header("🖼 Image Analysis")
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Display Image Details
        st.write(f"**Format:** {image.format}")
        st.write(f"**Size:** {image.size}")
        st.info("🧠 Generating AI-based caption...")

        try:
            # Image Caption Generation (Transformer)
            caption_pipeline = pipeline("image-to-text", model="nlpconnect/vit-gpt2-image-captioning")
            caption = caption_pipeline(image)[0]['generated_text']
            st.success(f"📝 AI Caption: *{caption}*")
        except Exception as e:
            st.error(f"⚠️ Caption generation failed: {e}")

# ---------------------------------------------------
# 🎧 TAB 2: AUDIO ANALYSIS
# ---------------------------------------------------
with tab2:
    st.header("🎧 Audio Analysis")

    # --- Text to Speech ---
    st.subheader("🗣 Text → Speech")
    text_input = st.text_area("Enter text to convert to audio:")

    if st.button("🔊 Convert to Audio"):
        if text_input.strip():
            tts = gTTS(text_input, lang='en')
            tts.save("tts_output.mp3")
            st.audio("tts_output.mp3")
            st.success("✅ Text converted to speech successfully!")
        else:
            st.warning("Please enter some text first.")

    st.markdown("---")

    # --- Speech to Text ---
    st.subheader("🎤 Speech → Text")
    audio_upload = st.file_uploader("Upload an audio file (wav/mp3):", type=["wav", "mp3"])

    if audio_upload and st.button("🧾 Convert Audio to Text"):
        recognizer = sr.Recognizer()
        try:
            with sr.AudioFile(audio_upload) as source:
                st.info("🎧 Listening to the audio...")
                audio_data = recognizer.record(source)
                text_output = recognizer.recognize_google(audio_data)
                st.success("✅ Transcription Complete!")
                st.write("### 🗒 Transcribed Text:")
                st.info(text_output)
        except Exception as e:
            st.error(f"❌ Error during transcription: {e}")

# ---------------------------------------------------
# 📝 TAB 3: TEXT ANALYSIS
# ---------------------------------------------------
with tab3:
    st.header("📝 Text Sentiment & Emotion Analysis")

    input_text = st.text_area("Enter text to analyze sentiment and emotions:")

    if st.button("🔍 Analyze Text"):
        if input_text.strip():
            blob = TextBlob(input_text)
            polarity = blob.sentiment.polarity
            subjectivity = blob.sentiment.subjectivity

            # Determine sentiment label
            if polarity > 0:
                sentiment = "Positive 😀"
            elif polarity < 0:
                sentiment = "Negative 😞"
            else:
                sentiment = "Neutral 😐"

            st.write(f"**Sentiment:** {sentiment}")
            st.write(f"**Polarity (−1 to +1):** {round(polarity, 2)}")
            st.write(f"**Subjectivity (0=Objective → 1=Subjective):** {round(subjectivity, 2)}")

            # Emotion Detection using Transformers
            try:
                emotion_pipeline = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base")
                emotion = emotion_pipeline(input_text)[0]
                st.success(f"🧩 Dominant Emotion: {emotion['label']} ({round(emotion['score']*100, 2)}%)")
            except Exception as e:
                st.warning(f"⚠️ Emotion detection unavailable: {e}")

            # Visualization
            sentiment_value = (polarity + 1) * 50
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=sentiment_value,
                title={'text': "Sentiment (%)"},
                gauge={'axis': {'range': [0, 100]},
                       'bar': {'color': "#1e3799"},
                       'steps': [
                           {'range': [0, 33], 'color': "#ff6b6b"},
                           {'range': [33, 66], 'color': "#feca57"},
                           {'range': [66, 100], 'color': "#1dd1a1"}
                       ]}
            ))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Please enter some text before analysis.")
