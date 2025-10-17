# 🧠 AI-Powered Unstructured Data Analyzer

## 📌 Project Overview

This project is an advanced **Streamlit-based web application** designed to analyze **unstructured data** such as **Images, Audio, and Text** using various AI and NLP frameworks. It integrates multiple functionalities like face detection, age & gender prediction, image captioning, speech-to-text, text-to-speech, sentiment analysis, emotion detection, NER, and POS tagging.

The application is organized into **three main modules**:

* 🖼 **Image Analysis**
* 🎧 **Audio Analysis**
* 📝 **Text Analysis**

---

## 🗂️ Features Summary

| Module             | Key Functionalities                                                                                           |
| ------------------ | ------------------------------------------------------------------------------------------------------------- |
| **Image Analysis** | Face Detection (DeepFace), Age & Gender Prediction, Emotion Recognition, Background Removal, Image Captioning |
| **Audio Analysis** | Text-to-Speech (gTTS), Speech-to-Text (Google SR)                                                             |
| **Text Analysis**  | Sentiment Analysis (TextBlob), Emotion Detection (Transformers), NER & POS Tagging (spaCy)                    |

---

## 🧰 Technologies & Libraries Used

### 🔹 Backend & Framework

* **Python**
* **Streamlit** (Web Interface)

### 🔹 Image Processing

* **DeepFace** (Face, Age, Gender, Emotion Detection)
* **rembg** (Background Removal)
* **OpenCV & PIL** (Image Handling)
* **Transformers** (`vit-gpt2-image-captioning` for Image Captioning)

### 🔹 Audio Processing

* **gTTS** (Text-to-Speech)
* **SpeechRecognition** (Speech-to-Text)
* **pydub** (Audio Format Conversion)

### 🔹 Text Processing

* **TextBlob** (Sentiment)
* **spaCy** (NER & POS Tagging)
* **Transformers** (`emotion-english-distilroberta-base` for Emotion Detection)
* **Plotly** (Visualization)

---

## 📁 Project Structure (Suggested)

```
📦 unstructured-data-analyzer
│── app.py                 # Main Streamlit application
│── requirements.txt       # Dependencies
│── models/                # Pre-trained models (if any)
│── assets/                # Images, icons, etc.
└── README.md              # Project Documentation
```

---

## 🚀 How to Run the Application

### ✅ Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

### ✅ Step 2: Run the Streamlit App

```bash
streamlit run app.py
```

---

## 🔍 Module-Wise Details

### 🖼 1️⃣ Image Analysis

* Upload an image (`jpg/png`)
* Perform:

  * Face Detection (DeepFace)
  * Age & Gender Prediction
  * Emotion Recognition
  * Background Removal (rembg)
  * AI Caption Generation (ViT-GPT2)

### 🎧 2️⃣ Audio Analysis

* **Text → Speech** using gTTS
* **Speech → Text** using SpeechRecognition and Google API

### 📝 3️⃣ Text Analysis

* Sentiment Analysis (Positive/Negative/Neutral)
* Emotion Prediction via Transformer Model
* Named Entity Recognition (NER) via spaCy
* POS Tagging (Part-of-Speech)

---

## 🖼 Sample Output

* Face Detected ✅
* Predicted Age: *25*
* Predicted Gender: *Male*
* Emotion: *Happy*
* Transcribed Audio: *"Hello, how are you?"*
* Sentiment: *Positive*
* Dominant Emotion: *Joy*
* Entities: *(Apple – ORG)*

---

## 🧪 Future Enhancements

* 🔍 Object Detection (YOLO)
* 🎭 Face Recognition
* 📊 PDF/CSV Export for Reports
* 🌍 Multilingual Support

---

## 🤝 Contribution Guidelines

1. Fork the repo
2. Create a new branch
3. Commit improvements
4. Create a Pull Request 🎉

---

## 🛡️ License

This project is licensed under the **MIT License**.

---

## 📬 Contact / Support

If you need help, feel free to reach out:
📧 Email: *[saimanishchada@gmail.com]*
💬 GitHub Issues / Discussions

---

Thanks for using **Unstructured Data Analyzer**! 🧠✨
