import streamlit as st
import cv2
import numpy as np

st.title("🖼️ Face Detection App")

# --- Upload Image ---
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read uploaded image as numpy array
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)  # OpenCV reads in BGR format

    # Convert to grayscale for face detection
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Load Haar cascade for face detection
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Draw rectangles around faces
    output_image = image.copy()
    for (x, y, w, h) in faces:
        cv2.rectangle(output_image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Convert images from BGR to RGB for displaying in Streamlit
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    output_rgb = cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB)

    # Display original and output images side by side
    st.subheader("Original vs Face Detection")
    col1, col2 = st.columns(2)
    with col1:
        st.image(image_rgb, caption="Original Image", use_column_width=True)
    with col2:
        st.image(output_rgb, caption=f"Detected {len(faces)} face(s)", use_column_width=True)
