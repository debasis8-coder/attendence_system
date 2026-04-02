import streamlit as st
import cv2
import os
import numpy as np
import time
from facenet_pytorch import MTCNN
from check_d import Emb_vec

# ---------------- CONFIG ----------------
st.set_page_config(page_title="AI Attendance System", layout="centered")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATASET_PATH = os.path.join(BASE_DIR, "face_dataset")
TEST_IMG_PATH = os.path.join(BASE_DIR, "test_face", "test.jpg")
CHECK_FILE = os.path.join(BASE_DIR, "check_face.txt")

os.makedirs(DATASET_PATH, exist_ok=True)
os.makedirs(os.path.dirname(TEST_IMG_PATH), exist_ok=True)

# ---------------- SESSION STATE ----------------
if "unknown_face" not in st.session_state:
    st.session_state.unknown_face = None

# ---------------- TITLE ----------------
st.title("🎯 Smart AI Attendence System")

# ---------------- CAMERA INPUT ----------------
st.subheader("📷 Capture Face")
img_file = st.camera_input("Take a picture")

if img_file is not None:
    # Convert uploaded file to OpenCV image
    file_bytes = np.asarray(bytearray(img_file.read()), dtype=np.uint8)
    frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    st.image(frame, channels="BGR", caption="Captured Image")

    # Save captured image for model
    cv2.imwrite(TEST_IMG_PATH, frame)
    time.sleep(0.1)

    # Run recognition model
    try:
        Emb_vec().check()  # Your face embedding check function
    except Exception as e:
        st.error(f"Model error: {e}")

    # Read result from check file
    if os.path.exists(CHECK_FILE):
        with open(CHECK_FILE, "r") as f:
            data = f.read().strip()

            if data:
                person = data.split(":")[-1]

                if person != "Unknown":
                    st.success(f"✅ Recognized: {person}")
                    st.session_state.unknown_face = None
                else:
                    st.warning("⚠️ Unknown person detected")
                    st.session_state.unknown_face = frame
            else:
                st.warning("⚠️ Unknown person detected")
                st.session_state.unknown_face = frame

# ---------------- REGISTER NEW FACE ----------------
if st.session_state.unknown_face is not None:
    st.subheader("📝 Register New Person")

    name = st.text_input("Enter Name")

    if st.button("Register"):
        if name.strip():
            mtcnn = MTCNN(image_size=160)
            frame_rgb = cv2.cvtColor(st.session_state.unknown_face, cv2.COLOR_BGR2RGB)
            save_path = os.path.join(DATASET_PATH, f"{name}.jpg")
            mtcnn(frame_rgb, save_path)

            st.success(f"{name} registered successfully")

            # Clear unknown face state
            st.session_state.unknown_face = None
        else:
            st.error("⚠️ Please enter a valid name")
