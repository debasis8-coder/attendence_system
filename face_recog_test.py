import streamlit as st
import cv2
import os
import numpy as np
import time
from facenet_pytorch import MTCNN
from check_d import Emb_vec

# ---------------- CONFIG ----------------
st.set_page_config(page_title="Face Attendance System", layout="wide")

DATASET_PATH = "face_dataset/"
TEST_IMG_PATH = "test_face/test.jpg"
CHECK_FILE = "check_face.txt"

os.makedirs(DATASET_PATH, exist_ok=True)
os.makedirs("test_face", exist_ok=True)

# ---------------- HEADER ----------------
st.title("🎯 Face Recognition System")

# ---------------- SIDEBAR ----------------
st.sidebar.header("Controls")
name = st.sidebar.text_input("Enter Name")

# ---------------- REGISTER ----------------
st.sidebar.subheader("Register Face")
register_img = st.sidebar.camera_input("Capture for Registration")

if register_img is not None and name.strip():
    file_bytes = np.asarray(bytearray(register_img.read()), dtype=np.uint8)
    frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    mtcnn = MTCNN(image_size=160)

    save_path = os.path.join(DATASET_PATH, f"{name}.jpg")
    mtcnn(frame_rgb, save_path)

    st.sidebar.success(f"{name} registered successfully")

# ---------------- RECOGNITION ----------------
st.subheader("Recognize Face")

img_file = st.camera_input("Take a picture")

if img_file is not None:
    # Convert to OpenCV image
    file_bytes = np.asarray(bytearray(img_file.read()), dtype=np.uint8)
    frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    st.image(frame, channels="BGR", caption="Captured Image")

    # Save image
    cv2.imwrite(TEST_IMG_PATH, frame)
    time.sleep(0.1)

    # Run model
    try:
        Emb_vec().check()
    except Exception as e:
        st.error(f"Model error: {e}")

    # Read result
    if os.path.exists(CHECK_FILE):
        with open(CHECK_FILE, "r") as f:
            data = f.read().strip()

            if data:
                person = data.split(":")[-1]
                st.success(f"✅ Recognized: {person}")
            else:
                st.warning("⚠️ Unknown person")
    else:
        st.warning("⚠️ No prediction file found")
