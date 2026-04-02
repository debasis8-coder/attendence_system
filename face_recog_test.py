import streamlit as st
import cv2
import os
import time
from facenet_pytorch import MTCNN
from check_d import Emb_vec

# ---------------- CONFIG ----------------
st.set_page_config(page_title="Face Attendance System", layout="wide")

DATASET_PATH = "attendence_system/face dataset/"
TEST_IMG_PATH = "attendence_system/test_face/test.jpg"
CHECK_FILE = "attendence_system/check_face.txt"

# ---------------- SESSION ----------------
if "camera_on" not in st.session_state:
    st.session_state.camera_on = False

if "captured_frame" not in st.session_state:
    st.session_state.captured_frame = None

if "prediction" not in st.session_state:
    st.session_state.prediction = ""

# ---------------- HEADER ----------------
st.title("🎯 Face Recognition System")

# ---------------- SIDEBAR ----------------
st.sidebar.header("Controls")

name = st.sidebar.text_input("Enter Name")

# -------- REGISTER --------
if st.sidebar.button("📸 Register Face"):
    cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
    ret, frame = cap.read()
    cap.release()

    if ret and name.strip():
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mtcnn = MTCNN(image_size=160)
        mtcnn(frame, DATASET_PATH + f"{name}.jpg")
        st.success(f"{name} registered")
    else:
        st.error("Camera or name error")

# -------- CAMERA CONTROL --------
if st.sidebar.button("▶ Start Camera"):
    st.session_state.camera_on = True

if st.sidebar.button("⏹ Stop Camera"):
    st.session_state.camera_on = False

# ---------------- CAMERA STREAM ----------------
frame_placeholder = st.empty()

if st.session_state.camera_on:
    cap = cv2.VideoCapture(0, cv2.CAP_V4L2)

    ret, frame = cap.read()
    cap.release()

    if ret:
        frame = cv2.flip(frame, 1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        frame_placeholder.image(frame_rgb, channels="RGB")

        # Save latest frame in session
        st.session_state.captured_frame = frame.copy()

# ---------------- CAPTURE BUTTON ----------------
if st.button("📷 Capture & Predict"):

    frame = st.session_state.get("captured_frame", None)

    if frame is None:
        st.error("Start camera first")
    else:
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
                    st.session_state.prediction = data.split(":")[-1]
                else:
                    st.session_state.prediction = "Unknown"
        else:
            st.session_state.prediction = "Unknown"

# ---------------- RESULT ----------------
if st.session_state.prediction:
    if st.session_state.prediction != "Unknown":
        st.success(f"✅ Recognized: {st.session_state.prediction}")
    else:
        st.warning("⚠️ Unknown person")
