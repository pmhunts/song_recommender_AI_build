import streamlit as st
from streamlit_webrtc import webrtc_streamer
import av
import cv2
import numpy as np
import urllib.request
import os
import warnings
warnings.filterwarnings('ignore')

from keras.models import load_model
import webbrowser

models = load_model("model.h5")
labels = np.load("labels.npy")

# Download MediaPipe task models if not present
FACE_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
HAND_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"

if not os.path.exists("face_landmarker.task"):
    urllib.request.urlretrieve(FACE_MODEL_URL, "face_landmarker.task")

if not os.path.exists("hand_landmarker.task"):
    urllib.request.urlretrieve(HAND_MODEL_URL, "hand_landmarker.task")

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from mediapipe.tasks.python.vision.core.vision_task_running_mode import VisionTaskRunningMode

face_base_options = mp_python.BaseOptions(model_asset_path="face_landmarker.task")
face_options = mp_vision.FaceLandmarkerOptions(
    base_options=face_base_options,
    running_mode=VisionTaskRunningMode.IMAGE,
    num_faces=1
)
face_landmarker = mp_vision.FaceLandmarker.create_from_options(face_options)

hand_base_options = mp_python.BaseOptions(model_asset_path="hand_landmarker.task")
hand_options = mp_vision.HandLandmarkerOptions(
    base_options=hand_base_options,
    running_mode=VisionTaskRunningMode.IMAGE,
    num_hands=2
)
hand_landmarker = mp_vision.HandLandmarker.create_from_options(hand_options)

if "run" not in st.session_state:
    st.session_state["run"] = "true"

try:
    emotion = np.load("emotion.npy")[0]
except:
    emotion = ""

if not emotion:
    st.session_state["run"] = "true"
else:
    st.session_state["run"] = "false"


class EmotionProcessor:
    def recv(self, frame):
        frm = frame.to_ndarray(format="bgr24")
        frm = cv2.flip(frm, 1)
        rgb = cv2.cvtColor(frm, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        face_result = face_landmarker.detect(mp_image)
        hand_result = hand_landmarker.detect(mp_image)

        lst = []

        if face_result.face_landmarks:
            face_lms = face_result.face_landmarks[0]
            ref_x = face_lms[1].x
            ref_y = face_lms[1].y

            for lm in face_lms:
                lst.append(lm.x - ref_x)
                lst.append(lm.y - ref_y)

            left_hand = None
            right_hand = None

            if hand_result.hand_landmarks and hand_result.handedness:
                for hand_lms, handedness in zip(hand_result.hand_landmarks, hand_result.handedness):
                    label = handedness[0].category_name
                    if label == "Left":
                        left_hand = hand_lms
                    else:
                        right_hand = hand_lms

            if left_hand:
                ref_x_h = left_hand[8].x
                ref_y_h = left_hand[8].y
                for lm in left_hand:
                    lst.append(lm.x - ref_x_h)
                    lst.append(lm.y - ref_y_h)
            else:
                lst.extend([0.0] * 42)

            if right_hand:
                ref_x_h = right_hand[8].x
                ref_y_h = right_hand[8].y
                for lm in right_hand:
                    lst.append(lm.x - ref_x_h)
                    lst.append(lm.y - ref_y_h)
            else:
                lst.extend([0.0] * 42)

            lst = np.array(lst).reshape(1, -1)

            if lst.shape[1] == models.input_shape[1]:
                pred = labels[np.argmax(models.predict(lst))]
                cv2.putText(frm, str(pred), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                np.save("emotion.npy", np.array([pred]))

        return av.VideoFrame.from_ndarray(frm, format="bgr24")


st.title("🎵 Emotion-Based Song Recommender")
lang = st.text_input("Language")
singer = st.text_input("Singer")

if lang and singer and st.session_state["run"] != "false":
    webrtc_streamer(key="key", desired_playing_state=True, video_processor_factory=EmotionProcessor)

btn = st.button("Recommend me a song")
if btn:
    if not emotion:
        st.warning("Please let me capture your emotion first")
        st.session_state["run"] = "true"
    else:
        webbrowser.open(f"https://www.youtube.com/results?search_query={lang}+{emotion}+song+{singer}")
        np.save("emotion.npy", np.array([""]))
        st.session_state["run"] = "false"