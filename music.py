import streamlit as st
from streamlit_webrtc import webrtc_streamer
import av
import cv2
import numpy as np
import mediapipe as mp
from keras.models import load_model
import webbrowser
import os

# Load model and labels
models = load_model("model.h5")
labels = np.load("labels.npy")

# New MediaPipe API
mp_face_mesh = mp.solutions.face_mesh
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
hands_detector = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

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

        face_results = face_mesh.process(rgb)
        hand_results = hands_detector.process(rgb)

        lst = []

        if face_results.multi_face_landmarks:
            face_landmarks = face_results.multi_face_landmarks[0]
            ref_x = face_landmarks.landmark[1].x
            ref_y = face_landmarks.landmark[1].y

            for lm in face_landmarks.landmark:
                lst.append(lm.x - ref_x)
                lst.append(lm.y - ref_y)

            # Process hands
            left_hand = None
            right_hand = None

            if hand_results.multi_hand_landmarks and hand_results.multi_handedness:
                for hand_lm, handedness in zip(hand_results.multi_hand_landmarks, hand_results.multi_handedness):
                    label = handedness.classification[0].label
                    if label == "Left":
                        left_hand = hand_lm
                    else:
                        right_hand = hand_lm

            if left_hand:
                ref_x_h = left_hand.landmark[8].x
                ref_y_h = left_hand.landmark[8].y
                for lm in left_hand.landmark:
                    lst.append(lm.x - ref_x_h)
                    lst.append(lm.y - ref_y_h)
            else:
                lst.extend([0.0] * 42)

            if right_hand:
                ref_x_h = right_hand.landmark[8].x
                ref_y_h = right_hand.landmark[8].y
                for lm in right_hand.landmark:
                    lst.append(lm.x - ref_x_h)
                    lst.append(lm.y - ref_y_h)
            else:
                lst.extend([0.0] * 42)

            lst = np.array(lst).reshape(1, -1)

            # Check feature size matches model input
            if lst.shape[1] == models.input_shape[1]:
                pred = labels[np.argmax(models.predict(lst))]
                cv2.putText(frm, pred, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                np.save("emotion.npy", np.array([pred]))

            # Draw face mesh
            mp_drawing.draw_landmarks(
                frm,
                face_results.multi_face_landmarks[0],
                mp_face_mesh.FACEMESH_CONTOURS,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style()
            )

            # Draw hands
            if hand_results.multi_hand_landmarks:
                for hand_lm in hand_results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(frm, hand_lm, mp_hands.HAND_CONNECTIONS)

        return av.VideoFrame.from_ndarray(frm, format="bgr24")


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