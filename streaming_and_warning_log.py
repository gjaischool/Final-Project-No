import cv2
import streamlit as st
import mediapipe as mp
import numpy as np
from PIL import Image
from datetime import datetime
import time

# Streamlit 페이지 설정
st.title("실시간 EAR 분석 및 경고 시스템")

# MediaPipe 얼굴 메쉬 모듈 불러오기
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# 눈 비율(EAR)을 계산하는 함수
def calculate_EAR(eye_landmarks, landmarks, image_shape):
    h, w, _ = image_shape
    p1 = np.array([landmarks[eye_landmarks[0]].x * w, landmarks[eye_landmarks[0]].y * h])
    p2 = np.array([landmarks[eye_landmarks[1]].x * w, landmarks[eye_landmarks[1]].y * h])
    p3 = np.array([landmarks[eye_landmarks[2]].x * w, landmarks[eye_landmarks[2]].y * h])
    p4 = np.array([landmarks[eye_landmarks[3]].x * w, landmarks[eye_landmarks[3]].y * h])
    p5 = np.array([landmarks[eye_landmarks[4]].x * w, landmarks[eye_landmarks[4]].y * h])
    p6 = np.array([landmarks[eye_landmarks[5]].x * w, landmarks[eye_landmarks[5]].y * h])

    # EAR 계산
    ear = (np.linalg.norm(p2 - p6) + np.linalg.norm(p3 - p5)) / (2.0 * np.linalg.norm(p1 - p4))
    return ear

# EAR 임계값
EAR_THRESHOLD = 0.2

# Streamlit의 세션 상태를 이용하여 웹캠 캡처 객체 및 경고 로그 유지
if 'cap' not in st.session_state:
    st.session_state.cap = cv2.VideoCapture(0)
    st.session_state.warnings = []

cap = st.session_state.cap
warnings = st.session_state.warnings

# Streamlit의 사이드바에 종료 버튼 추가
if st.sidebar.button("Stop"):
    cap.release()
    st.session_state.cap = None
    st.stop()

# 경고 로그 표시
st.sidebar.header("경고 로그")
for warning_time in warnings:
    st.sidebar.write(warning_time)

# Streamlit에 비디오 프레임을 표시할 자리 확보
frame_placeholder = st.empty()

# 실시간으로 프레임을 처리하고 EAR 계산
def process_frame():
    ret, frame = cap.read()
    if not ret:
        st.error("웹캠을 찾을 수 없습니다.")
        return None

    # 프레임 크기 조절 (필요 시)
    frame = cv2.resize(frame, (640, 480))

    # BGR을 RGB로 변환
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # MediaPipe로 얼굴 메쉬 추적
    result = face_mesh.process(frame_rgb)

    if result.multi_face_landmarks:
        for face_landmarks in result.multi_face_landmarks:
            # 눈 랜드마크 정의
            left_eye_landmarks = [33, 160, 158, 133, 153, 144]
            right_eye_landmarks = [362, 385, 387, 263, 373, 380]

            # EAR 계산
            left_EAR = calculate_EAR(left_eye_landmarks, face_landmarks.landmark, frame.shape)
            right_EAR = calculate_EAR(right_eye_landmarks, face_landmarks.landmark, frame.shape)
            ear = (left_EAR + right_EAR) / 2.0

            # EAR 값 표시 (콘솔에만 표시, 웹페이지에 표시하지 않음)
            # cv2.putText(frame, f'EAR: {ear:.2f}', (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

            # 경고 메시지 기록
            if ear < EAR_THRESHOLD:
                current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                warnings.append(current_time)
                st.sidebar.warning(f'EYES CLOSED at {current_time}')

            # 얼굴 메쉬 그리기 (옵션: 비디오에 그리지 않도록 주석 처리 가능)
            # mp_drawing.draw_landmarks(
            #     image=frame,
            #     landmark_list=face_landmarks,
            #     connections=mp_face_mesh.FACEMESH_TESSELATION,
            #     landmark_drawing_spec=None,
            #     connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1)
            # )

    # 현재 프레임을 웹페이지에 표시하지 않음 (비디오 스트림 제거)

    return frame

# 메인 루프 (Streamlit에서는 while 루프 사용이 제한적이므로, 애플리케이션의 주기적 실행을 활용)
while True:
    frame = process_frame()
    if frame is not None:
        # OpenCV의 BGR을 RGB로 변환하여 PIL 이미지로 변환
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)

        # Streamlit에 프레임 표시 (비디오 스트림 대신 업데이트된 이미지 표시)
        frame_placeholder.image(pil_image, channels="RGB")

    # 프레임 속도 조절 (예: 30 FPS)
    time.sleep(1/30)
