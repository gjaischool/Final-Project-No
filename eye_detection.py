import cv2
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt
import time


# MediaPipe 얼굴 메쉬 모듈 불러오기
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh()
mp_drawing = mp.solutions.drawing_utils

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

# EAR 임계값 및 연속 프레임 깜박임 기준
EAR_THRESHOLD = 0.2
EAR_CONSEC_FRAMES = 15  # 0.5초 동안 감고 있으면 (assuming 30fps)

# 눈 깜박임 카운터 및 플래그
blink_counter = 0
warning_displayed = False
blink_start_time = None

# OpenCV로 웹캠 열기
cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # BGR을 RGB로 변환
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # MediaPipe로 얼굴 메쉬 추적
    result = face_mesh.process(frame_rgb)

    # 얼굴이 감지되면 랜드마크 그리기 (눈 주위만)
    if result.multi_face_landmarks:
        for face_landmarks in result.multi_face_landmarks:
            # 눈 랜드마크 그리기
            left_eye_landmarks = [33, 160, 158, 133, 153, 144]  # 왼쪽 눈
            right_eye_landmarks = [362, 385, 387, 263, 373, 380]  # 오른쪽 눈

            # 왼쪽 및 오른쪽 눈의 EAR 계산
            left_EAR = calculate_EAR(left_eye_landmarks, face_landmarks.landmark, frame.shape)
            right_EAR = calculate_EAR(right_eye_landmarks, face_landmarks.landmark, frame.shape)

            # 양쪽 눈의 평균 EAR
            ear = (left_EAR + right_EAR) / 2.0

            # EAR 값이 임계값보다 낮으면 눈이 감긴 것으로 간주
            if ear < EAR_THRESHOLD:
                if blink_start_time is None:
                    blink_start_time = time.time()  # 눈을 감기 시작한 시간 기록
                blink_counter += 1
            else:
                blink_start_time = None
                blink_counter = 0
                warning_displayed = False  # 눈이 다시 떴으므로 경고 초기화

            # 눈이 0.5초 이상 감겨 있으면 경고 표시
            if blink_start_time is not None:
                elapsed_time = time.time() - blink_start_time
                if elapsed_time > 0.5:
                    warning_displayed = True

            # 눈 주변 랜드마크만 그리기
            for eye_landmarks in [left_eye_landmarks, right_eye_landmarks]:
                for idx in eye_landmarks:
                    landmark = face_landmarks.landmark[idx]
                    h, w, _ = frame.shape
                    x, y = int(landmark.x * w), int(landmark.y * h)
                    cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

            # EAR 값 화면에 표시
            cv2.putText(frame, f'EAR: {ear:.2f}', (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

            # 경고 문구 화면에 표시
            if warning_displayed:
                cv2.putText(frame, 'WARNING!', (frame.shape[1] - 300, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

    # 결과 출력
    cv2.imshow('MediaPipe Face Mesh with Blink Detection', frame)

    # ESC를 눌러 종료
    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()