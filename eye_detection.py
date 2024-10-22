import cv2
import mediapipe as mp
import numpy as np
import time
import requests

# MediaPipe 얼굴 메쉬 모듈 불러오기
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh()
mp_drawing = mp.solutions.drawing_utils

# 눈 비율(EAR)을 계산하는 함수
def calculate_EAR(eye_landmarks, landmarks, image_shape):
    h, w, _ = image_shape
    coords = [(int(landmarks[idx].x * w), int(landmarks[idx].y * h)) for idx in eye_landmarks]
    # EAR 계산
    ear = (np.linalg.norm(np.array(coords[1]) - np.array(coords[5])) +
           np.linalg.norm(np.array(coords[2]) - np.array(coords[4]))) / (
              2.0 * np.linalg.norm(np.array(coords[0]) - np.array(coords[3])))
    return ear

# EAR 임계값 및 연속 프레임 깜박임 기준
EAR_THRESHOLD = 0.2

# 휴대폰 앱의 IP 주소와 포트 번호
phone_ip = '192.168.80.145' 
phone_port = 8080  # 앱에서 사용하는 포트 번호
alert_start_url = f'http://{phone_ip}:{phone_port}/alert/start'
alert_stop_url = f'http://{phone_ip}:{phone_port}/alert/stop'

# OpenCV로 IP 카메라 열기
ip_camera_url = f'http://{phone_ip}:{phone_port}'
cap = cv2.VideoCapture(ip_camera_url)

# 변수 초기화
warning_displayed = False
alert_sent = False  # 진동 신호가 전송되었는지 여부

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("Failed to read frame from camera.")
        break

    # BGR을 RGB로 변환
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # MediaPipe로 얼굴 메쉬 추적
    result = face_mesh.process(frame_rgb)

    # 얼굴이 감지되면 랜드마크 그리기 (눈 주위만)
    if result.multi_face_landmarks:
        face_landmarks = result.multi_face_landmarks[0]
        # 왼쪽 및 오른쪽 눈의 랜드마크 인덱스
        left_eye_landmarks = [33, 160, 158, 133, 153, 144]
        right_eye_landmarks = [362, 385, 387, 263, 373, 380]

        # EAR 계산
        left_EAR = calculate_EAR(left_eye_landmarks, face_landmarks.landmark, frame.shape)
        right_EAR = calculate_EAR(right_eye_landmarks, face_landmarks.landmark, frame.shape)
        ear = (left_EAR + right_EAR) / 2.0

        # EAR 값이 임계값보다 낮으면 눈이 감긴 것으로 간주
        if ear < EAR_THRESHOLD:
            if not warning_displayed:
                warning_displayed = True
                # 진동 시작 신호를 휴대폰 앱으로 전송
                try:
                    response = requests.post(alert_start_url)
                    print(f'Start alert sent to phone: {response.status_code}')
                except requests.exceptions.RequestException as e:
                    print(f'Failed to send start alert: {e}')
            cv2.putText(frame, 'WARNING!', (frame.shape[1] - 300, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        else:
            if warning_displayed:
                warning_displayed = False
                # 진동 중지 신호를 휴대폰 앱으로 전송
                try:
                    response = requests.post(alert_stop_url)
                    print(f'Stop alert sent to phone: {response.status_code}')
                except requests.exceptions.RequestException as e:
                    print(f'Failed to send stop alert: {e}')
            alert_sent = False

        # 눈 주변 랜드마크 그리기
        for eye_landmarks in [left_eye_landmarks, right_eye_landmarks]:
            for idx in eye_landmarks:
                landmark = face_landmarks.landmark[idx]
                h, w, _ = frame.shape
                x, y = int(landmark.x * w), int(landmark.y * h)
                cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

        # EAR 값 화면에 표시
        cv2.putText(frame, f'EAR: {ear:.2f}', (30, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    else:
        # 얼굴이 감지되지 않으면 경고 상태 초기화
        if warning_displayed:
            warning_displayed = False
            # 진동 중지 신호를 휴대폰 앱으로 전송
            try:
                response = requests.post(alert_stop_url)
                print(f'Stop alert sent to phone: {response.status_code}')
            except requests.exceptions.RequestException as e:
                print(f'Failed to send stop alert: {e}')
        alert_sent = False

    # 결과 출력
    cv2.imshow('Blink Detection', frame)

    # ESC를 눌러 종료
    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
