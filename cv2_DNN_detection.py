import cv2

# 사전 학습된 모델 파일 경로
modelFile = "models/res10_300x300_ssd_iter_140000.caffemodel"
configFile = "models/deploy.prototxt"

# 네트워크 모델 로드
net = cv2.dnn.readNetFromCaffe(configFile, modelFile)

# 실시간 비디오 스트림(웹캠) 시작
cap = cv2.VideoCapture(0)  # 0은 기본 웹캠을 의미

# 최소 신뢰도 임계값 설정
conf_threshold = 0.5

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 프레임 크기 가져오기
    frameHeight, frameWidth = frame.shape[:2]

    # 입력 이미지로부터 blob 생성
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104, 117, 123), False, False)
    net.setInput(blob)

    # 얼굴 탐지 수행
    detections = net.forward()

    # 감지된 얼굴에 대해 박스 그리기
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence > conf_threshold:
            # 감지된 얼굴의 좌표 계산
            x1 = int(detections[0, 0, i, 3] * frameWidth)
            y1 = int(detections[0, 0, i, 4] * frameHeight)
            x2 = int(detections[0, 0, i, 5] * frameWidth)
            y2 = int(detections[0, 0, i, 6] * frameHeight)

            # 얼굴 주변에 박스 그리기
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # 신뢰도 표시
            text = f"{confidence * 100:.2f}%"
            cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # 결과 출력
    cv2.imshow("Face Detection", frame)

    # 'q' 키를 누르면 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 캡처 종료 및 창 닫기
cap.release()
cv2.destroyAllWindows()