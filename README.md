# Final-Project-No
졸음 운전 방지 어플리케이션 만들기!

## How?
핸드폰 카메라를 통해 Face detection!

## 각 Code 요약
### eye_detection.py  
: Android 기기를 IP 카메라로 변환하여 실시간으로 카메라 영상을 스트리밍하고, 눈 깜박임 감지를 통해 경고를 전송하는 코드
### eye_detection_for_streamlit.py  
: Streamlit을 이용하여 사진 촬영하면 촬영된 사진에 EAR 값 및 감겨있으면 WARNING! 문구 표시  
### cv2_DNN_detection.py  
: cv2.dnn 활용하여 얼굴 탐지. 사용시 models폴더 내에 있는 2개의 파일 활용해야함
