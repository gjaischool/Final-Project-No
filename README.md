# Final-Project-No
졸음 운전 방지 어플리케이션 만들기!

## How?
핸드폰 카메라를 통해 Face detection!

## 각 Code 요약
### eye_detection.py  
: opencv와 mediapipe를 결합하여 노트북에 내장된 캠을 통해 눈의 EAR 값을 좌측 상단에 표시하며 눈을 0.5초 이상 감겨 있으면 WARNING! 라는 문구를 우측 상단에 표시  
### eye_detection_for_streamlit.py  
: Streamlit을 이용하여 사진 촬영하면 촬영된 사진에 EAR 값 및 감겨있으면 WARNING! 문구 표시 
