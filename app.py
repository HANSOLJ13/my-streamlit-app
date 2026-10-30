
import os
import base64
import streamlit as st
from openai import OpenAI

# OpenAI 클라이언트 생성
client =

# 페이지 설정
st.set_page_config(
    page_title="텍스트 음성 변환", # 브라우저 페이지 제목
    page_icon = "🔊" # 브라우저 아이콘
)
st.title("🔊텍스트 음성 변환")

# 텍스트 입력
text = st.text_area(
    "읽을 텍스트를 입력하세요",
    "안녕하세요. OpenAI 텍스트 음성 변환 실습입니다."
)

# 버튼
if st.button("음성 변환하기"):
    if text.strip() == "":
        st.warning("텍스트를 입력하세요")
    else:
        # TTS 생성
        speech = client.audio.speech.create(
            model="tts-1", # 음성 변환 모델
            voice="alloy", # 음성 종류
            input=text # 음성으로 변환할 텍스트
        )
        audio_bytes = speech.context # 생성된 MP3 음성 데이터 저장

        # 음성 출력
        st.markdown("### 🔊 음성 재생") # 소제목 출력
        st.audio(speech.content, format="audio/mp3") # 음성 플레이어 표시
