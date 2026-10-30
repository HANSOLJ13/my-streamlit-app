
import os
import streamlit as st
from openai import OpenAI

# OpenAI 클라이언트 생성
client = OpenAI()

# 페이지 설정
st.set_page_config(
    page_title="한영 번역기", # 브라우저 페이지 제목
    page_icon="🌐" # 페이지 아이콘
    )
st.title("🌐한영 번역기") # 웹 화면 제목
st.write("한글 문장을 영어로 번역합니다") # 안내 문구

# 사용자 입력(텍스트)
korean_text = st.text_area(
    "번역할 한글 문장을 입력하세요.", # 입력창 안내
    height=150, # 입력창 높이
    placeholder="예: 인공지능은 우리의 일상과 업무 방식을 빠르게 변화시키고 있습니다."
)

# 번역 버튼 클릭 시 실행
if st.button("번역하기"):

    if not korean_text.strip():
        st.warning("번역할 문장을 입력하세요")
    else:
        with st.spinner("번역 중입니다..."):

            # 시스템 역할
            sys_role = """
              당신은 전문 한영 번역가입니다. 
              한국어 문장을 자연스럽고 정확한 영어로 번역하세요.
              번역 결과만 출력하세요.
            """

            # 프롬프트 구성
            prompt = f"""
            다음 문장을 영어로 번역하세요.
            TEXT:
            {korean_text}
            """

            # OpenAI API (GPT) 호출
            response = client.responses.create(
                model="gpt-4o-mini", # 사용할 AI 모델
                input=[
                    {"role": "system", "content": prompt}, # AI 역할
                    {"role": "user", "content": prompt} # 번역 요청
                ]
            )

            # 결과 출력 (번역된 텍스트 추출)
            translated_text = response.ouput_text

            st.subheader("영어 번역 결과")
            st.success(translated_text)
