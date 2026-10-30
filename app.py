
import os
import streamlit as st
from openai import OpenAI

# OpenAI 클라이언트 생성
client = OpenAI()

# 페이지 설정
st.set_page_config(
    page_title="다국어 번역기", # 브라우저 페이지 제목
    page_icon="🌐" # 브라우저 아이콘
    )
st.title("🌐다국어 번역기") # 웹 화면 제목
st.write("한국어 문장을 다양한 언어로 번역합니다") # 안내 문구

# 사용자 입력(텍스트)
korean_text = st.text_area(
    "번역할 한글 문장을 입력하세요.", # 입력창 제목
    height=150,
    placeholder="예: 인공지능은 우리의 일상과 업무 방식을 빠르게 변화시키고 있습니다."
)
# 번역할 언어 선택
language = st.selectbox(
    "언어선택", # 선택창 제목
     ["영어", "일본어", "중국어", "불어", "독일어"], # 언어 목록
    index=0 # 기본 선택값: 영어
    )

# 번역 버튼 클릭 시 실행
if st.button("번역하기"):

    # 입력한 문장이 비어 있는지 확인
    if not korean_text.strip():
        st.warning("번역할 문장을 입력하세요")
    else:
        # 번역 처리 중 로딩 메시지 표시
        with st.spinner("번역 중입니다..."):

            # 시스템 역할
            sys_role = """
              당신은 전문 다국어 번역가입니다.
              사용자가 입력한 한국어 문장을 지정된 언어로 
              자연스럽고 정확하게 번역하세요.
              번역 결과만 출력하세요.
            """

            # 프롬프트 구성
            prompt = f"""
            다음 문장을 {language}로 번역하세요.
            ---
            "{korean_text}"
            """

            # GPT 호출
            response = client.responses.create(
                model="gpt-4o-mini",
                input=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": prompt}
                ]
            )

            # 결과 출력
            translated_text = response.output_text

            st.subheader("번역 결과")
            st.success(translated_text)
