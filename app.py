
import time # time: 시간 지연 기능을 사용하기 위한 모듈
import streamlit as st # streamlit: 웹 애플리케이션을 만드는 라이브러리
from langchain_openai import ChatOpenAI # ChatOpenAI: OpenAI의 AI 모델을 사용하기 위한 클래스

# HumanMessage: 사용자 메시지
# SystemMessage: AI의 역할과 행동 지침
# AIMessage: AI가 생성한 답변
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

# 페이지 설정
# set_page_config(): 웹페이지 제목, 아이콘, 레이아웃 설정
st.set_page_config(page_title="LLM 챗봇", page_icon="💬", layout="centered")
# title(): 웹페이지에 제목 출력
st.title("💬 LLM과의 대화")

# LLM 구성
# ChatOpenAI(): 사용할 AI 모델 설정
llm = ChatOpenAI(model="gpt-4o-mini")

# 세션 상태 초기화
# session_state: 웹페이지가 다시 실행되어도 대화 기록을 유지하는 저장 공간
# 메시지 기록이 없으면 빈 리스트 생성
if "messages" not in st.session_state:
    st.session_state.messages = []

# 기존 대화 출력
# for: 저장된 기존 대화 내용을 하나씩 출력
for message in st.session_state.messages:
    # isinstance(): 메시지가 HumanMessage인지 확인하는 함수
    role = "user" if isinstance(message, HumanMessage) else "assistant"
    # chat_message(): 사용자 또는 AI의 채팅 메시지 영역 생성
    with st.chat_message(role):
        # markdown(): 메시지 내용을 화면에 출력
        st.markdown(message.content)

# 사용자 입력
# chat_input(): 사용자에게 채팅 메시지를 입력받는 함수
user_input = st.chat_input("메시지를 입력하세요...")

# 사용자가 메시지를 입력한 경우 실행
if user_input:
    # 사용자 메시지 출력 및 저장
    # chat_message(): 사용자 메시지를 채팅창에 출력
    st.chat_message("user").markdown(user_input)
    # append(): 사용자 메시지를 대화 기록에 추가
    st.session_state.messages.append(HumanMessage(content=user_input))

    # 질문과 답변
    # invoke(): 이전 대화와 현재 질문을 AI 모델에 전달하고 답변 받기
    response = llm.invoke(st.session_state.messages)

    # 타이핑 효과로 출력
    # chat_message(): AI 답변을 표시할 채팅 영역 생성
    with st.chat_message("assistant"):
        # empty(): 내용을 실시간으로 변경할 수 있는 빈 공간 생성
        response_container = st.empty()
        # AI 답변을 저장할 빈 문자열 생성
        full_response = ""
        # for: AI 답변을 한 글자씩 반복하여 출력
        for char in response.content:
            # 답변을 한 글자씩 누적
            full_response += char
            # markdown(): 누적된 답변과 커서(▌)를 화면에 출력
            response_container.markdown(full_response + "▌")
            # sleep(): 0.01초씩 지연시켜 타이핑 효과 구현
            time.sleep(0.01)
        # markdown(): 완성된 답변을 커서 없이 최종 출력
        response_container.markdown(full_response)

    # 대화 기록 저장
    # AIMessage(): AI 답변을 메시지 형식으로 생성
    # append(): AI 답변을 대화 기록에 추가
    st.session_state.messages.append(AIMessage(content=full_response))
