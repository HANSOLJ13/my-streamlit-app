import os
import hashlib
import tempfile

import streamlit as st

st.set_page_config(page_title="AIVLE School 학습 도우미", page_icon="🎓")

# API 키: Colab은 노트북에서 읽은 값, Cloud는 Secrets 사용
if not os.environ.get("OPENAI_API_KEY"):
    try:
        os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
    except Exception:
        st.error("OPENAI_API_KEY가 설정되지 않았어요.")
        st.stop()

from rag import rag_pipeline

# Streamlit은 입력할 때마다 파일 전체를 다시 실행하므로
# 체인과 대화 기록을 session_state에 저장해 둡니다.
if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None
if "file_hash" not in st.session_state:
    st.session_state.file_hash = None
if "file_name" not in st.session_state:
    st.session_state.file_name = None
if "messages" not in st.session_state:
    st.session_state.messages = []


def build_chain(uploaded_file):
    data = uploaded_file.getvalue()
    file_hash = hashlib.md5(data).hexdigest()

    # 같은 파일이면 다시 벡터화하지 않음
    if file_hash == st.session_state.file_hash:
        return

    # 업로드 객체 → 임시 .pdf 파일로 저장 → 경로를 rag_pipeline에 전달
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(data)
        tmp_path = tmp.name

    try:
        with st.spinner("백서를 읽고 벡터 저장소를 만드는 중입니다..."):
            st.session_state.rag_chain = rag_pipeline(tmp_path)
        st.session_state.file_hash = file_hash
        st.session_state.file_name = uploaded_file.name
        st.session_state.messages = []
    finally:
        os.remove(tmp_path)


# 사이드바: 업로드, 추천 질문, 대화 삭제
with st.sidebar:
    st.header("📄 백서 업로드")
    uploaded = st.file_uploader("AIVLE School 백서(PDF)", type="pdf")
    if uploaded is not None:
        build_chain(uploaded)

    if st.session_state.file_name:
        st.success(f"적용된 문서: {st.session_state.file_name}")
    else:
        st.info("PDF를 올리면 질문할 수 있어요.")

    st.divider()
    st.subheader("💡 추천 질문")
    suggested = [
        "비전공자도 따라갈 수 있나요?",
        "교육비와 자기부담금은 얼마인가요?",
        "미니 프로젝트는 어떻게 진행되나요?",
    ]
    for q in suggested:
        if st.button(q, use_container_width=True, disabled=st.session_state.rag_chain is None):
            st.session_state.pending_question = q

    st.divider()
    if st.button("🗑️ 대화 지우기", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


# 메인 화면: 채팅
st.title("🎓 AIVLE School 학습 도우미")
st.caption("백서 내용을 바탕으로 에이블스쿨에 대한 질문에 답합니다.")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_input = st.chat_input("질문을 입력하세요", disabled=st.session_state.rag_chain is None)
question = user_input or st.session_state.pop("pending_question", None)

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("답변을 찾는 중..."):
            try:
                answer = st.session_state.rag_chain.invoke({"input": question})
            except Exception as e:
                answer = f"오류가 발생했어요: {e}"
        st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})
