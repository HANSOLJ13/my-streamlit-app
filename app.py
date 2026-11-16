import os
import uuid
import hashlib
import tempfile

import streamlit as st

st.set_page_config(page_title="AIVLE School 학습 도우미", page_icon="🎓", layout="wide")

# API 키: Colab은 노트북에서 읽은 값, Cloud는 Secrets 사용
if not os.environ.get("OPENAI_API_KEY"):
    try:
        os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
    except Exception:
        st.error("OPENAI_API_KEY가 설정되지 않았어요.")
        st.stop()

from rag import rag_pipeline

LOGO_URL = "https://raw.githubusercontent.com/Jangrae/img/master/aivle.png"
SUGGESTED = [
    "비전공자도 따라갈 수 있나요?",
    "교육비와 자기부담금은 얼마인가요?",
    "미니 프로젝트는 어떻게 진행되나요?",
]

# ── 세션 상태 ───────────────────────────────────────────
# Streamlit은 입력할 때마다 파일 전체를 다시 실행하므로
# 체인과 대화 기록을 session_state에 저장해 둡니다.
ss = st.session_state
ss.setdefault("rag_chain", None)
ss.setdefault("file_hash", None)
ss.setdefault("file_name", None)
ss.setdefault("chats", {})        # {대화ID: {"title": 제목, "messages": [...]}}
ss.setdefault("current_id", None)


def new_chat():
    chat_id = uuid.uuid4().hex[:8]
    ss.chats[chat_id] = {"title": "새 대화", "messages": []}
    ss.current_id = chat_id


def delete_chat(chat_id):
    ss.chats.pop(chat_id, None)
    if ss.current_id == chat_id:
        ss.current_id = next(iter(ss.chats), None)


if ss.current_id is None:
    new_chat()


def build_chain(uploaded_file):
    data = uploaded_file.getvalue()
    file_hash = hashlib.md5(data).hexdigest()

    # 같은 파일이면 다시 벡터화하지 않음
    if file_hash == ss.file_hash:
        return

    # 업로드 객체 → 임시 .pdf 파일로 저장 → 경로를 rag_pipeline에 전달
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(data)
        tmp_path = tmp.name

    try:
        with st.spinner("백서를 읽고 벡터 저장소를 만드는 중입니다..."):
            ss.rag_chain = rag_pipeline(tmp_path)
        ss.file_hash = file_hash
        ss.file_name = uploaded_file.name
    finally:
        os.remove(tmp_path)


# ── 왼쪽 사이드바: 로고 · 검색 · 히스토리 ────────────────
with st.sidebar:
    st.image(LOGO_URL, width="stretch")

    keyword = st.text_input("검색", placeholder="🔍 대화 검색", label_visibility="collapsed")

    st.markdown("**히스토리**")
    if st.button("＋ 새 대화", width="stretch"):
        new_chat()
        st.rerun()

    for chat_id, chat in reversed(list(ss.chats.items())):
        # 검색어가 제목이나 대화 내용에 있는 것만 표시
        text = chat["title"] + " ".join(m["content"] for m in chat["messages"])
        if keyword and keyword not in text:
            continue

        c1, c2 = st.columns([5, 1], gap="small", vertical_alignment="center")
        if c1.button(chat["title"], key=f"open_{chat_id}", width="stretch", type="tertiary"):
            ss.current_id = chat_id
            st.rerun()
        if c2.button("✕", key=f"del_{chat_id}", help="대화 삭제", type="tertiary"):
            delete_chat(chat_id)
            if not ss.chats:
                new_chat()
            st.rerun()

current = ss.chats[ss.current_id]

# ── 히스토리 목록 스타일 (ChatGPT처럼 글자만 보이고, 마우스를 올리면 배경 표시) ──
st.markdown(f"""
<style>
/* 히스토리 제목 버튼: 테두리 없이 왼쪽 정렬, 길면 말줄임 */
[class*="st-key-open_"] button {{
    justify-content: flex-start;
    padding: 6px 10px;
    border-radius: 8px;
    color: inherit;
}}
[class*="st-key-open_"] button > div {{
    justify-content: flex-start;
    width: 100%;
}}
[class*="st-key-open_"] button p {{
    width: 100%;
    text-align: left;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}}
/* 히스토리 줄 간격 좁히기 */
[data-testid="stSidebar"] [data-testid="stHorizontalBlock"] {{
    margin-bottom: -0.7rem;
}}
/* 마우스를 올렸을 때 */
[class*="st-key-open_"] button:hover {{
    background-color: rgba(0, 0, 0, 0.06);
    color: inherit;
}}
/* 지금 보고 있는 대화 */
.st-key-open_{ss.current_id} button {{
    background-color: rgba(0, 0, 0, 0.08);
    font-weight: 600;
}}
/* 삭제(✕) 버튼: 항상 보이도록, 흐린 회색 */
[class*="st-key-del_"] button {{
    color: #999;
}}
[class*="st-key-del_"] button:hover {{
    color: #e5484d;
}}
</style>
""", unsafe_allow_html=True)

# ── 헤더 ────────────────────────────────────────────────
st.title("🎓 AIVLE School 학습 도우미")
st.caption(
    f"적용된 문서: {ss.file_name}" if ss.file_name
    else "아래 ＋ 파일 업로드에서 백서 PDF를 올리면 질문할 수 있어요."
)

# ── 본문: 채팅(왼쪽) · 추천 질문(오른쪽) ────────────────
chat_col, suggest_col = st.columns([3, 1], gap="large")

# 오른쪽: 추천 질문 카드
with suggest_col:
    st.markdown("**추천 질문**")
    for i, q in enumerate(SUGGESTED):
        with st.container(border=True):
            st.markdown(f"**Q.** {q}")
            if st.button("질문하기", key=f"suggest_{i}", width="stretch",
                         disabled=ss.rag_chain is None):
                ss.pending_question = q

# 왼쪽: 대화 영역 + 하단 입력줄(파일 업로드 · 채팅 입력)
with chat_col:
    chat_box = st.container(height=480)   # 대화 내용이 표시될 자리

    up_col, input_col = st.columns([1.5, 4.5], vertical_alignment="bottom")
    with up_col:
        with st.popover("＋ 파일 업로드", width="stretch"):
            uploaded = st.file_uploader("AIVLE School 백서(PDF)", type="pdf")
            if uploaded is not None:
                build_chain(uploaded)
                st.success("적용 완료")
    with input_col:
        user_input = st.chat_input("채팅 입력", disabled=ss.rag_chain is None)

    question = user_input or ss.pop("pending_question", None)

    with chat_box:
        for msg in current["messages"]:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        if question:
            # 첫 질문을 히스토리 제목으로 사용
            if not current["messages"]:
                current["title"] = question[:20]

            current["messages"].append({"role": "user", "content": question})
            with st.chat_message("user"):
                st.markdown(question)

            with st.chat_message("assistant"):
                with st.spinner("답변을 찾는 중..."):
                    try:
                        answer = ss.rag_chain.invoke({"input": question})
                    except Exception as e:
                        answer = f"오류가 발생했어요: {e}"
                st.markdown(answer)

            current["messages"].append({"role": "assistant", "content": answer})
            st.rerun()   # 히스토리 제목을 바로 갱신
