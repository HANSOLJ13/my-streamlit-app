# Streamlit Cloud의 sqlite 버전 문제 대비 (chromadb보다 먼저 실행돼야 함, Colab에서는 영향 없음)
try:
    __import__("pysqlite3"); import sys; sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
except ImportError:
    pass

import os
import uuid
import hashlib
import tempfile
import warnings
from operator import itemgetter

import streamlit as st

warnings.filterwarnings(action="ignore")

import pymupdf
pymupdf.set_messages(path=os.devnull)

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_pymupdf4llm import PyMuPDF4LLMLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

st.set_page_config(page_title="AIVLE School 학습 도우미", page_icon="🎓", layout="wide")

# API 키: Colab은 노트북에서 읽은 값, Cloud는 Secrets 사용
if not os.environ.get("OPENAI_API_KEY"):
    try:
        os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
    except Exception:
        st.error("OPENAI_API_KEY가 설정되지 않았어요.")
        st.stop()


# ════════════════════════════════════════════════════════
# [RAG 영역] 답변을 만드는 부분 (구 rag.py)
# ════════════════════════════════════════════════════════

# ① 설정값 (조별 표준안이 정해지면 여기만 바꾸세요)
CHUNK_SIZE      = 600        # 한 조각의 최대 글자 수
CHUNK_OVERLAP   = 100        # 앞뒤 조각이 겹치는 글자 수 (CHUNK_SIZE보다 작아야 함)
TOP_K           = 5          # 답변에 넘길 문단 수
FETCH_K         = 20         # MMR이 먼저 뽑는 후보 문단 수
EMBEDDING_MODEL = "text-embedding-3-small"
LLM_MODEL       = "gpt-4o-mini"
TEMPERATURE     = 0          # 0에 가까울수록 문서에 충실한 답변
NOT_FOUND_MSG   = "백서에서 해당 내용을 찾을 수 없습니다."

# ④ 프롬프트 템플릿
SYSTEM_TEMPLATE = """
[역할]
당신은 KT 에이블스쿨 지원자와 교육생을 돕는 학습 도우미입니다.
아래 [문서]는 에이블스쿨 백서에서 질문과 관련된 부분만 뽑은 것입니다.

[답변 규칙]
1. 반드시 [문서]에 있는 내용만 근거로 답하세요. 일반 상식이나 추측을 더하지 마세요.
2. 숫자, 날짜, 금액, 비율, 기수는 [문서]에 적힌 그대로 옮기세요.
3. 기수나 시점에 따라 내용이 다르면 "10기 기준", "2024년 기준"처럼 기준을 함께 밝히세요.
4. [문서]에 일부만 있으면 확인되는 부분만 답하고, 나머지는 백서에 없다고 말하세요.
5. [문서]에 전혀 없으면 다른 말을 덧붙이지 말고 "{not_found}"라고만 답하세요.
6. 질문이 모호하면 가장 가능성 높은 뜻으로 답하고, 어떻게 이해했는지 한 줄로 밝히세요.

[답변 형식]
- 첫 문장에서 질문에 대한 핵심 답을 바로 말하세요.
- 항목이 3개 이상이면 글머리표로 정리하세요.
- 답변 끝에 근거가 된 쪽 번호를 (근거: p.3, p.5)처럼 적으세요.
- 친절하고 격려하는 말투로, 존댓말을 쓰세요.

[문서]
{context}
""".strip()

prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_TEMPLATE),
    ("human", "{input}"),
]).partial(not_found=NOT_FOUND_MSG)


# 문서 포맷터: 각 문단 앞에 쪽 번호를 붙여 근거를 표시할 수 있게 함
def format_docs(docs):
    blocks = []
    for d in docs:
        page = d.metadata.get("page")
        label = f"[p.{page + 1}]" if isinstance(page, int) else "[p.?]"
        blocks.append(f"{label}\n{d.page_content}")
    return "\n\n---\n\n".join(blocks)


# RAG 파이프라인 함수 (01번 노트북의 rag_pipeline 그대로)
def rag_pipeline(filepath):

    # PDF 파일 읽기 (PyMuPDF4LLM은 제목을 마크다운 #으로 바꿔 줌)
    loader = PyMuPDF4LLMLoader(filepath)
    docs = loader.load()

    # ② 텍스트 분할: 장·절 제목 → 문단 → 줄 → 문장 순서로 자르기
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n## ", "\n### ", "\n#### ", "\n\n", "\n", ". ", " ", ""],
    )
    splits = text_splitter.split_documents(docs)

    # Embedding 모델
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)

    # 벡터저장소 만들기
    vectorstore = Chroma(
        collection_name="aivle_docs",
        embedding_function=embeddings,
        persist_directory="chroma_db",
    )

    # 기존 데이터 비우기
    vectorstore.reset_collection()

    # 문서 추가
    vectorstore.add_documents(splits)

    # ③ 검색기 구성: MMR로 관련 있으면서 서로 겹치지 않는 문단 선택
    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": TOP_K, "fetch_k": FETCH_K},
    )

    # LLM 선언
    llm = ChatOpenAI(model=LLM_MODEL, temperature=TEMPERATURE)

    # RAG 체인 구성
    rag_chain = (
        {
            "context": itemgetter("input") | retriever | format_docs,
            "input": itemgetter("input"),
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain


# ════════════════════════════════════════════════════════
# [화면 영역] 사용자가 보는 부분 (구 app.py)
# ════════════════════════════════════════════════════════

import math                      # [추가] 유사도(코사인) 계산
import threading                 # [추가] 여러 사용자가 동시에 질문해도 횟수가 꼬이지 않게 잠금
from collections import Counter  # [추가] 질문 횟수 세기

LOGO_URL = "https://raw.githubusercontent.com/Jangrae/img/master/aivle.png"
MODEL_NAME = LLM_MODEL       # 하단 안내 문구용 (위 설정값을 그대로 사용)

# [수정] 기본 추천 질문
# - 아직 질문 기록이 3개가 안 될 때 빈자리를 채우는 용도입니다.
# - 질문이 쌓이면 '많이 묻는 질문'이 앞자리부터 차지합니다.
DEFAULT_SUGGESTED = [
    "비전공자도 따라갈 수 있나요?",
    "교육비와 자기부담금은 얼마인가요?",
    "미니 프로젝트는 어떻게 진행되나요?",
]
TOP_N = 3                    # [추가] 추천 질문 카드 개수

# [추가] 같은 뜻 질문으로 볼 유사도 기준값 (0~1, 높을수록 엄격)
# - 너무 낮으면: 다른 질문까지 하나로 묶임 (예: 교육비 ↔ 훈련장려금)
# - 너무 높으면: 같은 뜻인데도 따로 셈 (예: 교육비 얼마예요? ↔ 비용이 얼마인가요?)
# - 0.80에서 시작해, 테스트 질문으로 확인하며 0.75~0.90 사이에서 조정하세요.
SIMILAR_THRESHOLD = 0.80


# ════════════════════════════════════════════════════════
# [추가] 많이 묻는 질문 TOP3 기능 (임베딩 유사도로 같은 뜻 질문 묶기)
# ════════════════════════════════════════════════════════
# 동작 순서
#   ① 새 질문을 임베딩(뜻을 숫자 목록으로 바꾼 것)으로 변환
#   ② 이미 기록된 질문 묶음들과 유사도를 비교
#   ③ 가장 비슷한 묶음의 유사도가 SIMILAR_THRESHOLD 이상 → 그 묶음 횟수 +1
#      아니면 → 새 묶음으로 등록 (이 질문이 카드에 보일 대표 문장이 됨)
#
# st.session_state는 사용자(브라우저 탭)마다 따로 저장되지만,
# st.cache_resource로 만든 객체는 앱을 쓰는 '모든 사용자'가 함께 씁니다.
# → 여러 사람이 물어본 질문 횟수를 한곳에 모아 TOP3를 뽑을 수 있습니다.
# ※ 앱이 재시작(재배포·장시간 미사용)되면 횟수는 0부터 다시 셉니다.
@st.cache_resource
def get_question_stats():
    return {
        "counts": Counter(),   # {묶음 번호: 횟수}
        "texts": {},           # {묶음 번호: 카드에 보여줄 대표 문장(처음 들어온 질문)}
        "norms": {},           # {묶음 번호: 정리된 대표 문장} → 완전히 같은 질문은 임베딩 없이 바로 찾기
        "vecs": {},            # {묶음 번호: 대표 문장의 임베딩}
        "lock": threading.Lock(),
    }


# [추가] 질문 임베딩용 모델 (RAG와 같은 모델 사용, 모든 사용자가 하나를 공유)
@st.cache_resource
def get_question_embedder():
    return OpenAIEmbeddings(model=EMBEDDING_MODEL)


def normalize_question(q):
    """띄어쓰기·끝의 물음표 차이로 같은 질문이 따로 세지지 않도록 정리"""
    q = " ".join(q.split())             # 여러 칸 띄어쓰기 → 한 칸
    return q.rstrip("?？.!~ ").strip()


def cosine_similarity(a, b):
    """[추가] 두 임베딩이 얼마나 비슷한지 계산 (1에 가까울수록 같은 뜻)"""
    dot = sum(x * y for x, y in zip(a, b))
    norm = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    return dot / norm if norm else 0.0


def embed_question(text):
    """[추가] 질문 → 임베딩. 실패하면 None을 돌려줘 앱이 멈추지 않게 함"""
    try:
        return get_question_embedder().embed_query(text)
    except Exception:
        return None


def find_similar_group(vec, groups, threshold=SIMILAR_THRESHOLD):
    """[추가] 기록된 묶음 중 가장 비슷한 묶음 번호 찾기 (기준값 미만이면 None)"""
    best_id, best_score = None, threshold
    for gid, gvec in groups.items():
        score = cosine_similarity(vec, gvec)
        if score >= best_score:
            best_id, best_score = gid, score
    return best_id


def record_question(question, answer):
    """[수정] 답변을 받은 질문을 횟수에 더하기 (같은 뜻 질문은 하나로 묶어서 셈)
    - 백서에 없는 질문(NOT_FOUND_MSG)이나 오류 답변은 추천하면 안 되므로 세지 않습니다.
    """
    if NOT_FOUND_MSG in answer or answer.startswith("오류가 발생했어요"):
        return
    key = normalize_question(question)
    if not key:
        return
    stats = get_question_stats()

    # ① 글자까지 완전히 같은 질문이 이미 있으면 → 임베딩 없이 바로 +1 (API 호출 절약)
    with stats["lock"]:
        for gid, norm in stats["norms"].items():
            if norm == key:
                stats["counts"][gid] += 1
                return
        groups = dict(stats["vecs"])   # 비교용 복사본 (API 호출 동안 잠그지 않으려고)

    # ② 임베딩으로 뜻이 비슷한 묶음 찾기
    vec = embed_question(key)
    gid = find_similar_group(vec, groups) if vec is not None else None

    with stats["lock"]:
        if gid is not None:
            # ③-1 비슷한 묶음이 있음 → 그 묶음 횟수 +1
            stats["counts"][gid] += 1
        else:
            # ③-2 처음 보는 질문 → 새 묶음 등록 (임베딩 실패 시엔 글자 비교로만 묶임)
            new_id = len(stats["texts"])
            stats["counts"][new_id] = 1
            stats["texts"][new_id] = question.strip()
            stats["norms"][new_id] = key
            if vec is not None:
                stats["vecs"][new_id] = vec


# [추가] 기본 추천 질문의 임베딩 (처음 필요할 때 한 번만 계산해서 공유)
@st.cache_resource
def get_default_vecs():
    return {q: embed_question(normalize_question(q)) for q in DEFAULT_SUGGESTED}


def get_top_questions(n=TOP_N):
    """[수정] 많이 물은 순서로 n개 반환: [(질문, 횟수), ...]
    - 기록이 부족하면 DEFAULT_SUGGESTED로 채우고, 이때 횟수는 None
    - 이미 TOP에 있는 질문과 뜻이 같은 기본 질문은 건너뜀 (카드 중복 방지)
    """
    stats = get_question_stats()
    with stats["lock"]:
        top_ids = [gid for gid, _ in stats["counts"].most_common(n)]
        top = [(stats["texts"][g], stats["counts"][g]) for g in top_ids]
        top_norms = {stats["norms"][g] for g in top_ids}
        top_vecs = {g: stats["vecs"][g] for g in top_ids if g in stats["vecs"]}

    if len(top) >= n:
        return top

    default_vecs = get_default_vecs() if top_vecs else {}
    for q in DEFAULT_SUGGESTED:
        if len(top) >= n:
            break
        if normalize_question(q) in top_norms:
            continue
        dvec = default_vecs.get(q)
        if dvec is not None and find_similar_group(dvec, top_vecs) is not None:
            continue   # 이미 카드에 있는 질문과 같은 뜻 → 건너뜀
        top.append((q, None))
    return top


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
    if not ss.chats:
        new_chat()


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


def ask(question):
    try:
        return ss.rag_chain.invoke({"input": question})
    except Exception as e:
        return f"오류가 발생했어요: {e}"


# ── 스타일 ──────────────────────────────────────────────
st.markdown(f"""
<style>
.block-container {{ padding-top: 4rem; padding-bottom: 1rem; }}

/* 사이드바 메뉴(새 대화): 글자만 보이고 마우스를 올리면 배경 */
[class*="st-key-nav_"] button {{
    justify-content: flex-start; padding: 6px 10px; border-radius: 8px; color: inherit;
}}
[class*="st-key-nav_"] button > div {{ justify-content: flex-start; width: 100%; }}
[class*="st-key-nav_"] button p {{
    width: 100%; text-align: left; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
}}
[class*="st-key-nav_"] button:hover {{ background-color: rgba(0,0,0,0.06); color: inherit; }}
.st-key-nav_new button {{ background-color: rgba(0,0,0,0.06); font-weight: 600; }}

/* 사이드바 하단 문서 카드 */
.doc-card {{
    margin-top: 1.5rem; padding: 12px 14px; border: 1px solid rgba(0,0,0,0.1);
    border-radius: 12px; font-size: 0.85rem; line-height: 1.4;
}}
.doc-card .label {{ color: #888; font-size: 0.75rem; }}

/* 상단 바 */
.topbar {{
    display: flex; justify-content: space-between; align-items: center;
    flex-wrap: wrap; gap: 8px;
    padding: 4px 0 12px; margin-bottom: 8px; border-bottom: 1px solid rgba(0,0,0,0.08);
}}
.topbar .title {{ font-size: 1.35rem; font-weight: 700; line-height: 1.6; }}
.topbar .badge {{
    font-size: 0.8rem; padding: 6px 12px; border-radius: 8px;
    background: #111; color: #fff;
}}
.topbar .badge.off {{ background: #eee; color: #666; }}

/* 채팅 말풍선: 질문은 테두리 박스, 답변은 회색 카드 */
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {{
    background: transparent;
}}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"])
  [data-testid="stChatMessageContent"] {{
    flex: 0 1 auto; margin-left: 0 !important; margin-right: auto !important;
    border: 1px solid rgba(0,0,0,0.12); border-radius: 12px; padding: 6px 14px;
}}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) {{
    background: #f6f7f9; border-radius: 14px; padding: 14px 16px;
}}

/* [수정] 왼쪽 사이드바 히스토리: 제목 + ✕, 아래 밑줄 (스케치 형태) */
[class*="st-key-hist_"] {{
    border-bottom: 1px solid rgba(0,0,0,0.15); padding: 2px 0 4px; gap: 0;
}}
.st-key-hist_{ss.current_id} [class*="st-key-open_"] button p {{ font-weight: 700; }}
[class*="st-key-open_"] button {{
    justify-content: flex-start; padding: 0; min-height: 0; color: inherit;
}}
[class*="st-key-open_"] button > div {{ justify-content: flex-start; width: 100%; }}
[class*="st-key-open_"] button p {{
    width: 100%; text-align: left; font-size: 0.9rem; font-weight: 400;
    white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
}}
[class*="st-key-open_"] button:hover {{ color: inherit; text-decoration: underline; }}
[class*="st-key-del_"] button {{ color: #aaa; padding: 0; min-height: 0; }}
[class*="st-key-del_"] button:hover {{ color: #e5484d; }}

/* [추가] 오른쪽 추천 질문(TOP3) 카드 */
[class*="st-key-faq_"] {{
    border: 1px solid rgba(0,0,0,0.15); border-radius: 12px;
    padding: 12px 14px; min-height: 110px; gap: 0.4rem;
}}
[class*="st-key-faq_"]:hover {{ background: rgba(0,0,0,0.03); border-color: rgba(0,0,0,0.3); }}
[class*="st-key-faqbtn_"] button {{
    justify-content: flex-start; padding: 0; min-height: 0; color: inherit;
}}
[class*="st-key-faqbtn_"] button > div {{ justify-content: flex-start; width: 100%; }}
[class*="st-key-faqbtn_"] button p {{
    width: 100%; text-align: left; white-space: normal;   /* 긴 질문은 줄바꿈 */
    font-weight: 600; font-size: 0.9rem; line-height: 1.45;
}}
[class*="st-key-faqbtn_"] button:hover {{ color: inherit; }}
.faq-meta {{ font-size: 0.75rem; color: #999; }}
.faq-meta.hot {{ color: #e5484d; font-weight: 600; }}

/* 하단 안내 문구 */
.disclaimer {{ text-align: center; font-size: 0.75rem; color: #999; margin-top: 4px; }}
</style>
""", unsafe_allow_html=True)


# ── [수정] 왼쪽 사이드바: 로고 · 검색 · 새 대화 · 히스토리 · 문서 상태 ──
# (스케치처럼 히스토리를 오른쪽에서 왼쪽 사이드바로 옮겼습니다)
with st.sidebar:
    st.image(LOGO_URL, width=150)
    keyword = st.text_input("검색", placeholder="🔍 대화 검색", label_visibility="collapsed")

    if st.button("💬  새 대화", key="nav_new", width="stretch", type="tertiary"):
        new_chat()
        st.rerun()

    # 히스토리 목록: 제목 클릭 → 해당 대화 열기, ✕ → 대화 삭제
    st.markdown(f"**히스토리** <span style='color:#999'>({len(ss.chats)})</span>",
                unsafe_allow_html=True)
    for chat_id, chat in reversed(list(ss.chats.items())):
        text = chat["title"] + " ".join(m["content"] for m in chat["messages"])
        if keyword and keyword not in text:
            continue

        with st.container(key=f"hist_{chat_id}"):
            c1, c2 = st.columns([6, 1], gap="small", vertical_alignment="center")
            if c1.button(chat["title"], key=f"open_{chat_id}", width="stretch", type="tertiary"):
                ss.current_id = chat_id
                st.rerun()
            if c2.button("✕", key=f"del_{chat_id}", help="대화 삭제", type="tertiary"):
                delete_chat(chat_id)
                st.rerun()

    doc = ss.file_name or "아직 문서가 없어요"
    st.markdown(
        f'<div class="doc-card"><div class="label">적용된 문서</div>📄 {doc}</div>',
        unsafe_allow_html=True,
    )

current = ss.chats[ss.current_id]

# ── 상단 바 ─────────────────────────────────────────────
badge = '<span class="badge">✔ 백서 적용됨</span>' if ss.rag_chain else '<span class="badge off">문서 미적용</span>'
st.markdown(
    f'<div class="topbar"><span class="title">AIVLE School 학습 도우미</span>{badge}</div>',
    unsafe_allow_html=True,
)

# ── [수정] 본문: 채팅(가운데) · 추천 질문 TOP3(오른쪽) ───
chat_col, faq_col = st.columns([3, 1], gap="large")

# [추가] 오른쪽: 많이 묻는 질문 TOP3 카드
# - 카드를 누르면 그 질문이 바로 채팅으로 들어갑니다(pending_question).
# - 실제로 물어본 횟수가 있으면 '🔥 N회 질문', 기본 질문이면 '추천 질문'으로 표시
with faq_col:
    st.markdown("**추천 질문**")
    st.caption("많이 묻는 질문 TOP3")
    for i, (q, cnt) in enumerate(get_top_questions()):
        with st.container(key=f"faq_{i}"):
            if st.button(f"Q. {q}", key=f"faqbtn_{i}", width="stretch", type="tertiary",
                         disabled=ss.rag_chain is None):
                ss.pending_question = q
            if cnt:
                meta = f'<span class="faq-meta hot">🔥 {cnt}회 질문</span>'
            else:
                meta = '<span class="faq-meta">💡 추천 질문</span>'
            st.markdown(meta, unsafe_allow_html=True)
    if ss.rag_chain is None:
        st.caption("백서를 올리면 눌러서 바로 물어볼 수 있어요.")

# 가운데: 대화 영역 + 다시 생성 + 입력창 + 첨부
with chat_col:
    chat_box = st.container(height=470, border=False)

    # 다시 생성: 마지막 질문으로 답변을 새로 받기
    regenerate = False
    if current["messages"] and ss.rag_chain is not None:
        _, mid, _ = st.columns([2, 1, 2])
        regenerate = mid.button("↻ 다시 생성", width="stretch")

    with st.container(border=True):
        user_input = st.chat_input("메시지를 입력하세요", max_chars=3000,
                                   disabled=ss.rag_chain is None)
        a1, a2 = st.columns([1, 3], vertical_alignment="center")
        with a1:
            with st.popover("📎 파일 업로드", width="stretch"):
                uploaded = st.file_uploader("AIVLE School 백서(PDF)", type="pdf")
                if uploaded is not None:
                    build_chain(uploaded)
                    st.success("적용 완료")
        a2.caption("백서 PDF를 올리면 그 내용을 바탕으로 답해요.")

    st.markdown(
        f'<div class="disclaimer">답변은 업로드한 백서를 바탕으로 하며 부정확할 수 있어요. '
        f'Model: {MODEL_NAME}</div>',
        unsafe_allow_html=True,
    )

    question = user_input or ss.pop("pending_question", None)

    with chat_box:
        for idx, msg in enumerate(current["messages"]):
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                if msg["role"] == "assistant":
                    st.feedback("thumbs", key=f"fb_{ss.current_id}_{idx}")

        if regenerate:
            last_q = next(m["content"] for m in reversed(current["messages"]) if m["role"] == "user")
            with st.spinner("답변을 다시 만드는 중..."):
                new_answer = ask(last_q)
            if current["messages"][-1]["role"] == "assistant":
                current["messages"][-1]["content"] = new_answer
            st.rerun()   # (다시 생성은 같은 질문이므로 TOP3 횟수에 더하지 않음)

        if question:
            # 첫 질문을 히스토리 제목으로 사용
            if not current["messages"]:
                current["title"] = question[:20]

            current["messages"].append({"role": "user", "content": question})
            with st.chat_message("user"):
                st.markdown(question)

            with st.chat_message("assistant"):
                with st.spinner("답변을 찾는 중..."):
                    answer = ask(question)
                st.markdown(answer)

            current["messages"].append({"role": "assistant", "content": answer})
            record_question(question, answer)   # [추가] TOP3 집계에 이 질문 반영
            st.rerun()   # 히스토리 제목·추천 질문 TOP3를 바로 갱신
