# Streamlit Cloud의 sqlite 버전 문제 대비 (Colab에서는 영향 없음)
try:
    __import__("pysqlite3"); import sys; sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
except ImportError:
    pass

import os
import warnings
from operator import itemgetter

warnings.filterwarnings(action="ignore")

import pymupdf
pymupdf.set_messages(path=os.devnull)

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_pymupdf4llm import PyMuPDF4LLMLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma


# 문서 포맷터
def format_docs(docs):
    return "\n\n".join(d.page_content for d in docs)


# RAG 파이프라인 함수
def rag_pipeline(filepath):

    # PDF 파일 읽기
    loader = PyMuPDF4LLMLoader(filepath)
    docs = loader.load()

    # 텍스트 분할
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
    splits = text_splitter.split_documents(docs)

    # Embedding 모델
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

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

    # 검색기 구성
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    # LLM 선언
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    # 프롬프트 구성
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "당신은 KT 에이블스쿨 학습 도우미입니다. "
         "아래 [문서] 내용만 근거로 질문에 답하세요. "
         "문서에 없는 내용은 추측하지 말고 '백서에서 해당 내용을 찾을 수 없습니다.'라고 답하세요.\n\n"
         "[문서]\n{context}"),
        ("human", "{input}"),
    ])

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
