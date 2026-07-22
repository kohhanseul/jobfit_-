# JobFit 배포용 Streamlit 앱 (HF Spaces 진입점)
# 08_streamlit_app.py의 배포 버전. 차이점은 모델을 llm_provider로 받는 것뿐
#   - 로컬 실행: streamlit run app.py  (Ollama)
#   - HF Spaces: GROQ_API_KEY 설정 시 자동으로 클라우드 모드

import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, AIMessage

from llm_provider import get_embeddings, get_llm, mode_label

st.set_page_config(page_title="JobFit - 채용공고 분석기", page_icon="💼", layout="wide")


import os

# jobs.txt 경로 찾기 (data/ 안에 있든 루트에 있든)
def find_jobs_file():
    for p in ("data/jobs.txt", "jobs.txt"):
        if os.path.exists(p):
            return p
    raise FileNotFoundError("jobs.txt를 찾을 수 없음 (data/jobs.txt 또는 jobs.txt)")


# 벡터DB 로드 (서버 시작 시 1회, cache_resource로 재사용)
@st.cache_resource
def load_vectorstore():
    embeddings = get_embeddings()
    with open(find_jobs_file(), "r", encoding="utf-8") as f:
        content = f.read()
    job_list = [job.strip() for job in content.split("---") if job.strip()]
    documents = [Document(page_content=job) for job in job_list]
    return FAISS.from_documents(documents, embeddings)


with st.spinner("벡터DB 로딩 중... (클라우드 첫 실행은 임베딩 모델 다운로드로 2~3분)"):
    vectorstore = load_vectorstore()


@st.cache_resource
def load_llm():
    return get_llm()


llm = load_llm()

analysis_prompt = ChatPromptTemplate.from_messages([
    ("system", """너는 채용공고 분석 전문가야.
아래 채용공고 정보를 바탕으로 지원자의 이력서와 적합도를 분석해줘.

규칙:
- 아래 제공된 공고 중에서만 추천하고, 반드시 공고 번호와 회사명을 인용해
- 지원자의 이력서에 실제로 적힌 경험만 근거로 사용해 (없는 경험을 만들지 마)
- 적합한 공고가 없으면 억지로 연결하지 말고 없다고 말해
- 경력 요건이 지원자와 맞지 않는 공고(예: 신입인데 경력 3년 이상)는 그 사실을 명시해

채용공고 정보:
{context}"""),
    ("human", "내 이력서: {resume}\n\n질문: {question}")
])

glossary_prompt = ChatPromptTemplate.from_messages([
    ("system", """너는 AI/개발 기술 용어 전문가야.
사용자가 묻는 기술 용어나 프레임워크를 쉽고 간결하게 설명해줘.
- 3~5문장으로 핵심만 설명
- 가능하면 실제 사용 예시 한 가지 포함
- 어려운 말 없이 쉽게 설명"""),
    ("human", "{question}")
])


def run_rag(resume, question):
    # 질문+이력서로 검색 (질문만으로는 직무 정보가 없음)
    search_query = f"{question}\n{resume}"
    docs = vectorstore.similarity_search(search_query, k=5)
    context = "\n\n".join(doc.page_content for doc in docs)
    chain = analysis_prompt | llm
    response = chain.invoke({"context": context, "resume": resume, "question": question})
    return response.content


# 사이드바 - 기술 용어 도우미
with st.sidebar:
    st.header("🤖 기술 용어 도우미")
    st.caption("궁금한 기술 용어를 물어보세요!")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    for message in st.session_state.chat_history:
        if isinstance(message, HumanMessage):
            with st.chat_message("user"):
                st.write(message.content)
        else:
            with st.chat_message("assistant"):
                st.write(message.content)

    user_input = st.chat_input("예: RAG가 뭐야? FAISS는?")

    if user_input:
        st.session_state.chat_history.append(HumanMessage(content=user_input))
        with st.spinner("답변 생성 중..."):
            chain = glossary_prompt | llm
            answer = chain.invoke({"question": user_input}).content
        st.session_state.chat_history.append(AIMessage(content=answer))
        st.rerun()

    if st.session_state.chat_history:
        if st.button("대화 초기화 🗑️", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()

# 메인 UI
st.title("💼 JobFit - 채용공고 분석기")
st.caption(f"이력서와 채용공고 적합도 분석  ·  실행: {mode_label()}")

col1, col2 = st.columns(2)
with col1:
    st.subheader("📄 내 이력서")
    resume = st.text_area(
        "이력서 내용을 입력하세요",
        placeholder="예: 전자공학 전공, Python/PyTorch, LangChain 기반 RAG 챗봇 개인 프로젝트, EDA와 시계열 분석 경험. AI 엔지니어 신입 지원.",
        height=200,
    )
with col2:
    st.subheader("🔍 질문")
    question = st.text_input("질문을 입력하세요", value="내 이력서에 맞는 공고 추천해줘")
    analyze = st.button("분석 시작 🚀", use_container_width=True)

if analyze:
    with st.spinner("분석 중..."):
        st.session_state.analysis_result = run_rag(resume, question)

if "analysis_result" in st.session_state:
    st.subheader("📊 분석 결과")
    st.markdown(st.session_state.analysis_result)
    st.info("💡 왼쪽 사이드바에서 궁금한 기술 용어를 바로 물어보세요!")
