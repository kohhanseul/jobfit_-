# JobFit 배포용 Streamlit 앱 (HF Spaces 진입점)
# 08_streamlit_app.py의 배포 버전. 차이점은 모델을 llm_provider로 받는 것뿐
#   - 로컬 실행: streamlit run app.py  (Ollama)
#   - HF Spaces: GROQ_API_KEY 설정 시 자동으로 클라우드 모드

import base64  # 이미지를 텍스트(문자열)로 인코딩해서 LLM에 보내기 위함
from io import BytesIO  # 붙여넣기된 이미지(PIL)를 바이트로 바꾸기 위함

import streamlit as st

# 클립보드 붙여넣기 부품 (없으면 파일 업로드만 사용)
try:
    from streamlit_paste_button import paste_image_button
    HAS_PASTE = True
except ImportError:
    HAS_PASTE = False
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

from llm_provider import get_embeddings, get_llm, get_vision_llm, mode_label

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

# 공고 직접 분석용 규칙 (텍스트 붙여넣기 / 이미지 업로드 양쪽이 같은 규칙 씀)
# 부족한 점을 짚는 방향이라 "강점 지어내기"보다 할루시네이션이 덜함
FIT_SYSTEM = """너는 채용 적합도 분석 전문가야.
채용공고와 지원자 이력서를 비교해서 분석해줘.

규칙:
- 이력서에 실제로 적힌 내용만 근거로 사용 (없는 경험을 지어내지 마)
- 공고 요구사항을 기준으로, 아래 세 항목으로 나눠서 답해:

## 강점
공고 요건 중 이력서가 충족하는 부분. 공고 문구를 인용하고 이력서의 어떤 경험이 맞는지 연결

## 부족한 점
공고가 요구하는데 이력서에 없거나 약한 부분. 솔직하게

## 보완 제안
부족한 점을 자소서나 준비로 어떻게 메울지 구체적으로 (없는 경험을 만들라는 게 아니라, 가진 것 중 뭘 강조하거나 뭘 배우면 되는지)"""

# 텍스트 붙여넣기용 프롬프트
fit_prompt = ChatPromptTemplate.from_messages([
    ("system", FIT_SYSTEM),
    ("human", "채용공고:\n{posting}\n\n지원자 이력서:\n{resume}")
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

# st.tabs: 한 앱 안에서 기능을 탭으로 나눔
#   탭1 = 공고 추천 (내 이력서 -> DB 49건 중 맞는 공고 검색, RAG 사용)
#   탭2 = 공고 직접 분석 (공고 붙여넣기 -> 이력서와 1:1 비교, 검색/DB 불필요)
# 탭2는 공고 DB가 필요 없어서, 채용 API 없이도 아무 공고나 분석 가능
tab_recommend, tab_fit = st.tabs(["📋 공고 추천", "🔍 공고 직접 분석"])

with tab_recommend:
    st.caption("내 이력서를 넣으면 수집된 공고 49건 중 맞는 걸 찾아줌")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📄 내 이력서")
        resume_r = st.text_area(
            "이력서 내용을 입력하세요",
            placeholder="예: 전자공학 전공, Python/PyTorch, LangChain 기반 RAG 챗봇 개인 프로젝트, EDA와 시계열 분석 경험. AI 엔지니어 신입 지원.",
            height=200,
            key="resume_recommend",  # 탭마다 입력칸이 겹치지 않게 key 지정
        )
    with col2:
        st.subheader("🔍 질문")
        question = st.text_input("질문", value="내 이력서에 맞는 공고 추천해줘", key="q_recommend")
        analyze = st.button("추천 시작 🚀", use_container_width=True, key="btn_recommend")

    if analyze:
        with st.spinner("분석 중..."):
            st.session_state.recommend_result = run_rag(resume_r, question)
    if "recommend_result" in st.session_state:
        st.subheader("📊 추천 결과")
        st.markdown(st.session_state.recommend_result)

with tab_fit:
    st.caption("공고를 붙여넣거나 스크린샷을 올리면, 내 이력서와 비교해 강점/부족한 점/보완법을 알려줌")
    col3, col4 = st.columns(2)
    with col3:
        st.subheader("📌 채용공고")
        posting = st.text_area(
            "공고 내용 붙여넣기",
            placeholder="담당업무, 자격요건, 우대사항 붙여넣기",
            height=140,
            key="posting_fit",
        )
        # 복사 안 되는 공고 대응 (멀티모달 입력): 이미지 바이트를 붙여넣기 또는 업로드로 받음
        img_bytes = None
        img_mime = "image/png"
        if HAS_PASTE:
            st.caption("공고 복사 안 되면: Win+Shift+S로 캡처 후 아래 버튼")
            paste_result = paste_image_button("📋 클립보드 이미지 붙여넣기", key="paste_fit")
            if paste_result.image_data is not None:
                buf = BytesIO()
                paste_result.image_data.save(buf, format="PNG")  # PIL 이미지 -> PNG 바이트
                img_bytes = buf.getvalue()
        posting_img = st.file_uploader(
            "또는 파일로 업로드", type=["png", "jpg", "jpeg"], key="img_fit"
        )
        if posting_img is not None:
            img_bytes = posting_img.getvalue()
            img_mime = posting_img.type
    with col4:
        st.subheader("📄 내 이력서")
        resume_f = st.text_area(
            "이력서 내용을 입력하세요",
            placeholder="내 경험, 기술 스택, 프로젝트를 적기",
            height=260,
            key="resume_fit",
        )
    fit_go = st.button("적합도 분석 🎯", use_container_width=True, key="btn_fit")

    if fit_go:
        with st.spinner("분석 중..."):
            if img_bytes is not None:
                # 이미지 경로: 비전 LLM이 스크린샷을 직접 읽어서 분석 (OCR + 비교 한 번에)
                b64 = base64.b64encode(img_bytes).decode()
                # 멀티모달 메시지 형식: 텍스트 조각 + 이미지 조각을 한 메시지에 리스트로
                messages = [
                    SystemMessage(content=FIT_SYSTEM),
                    HumanMessage(content=[
                        {"type": "text",
                         "text": f"아래 이미지는 채용공고 스크린샷이야. 공고를 읽고 내 이력서와 비교해줘.\n\n지원자 이력서:\n{resume_f}"},
                        {"type": "image_url",
                         "image_url": {"url": f"data:{img_mime};base64,{b64}"}},
                    ]),
                ]
                st.session_state.fit_result = get_vision_llm().invoke(messages).content
            elif posting.strip():
                # 텍스트 경로: 기존 방식
                chain = fit_prompt | llm
                st.session_state.fit_result = chain.invoke(
                    {"posting": posting, "resume": resume_f}
                ).content
            else:
                st.warning("공고를 붙여넣거나 스크린샷을 올려주세요")

    if "fit_result" in st.session_state:
        st.markdown(st.session_state.fit_result)

st.info("💡 왼쪽 사이드바에서 궁금한 기술 용어를 바로 물어보세요!")
