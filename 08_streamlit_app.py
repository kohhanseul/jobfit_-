import streamlit as st
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, AIMessage

# =============================================
# 1. 페이지 설정
# =============================================
st.set_page_config(page_title="JobFit - 채용공고 분석기", page_icon="💼", layout="wide")

# =============================================
# 2. 벡터DB 로드
# =============================================
@st.cache_resource
def load_vectorstore():
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    with open("data/jobs.txt", "r", encoding="utf-8") as f:
        content = f.read()
    job_list = [job.strip() for job in content.split("---") if job.strip()]
    documents = [Document(page_content=job) for job in job_list]
    return FAISS.from_documents(documents, embeddings)

with st.spinner("벡터DB 로딩 중..."):
    vectorstore = load_vectorstore()

# =============================================
# 3. LLM 초기화
# =============================================
llm = ChatOllama(model="qwen3", temperature=0)

# =============================================
# 4. 프롬프트 템플릿 (공고 분석용)
# =============================================
analysis_prompt = ChatPromptTemplate.from_messages([
    ("system", """너는 채용공고 분석 전문가야.
아래 채용공고 정보를 바탕으로 지원자의 이력서와 적합도를 분석해줘.

채용공고 정보:
{context}"""),
    ("human", "내 이력서: {resume}\n\n질문: {question}")
])

# =============================================
# 5. 용어 설명 프롬프트 (사이드바 챗봇용)
# =============================================
glossary_prompt = ChatPromptTemplate.from_messages([
    ("system", """너는 AI/개발 기술 용어 전문가야.
사용자가 묻는 기술 용어나 프레임워크를 쉽고 간결하게 설명해줘.
- 3~5문장으로 핵심만 설명
- 가능하면 실제 사용 예시 한 가지 포함
- 어려운 말 없이 쉽게 설명"""),
    ("human", "{question}")
])

# =============================================
# 6. RAG 실행 함수
# =============================================
def run_rag(resume, question):
    docs = vectorstore.similarity_search(question, k=2)
    context = "\n\n".join(doc.page_content for doc in docs)
    chain = analysis_prompt | llm
    response = chain.invoke({
        "context": context,
        "resume": resume,
        "question": question
    })
    return response.content

# =============================================
# 7. 사이드바 - 기술 용어 도우미
# =============================================
with st.sidebar:
    st.header("🤖 기술 용어 도우미")
    st.caption("궁금한 기술 용어를 물어보세요!")

    # 채팅 기록 초기화
    # session_state: 버튼 클릭/새로고침 후에도 값을 유지하는 Streamlit 저장소
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # 채팅 기록 표시
    for message in st.session_state.chat_history:
        if isinstance(message, HumanMessage):
            with st.chat_message("user"):
                st.write(message.content)
        else:
            with st.chat_message("assistant"):
                st.write(message.content)

    # 입력창
    # st.chat_input: 채팅 입력창 (엔터로 전송)
    user_input = st.chat_input("예: RAG가 뭐야? FAISS는?")

    if user_input:
        # 사용자 메시지 기록에 추가
        st.session_state.chat_history.append(HumanMessage(content=user_input))

        # LLM 호출
        with st.spinner("답변 생성 중..."):
            chain = glossary_prompt | llm
            response = chain.invoke({"question": user_input})
            answer = response.content

        # AI 응답 기록에 추가
        st.session_state.chat_history.append(AIMessage(content=answer))

        # 화면 새로고침
        st.rerun()

    # 대화 초기화 버튼
    if st.session_state.chat_history:
        if st.button("대화 초기화 🗑️", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()

# =============================================
# 8. 메인 UI
# =============================================
st.title("💼 JobFit - 채용공고 분석기")
st.caption("이력서와 채용공고를 비교해 적합도를 분석해드립니다")

col1, col2 = st.columns(2)

with col1:
    st.subheader("📄 내 이력서")
    resume = st.text_area(
    "이력서 내용을 입력하세요",
    placeholder="여기에 이력서 내용을 입력하세요\n예: 전자공학 전공, Python 경험, 딥러닝 프로젝트 수행 등",
    height=200  
    )
with col2:
    st.subheader("🔍 질문")
    question = st.text_input(
        "질문을 입력하세요",
        value="내 이력서에 맞는 공고 추천해줘"
    )
    analyze = st.button("분석 시작 🚀", use_container_width=True)

# =============================================
# 9. 분석 결과 출력
# =============================================

# 분석 결과 session_state에 저장
if analyze:
    with st.spinner("분석 중... (로컬 LLM이라 30초~1분 걸릴 수 있어요)"):
        result = run_rag(resume, question)
    # 결과를 session_state에 저장 → 사이드바 챗봇 사용해도 유지됨
    st.session_state.analysis_result = result

# session_state에 결과가 있으면 항상 표시
if "analysis_result" in st.session_state:
    st.subheader("📊 분석 결과")
    st.markdown(st.session_state.analysis_result)
    st.info("💡 왼쪽 사이드바에서 궁금한 기술 용어를 바로 물어보세요!")