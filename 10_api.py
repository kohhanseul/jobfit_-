"""JobFit API 서버 (FastAPI).

Streamlit(사람용 화면)과 별개로, 다른 프로그램이 호출할 수 있는
JSON 창구를 연다.

실행:
    uvicorn 10_api:app --host 0.0.0.0 --port 8000

엔드포인트:
    GET  /health   서버 살아있는지 확인
    POST /search   이력서로 공고 검색만 (빠름, LLM 미사용)
    POST /analyze  검색 + LLM 적합도 분석 (로컬 LLM이라 1~2분)

문서: 서버 실행 후 http://localhost:8000/docs 에서
      FastAPI가 자동 생성한 API 문서와 테스트 화면을 볼 수 있다.
"""

import re

from fastapi import FastAPI
from pydantic import BaseModel, Field

from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate

JOBS_PATH = "data/jobs.txt"
EMBED_MODEL = "bge-m3"
LLM_MODEL = "qwen3"

app = FastAPI(
    title="JobFit API",
    description="이력서와 채용공고 적합도 분석 (RAG, 로컬 LLM)",
    version="0.1.0",
)


def build_vectorstore():
    with open(JOBS_PATH, "r", encoding="utf-8") as f:
        content = f.read()
    jobs = [j.strip() for j in content.split("---") if j.strip()]
    docs = [Document(page_content=j) for j in jobs]
    embeddings = OllamaEmbeddings(model=EMBED_MODEL)
    return FAISS.from_documents(docs, embeddings)


# 서버 시작 시 한 번만 로드 (요청마다 다시 만들면 느리다)
vectorstore = build_vectorstore()
llm = ChatOllama(model=LLM_MODEL, temperature=0)

analysis_prompt = ChatPromptTemplate.from_messages([
    ("system", """너는 채용공고 분석 전문가야.
아래 채용공고 정보를 바탕으로 지원자의 이력서와 적합도를 분석해줘.

규칙:
- 아래 제공된 공고 중에서만 추천하고, 반드시 공고 번호와 회사명을 인용해
- 지원자의 이력서에 실제로 적힌 경험만 근거로 사용해 (없는 경험을 만들지 마)
- 적합한 공고가 없으면 억지로 연결하지 말고 없다고 말해
- 경력 요건이 지원자와 맞지 않는 공고는 그 사실을 명시해

채용공고 정보:
{context}"""),
    ("human", "내 이력서: {resume}\n\n질문: {question}"),
])


class SearchRequest(BaseModel):
    resume: str = Field(..., description="이력서 요약 텍스트")
    k: int = Field(5, ge=1, le=10, description="반환할 공고 수")


class AnalyzeRequest(BaseModel):
    resume: str = Field(..., description="이력서 요약 텍스트")
    question: str = Field("내 이력서에 맞는 공고 추천해줘", description="분석 요청 문장")
    k: int = Field(5, ge=1, le=10, description="LLM에 전달할 공고 수")


def doc_summary(doc, score=None):
    """공고 본문에서 번호, 회사명, 직무를 뽑아 요약 dict로 만든다."""
    text = doc.page_content
    get = lambda p: (re.search(p, text) or [None, ""])[1]
    item = {
        "id": get(r"\[(공고\d+)\]"),
        "회사명": get(r"회사명: (.+)"),
        "직무": get(r"직무: (.+)"),
        "출처": get(r"출처: (.+)"),
    }
    if score is not None:
        item["distance"] = round(float(score), 4)  # 작을수록 가까움
    return item


@app.get("/health")
def health():
    return {"status": "ok", "postings": vectorstore.index.ntotal,
            "embed_model": EMBED_MODEL, "llm_model": LLM_MODEL}


@app.post("/search")
def search(req: SearchRequest):
    """이력서 내용으로 유사 공고 검색 (LLM 미사용, 즉시 응답)."""
    results = vectorstore.similarity_search_with_score(req.resume, k=req.k)
    return {"results": [doc_summary(d, s) for d, s in results]}


@app.post("/analyze")
def analyze(req: AnalyzeRequest):
    """검색 + LLM 적합도 분석. 로컬 LLM이라 1~2분 걸릴 수 있다."""
    search_query = f"{req.question}\n{req.resume}"
    docs = vectorstore.similarity_search(search_query, k=req.k)
    context = "\n\n".join(d.page_content for d in docs)
    chain = analysis_prompt | llm
    response = chain.invoke({
        "context": context, "resume": req.resume, "question": req.question,
    })
    return {
        "retrieved": [doc_summary(d) for d in docs],
        "analysis": response.content,
    }
