"""JobFit API 서버 (FastAPI)

스트림릿은 사람용 화면, 이건 프로그램용 JSON 창구. 주방(검색+LLM)은 공유

실행: uvicorn 10_api:app --host 0.0.0.0 --port 8000
  - uvicorn = FastAPI 앱 돌려주는 웹서버
  - "10_api:app" = 10_api.py 안의 app 객체 실행하라는 뜻

엔드포인트
  GET  /health   서버 생존 확인
  POST /search   이력서로 공고 검색만 (LLM 안 씀, 0.4초)
  POST /analyze  검색 + LLM 적합도 분석 (로컬 LLM이라 1~3분)
  GET /docs 에서 자동 생성 문서 + 테스트 가능

GET vs POST: GET은 조회, POST는 데이터(이력서)를 몸통에 담아 보냄
"""

import os  # 환경변수 읽기용
import re  # 정규표현식. [공고12] 같은 패턴 찾기

from fastapi import FastAPI
# pydantic = 요청 양식 검사기. 양식에 안 맞는 요청은 내 코드 실행 전에 자동 거절됨
from pydantic import BaseModel, Field

from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate

# 설정값 모음. 바꿀 일 있으면 여기만 수정
JOBS_PATH = "data/jobs.txt"
EMBED_MODEL = "bge-m3"  # 한국어 검색 때문에 nomic에서 교체 (09_eval.py 결과 hit@1 40->90%)
LLM_MODEL = "qwen3"

# ollama 주소. 도커 컨테이너 안에서는 localhost = 컨테이너 자신이라 ollama 못 찾음
# -> 환경변수로 주입 가능하게. 도커 실행 시 -e OLLAMA_BASE_URL=http://host.docker.internal:11434
# (host.docker.internal = 컨테이너를 실행한 호스트 컴퓨터를 가리키는 도커 특수 주소)
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

app = FastAPI(
    title="JobFit API",
    description="이력서와 채용공고 적합도 분석 (RAG, 로컬 LLM)",
    version="0.1.0",
)


def build_vectorstore():
    # jobs.txt 읽기 -> '---' 기준으로 공고 단위 청킹 -> 임베딩 -> FAISS 색인
    with open(JOBS_PATH, "r", encoding="utf-8") as f:
        content = f.read()
    jobs = [j.strip() for j in content.split("---") if j.strip()]  # 빈 조각 제거
    docs = [Document(page_content=j) for j in jobs]
    embeddings = OllamaEmbeddings(model=EMBED_MODEL, base_url=OLLAMA_BASE_URL)
    return FAISS.from_documents(docs, embeddings)


# 서버 시작할 때 1번만 로드. 요청마다 새로 만들면 매번 1분씩 걸림
# (스트림릿의 @st.cache_resource와 같은 목적)
vectorstore = build_vectorstore()

# temperature=0 -> 같은 입력에 최대한 같은 답 (분석 도구는 일관성이 중요)
llm = ChatOllama(model=LLM_MODEL, temperature=0, base_url=OLLAMA_BASE_URL)

# 규칙 4줄 = 3차 개선에서 추가. LLM이 없는 경험 지어내거나 억지 추천하는 것 방지용
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


# 요청 양식 정의 (pydantic)
# Field(...) 의 ... = 필수값. Field(5, ge=1, le=10) = 기본 5, 1~10 범위 강제
# 범위 벗어나면 (k=100) fastapi가 알아서 422 에러 반환
class SearchRequest(BaseModel):
    resume: str = Field(..., description="이력서 요약 텍스트")
    k: int = Field(5, ge=1, le=10, description="반환할 공고 수")


class AnalyzeRequest(BaseModel):
    resume: str = Field(..., description="이력서 요약 텍스트")
    question: str = Field("내 이력서에 맞는 공고 추천해줘", description="분석 요청 문장")
    k: int = Field(5, ge=1, le=10, description="LLM에 전달할 공고 수")


def find_field(text, pattern):
    # re.search: 패턴 찾으면 매치 객체, 못 찾으면 None. 괄호 부분이 group(1)
    match = re.search(pattern, text)
    if match:
        return match.group(1)
    return ""


def doc_summary(doc, score=None):
    # 공고 전문은 너무 기니까 응답용으로 번호/회사명/직무/출처만 추림
    text = doc.page_content
    item = {
        "id": find_field(text, r"\[(공고\d+)\]"),
        "회사명": find_field(text, r"회사명: (.+)"),
        "직무": find_field(text, r"직무: (.+)"),
        "출처": find_field(text, r"출처: (.+)"),
    }
    if score is not None:
        item["distance"] = round(float(score), 4)  # faiss score는 거리. 작을수록 유사
    return item


# @app.get("/health") = 데코레이터. GET으로 /health 요청 오면 아래 함수 실행
# 반환한 dict는 fastapi가 알아서 JSON으로 변환

@app.get("/health")
def health():
    return {"status": "ok", "postings": vectorstore.index.ntotal,
            "embed_model": EMBED_MODEL, "llm_model": LLM_MODEL}


@app.post("/search")
def search(req: SearchRequest):
    # req: SearchRequest 라고 타입 적으면 fastapi가 자동 검증 + 객체 변환해줌
    results = vectorstore.similarity_search_with_score(req.resume, k=req.k)
    return {"results": [doc_summary(d, s) for d, s in results]}


@app.post("/analyze")
def analyze(req: AnalyzeRequest):
    # 핵심: 질문만으로 검색하면 안 됨 ("추천해줘"에는 직무 정보가 없음)
    # -> 질문+이력서 합쳐서 검색. 이력서가 진짜 검색 신호 (3차 개선)
    search_query = f"{req.question}\n{req.resume}"
    docs = vectorstore.similarity_search(search_query, k=req.k)
    context = "\n\n".join(d.page_content for d in docs)

    # 체인 = 프롬프트에 값 채워서 -> LLM 실행. | 로 연결, invoke가 실행
    chain = analysis_prompt | llm
    response = chain.invoke({
        "context": context, "resume": req.resume, "question": req.question,
    })

    # 검색된 공고 목록도 같이 반환 (LLM 답변만 주면 근거 확인 불가)
    return {
        "retrieved": [doc_summary(d) for d in docs],
        "analysis": response.content,
    }
