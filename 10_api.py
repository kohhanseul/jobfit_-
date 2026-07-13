"""JobFit API 서버 (FastAPI).

[이 파일이 하는 일]
Streamlit(사람이 눈으로 보는 화면)과 별개로, 다른 프로그램이 호출할 수 있는
JSON 창구(API)를 연다. 식당으로 비유하면 Streamlit은 홀 좌석이고,
이 파일은 전화 주문 창구다. 주방(검색 엔진 + LLM)은 둘이 같이 쓴다.

[실행 방법]
    uvicorn 10_api:app --host 0.0.0.0 --port 8000

    - uvicorn: FastAPI 앱을 실제로 구동하는 웹 서버 프로그램
      (FastAPI는 창구 설계도이고, uvicorn이 창구 문을 여는 직원이다)
    - "10_api:app" 의미: 10_api.py 파일 안의 app 이라는 객체를 실행해라
    - --port 8000: 8000번 문(포트)에서 손님을 받아라

[엔드포인트 = 창구 목록]
    GET  /health   서버 살아있는지 확인 (즉시 응답)
    POST /search   이력서로 공고 검색만 (LLM 안 씀, 약 0.4초)
    POST /analyze  검색 + LLM 적합도 분석 (로컬 LLM이라 1~3분)

    GET과 POST의 차이: GET은 "그냥 보여줘"(입력 없이 조회),
    POST는 "이 데이터를 처리해줘"(이력서 같은 입력을 몸통에 담아 보냄)

[자동 문서]
서버 실행 후 브라우저에서 http://localhost:8000/docs 를 열면
FastAPI가 자동으로 만든 API 문서와 테스트 화면이 나온다.
"""

import re  # 정규표현식: 문자열에서 패턴(예: [공고12])을 찾는 도구

# FastAPI: 파이썬으로 API 창구를 만드는 프레임워크
from fastapi import FastAPI

# Pydantic: "주문서 양식"을 정의하는 도구.
# 손님(호출자)이 양식에 안 맞는 주문(예: 이력서 없이 요청)을 보내면
# 내 코드가 실행되기도 전에 FastAPI가 자동으로 거절해준다.
from pydantic import BaseModel, Field

# LangChain 부품들 (Streamlit 앱과 동일한 주방 설비)
from langchain_ollama import ChatOllama, OllamaEmbeddings  # 로컬 LLM, 임베딩
from langchain_community.vectorstores import FAISS          # 벡터 검색 엔진
from langchain_core.documents import Document               # 문서 한 건을 담는 상자
from langchain_core.prompts import ChatPromptTemplate       # 프롬프트 틀

# ── 설정값: 한 곳에 모아두면 나중에 바꿀 때 여기만 고치면 된다 ──
JOBS_PATH = "data/jobs.txt"   # 채용공고 데이터 파일
EMBED_MODEL = "bge-m3"        # 임베딩 모델 (한국어 검색 품질 때문에 선택, 09_eval.py 참고)
LLM_MODEL = "qwen3"           # 분석 답변을 쓰는 로컬 LLM

# FastAPI 앱 객체 = 창구 설계도의 본체.
# title 등은 /docs 자동 문서에 표시된다.
app = FastAPI(
    title="JobFit API",
    description="이력서와 채용공고 적합도 분석 (RAG, 로컬 LLM)",
    version="0.1.0",
)


def build_vectorstore():
    """공고 파일을 읽어 검색 가능한 벡터 저장소(FAISS)를 만든다.

    과정을 풀어 쓰면:
    1. jobs.txt를 통째로 읽는다
    2. '---' 구분자로 잘라 공고 한 건씩 리스트로 만든다 (공고 단위 청킹)
    3. 각 공고를 Document 상자에 담는다
    4. 임베딩 모델이 공고마다 좌표(숫자 벡터)를 찍고,
       FAISS가 그 좌표들을 "가까운 것 빨리 찾기" 가능한 색인으로 저장한다
    """
    with open(JOBS_PATH, "r", encoding="utf-8") as f:
        content = f.read()

    # if j.strip(): 빈 조각(공백뿐인 조각)은 버린다
    jobs = [j.strip() for j in content.split("---") if j.strip()]
    docs = [Document(page_content=j) for j in jobs]

    embeddings = OllamaEmbeddings(model=EMBED_MODEL)
    return FAISS.from_documents(docs, embeddings)


# ── 서버 시작 시 딱 한 번 실행되는 준비 작업 ──
# 요청이 올 때마다 벡터 저장소를 다시 만들면 매번 1분씩 걸린다.
# 그래서 서버가 켜질 때 한 번만 만들어 전역 변수에 담아두고,
# 모든 요청이 이걸 재사용한다. (Streamlit의 @st.cache_resource와 같은 목적)
vectorstore = build_vectorstore()

# temperature=0: LLM의 창의성 조절 손잡이. 0이면 같은 질문에 최대한
# 일관된 답을 낸다. 분석 도구는 들쭉날쭉하면 안 되므로 0으로 고정.
llm = ChatOllama(model=LLM_MODEL, temperature=0)

# 프롬프트 틀: {context}, {resume}, {question} 자리에 실제 값이 채워진다.
# 규칙 4줄은 3차 개선에서 추가한 것 — LLM이 검색 결과를 억지로
# 정당화하거나 없는 경험을 지어내는 것을 막는 안전장치다.
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


# ── 주문서 양식 (Pydantic 모델) ──
# BaseModel을 상속하면 "이 API는 이런 형태의 JSON을 받는다"가 정의된다.
# Field(...) 의 ... 은 "필수 항목"이라는 뜻이고,
# Field(5, ge=1, le=10) 은 "기본값 5, 최소 1, 최대 10"이라는 뜻이다.
# 범위를 벗어난 값(k=100)이 오면 FastAPI가 자동으로 에러를 돌려준다.
class SearchRequest(BaseModel):
    resume: str = Field(..., description="이력서 요약 텍스트")
    k: int = Field(5, ge=1, le=10, description="반환할 공고 수")


class AnalyzeRequest(BaseModel):
    resume: str = Field(..., description="이력서 요약 텍스트")
    question: str = Field("내 이력서에 맞는 공고 추천해줘", description="분석 요청 문장")
    k: int = Field(5, ge=1, le=10, description="LLM에 전달할 공고 수")


def find_field(text, pattern):
    """공고 본문(text)에서 정규식 패턴에 맞는 첫 값을 찾아 돌려준다.

    re.search는 패턴을 찾으면 매치 객체를, 못 찾으면 None을 준다.
    괄호로 감싼 부분이 group(1)에 담긴다.
    예: pattern이 r"회사명: (.+)" 이고 본문에 "회사명: 윕스"가 있으면 "윕스" 반환.
    """
    match = re.search(pattern, text)
    if match:
        return match.group(1)
    return ""  # 못 찾으면 빈 문자열 (에러 대신 조용히 넘어가는 선택)


def doc_summary(doc, score=None):
    """공고 문서에서 핵심 정보만 뽑아 딕셔너리(→JSON)로 만든다.

    공고 본문 전체를 응답에 넣으면 너무 길어서,
    API 응답용으로 번호/회사명/직무/출처만 추린다.
    """
    text = doc.page_content
    item = {
        "id": find_field(text, r"\[(공고\d+)\]"),   # \d+ = 숫자 1개 이상
        "회사명": find_field(text, r"회사명: (.+)"),  # .+ = 아무 글자 1개 이상
        "직무": find_field(text, r"직무: (.+)"),
        "출처": find_field(text, r"출처: (.+)"),
    }
    if score is not None:
        # FAISS의 score는 "거리"라서 작을수록 이력서와 가깝다(더 유사하다)
        item["distance"] = round(float(score), 4)
    return item


# ── 여기서부터 창구(엔드포인트) 정의 ──
# @app.get("/health") 같은 줄을 데코레이터라고 부른다.
# 뜻: "GET 방식으로 /health 주소에 요청이 오면, 바로 아래 함수를 실행해라"
# 함수가 반환하는 딕셔너리는 FastAPI가 알아서 JSON으로 바꿔 보낸다.

@app.get("/health")
def health():
    """서버 생존 확인용. 모니터링 도구나 배포 환경이 주기적으로 찔러본다."""
    return {"status": "ok", "postings": vectorstore.index.ntotal,  # 색인된 공고 수
            "embed_model": EMBED_MODEL, "llm_model": LLM_MODEL}


@app.post("/search")
def search(req: SearchRequest):
    """이력서 내용으로 유사 공고 검색 (LLM 미사용, 즉시 응답).

    req: SearchRequest 라고 타입을 적어두면, FastAPI가 요청 JSON을
    자동으로 검사하고 SearchRequest 객체로 변환해서 넣어준다.
    req.resume, req.k 로 꺼내 쓰면 된다.

    similarity_search_with_score: 검색 결과와 함께 거리 점수도 돌려주는 버전.
    """
    results = vectorstore.similarity_search_with_score(req.resume, k=req.k)
    return {"results": [doc_summary(d, s) for d, s in results]}


@app.post("/analyze")
def analyze(req: AnalyzeRequest):
    """검색 + LLM 적합도 분석. 로컬 LLM이라 1~3분 걸릴 수 있다.

    3차 개선의 핵심이 여기 있다: 검색 쿼리에 질문만 쓰면
    ("추천해줘"에는 직무 정보가 없어서) 엉뚱한 공고가 나온다.
    그래서 질문+이력서를 합쳐 검색한다. 이력서가 검색의 진짜 신호다.
    """
    search_query = f"{req.question}\n{req.resume}"
    docs = vectorstore.similarity_search(search_query, k=req.k)

    # 검색된 공고들을 줄바꿈 두 개로 이어붙여 프롬프트의 {context} 재료로 만든다
    context = "\n\n".join(d.page_content for d in docs)

    # 체인 = 부품 연결. "프롬프트 틀에 값을 채워서 -> LLM에 넣어라"를
    # | (파이프) 기호로 연결한다. invoke가 실행 버튼.
    chain = analysis_prompt | llm
    response = chain.invoke({
        "context": context, "resume": req.resume, "question": req.question,
    })

    # 어떤 공고가 검색됐는지(retrieved)도 함께 돌려준다.
    # LLM 답변만 주면 사용자가 근거를 확인할 수 없기 때문.
    return {
        "retrieved": [doc_summary(d) for d in docs],
        "analysis": response.content,
    }
