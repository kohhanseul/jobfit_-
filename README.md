# JobFit - 채용공고 분석 RAG 챗봇

**▶ 라이브 데모: https://p9kr69lbjsctwed7azrt66.streamlit.app/**
(이력서 넣고 "분석 시작" 누르면 실제 공고 49건 중 적합 공고 추천)

이력서와 채용공고 비교해서 적합도 분석해주는 챗봇
키워드 매칭이 아니라 의미 기반 검색(RAG)으로 관련 공고 찾음
로컬은 Ollama로 완전 오프라인, 배포 데모는 오픈모델 무료 호스팅(Groq)으로 실행

## 배포 (2모드)

같은 코드가 환경변수(GROQ_API_KEY) 유무로 자동 전환됨

| | 임베딩 | LLM | 비고 |
|---|---|---|---|
| 로컬 | bge-m3 (Ollama) | qwen3 (Ollama) | 완전 오프라인, hit@1 90% |
| 클라우드 (Streamlit) | multilingual-e5-small | Groq (오픈모델) | 무료 티어 메모리 대응, hit@1 90% 재측정 |

상용 유료 API는 안 씀. 오픈소스 모델을 로컬은 Ollama, 클라우드는 Groq 무료 티어로 호스팅

## 기술 스택

**Language** Python

**LLM** Ollama (Qwen3), Gemini API

**Framework** LangChain, Streamlit, FastAPI

**Vector DB** FAISS

**Embedding** bge-m3 (한국어 검색 문제로 nomic-embed-text에서 교체. 아래 평가 참고)

## 프로젝트 구조

- 01_basic.py : LangChain 기본 호출
- 02_prompt.py : 프롬프트 템플릿
- 03_loader.py : 문서 로더 및 청킹
- 04_vectordb.py : FAISS 벡터DB 생성
- 05_rag.py : RAG 체인 구성
- 06_ollama.py : Ollama 기본 호출
- 07_langchain_ollama.py : LangChain + Ollama 연동
- 08_streamlit_app.py : Streamlit UI
- 09_eval.py : 검색 품질 평가 (hit@1, hit@k 측정)
- 10_api.py : FastAPI 서빙 API (/search, /analyze)
- data/jobs.txt : 실제 채용공고 49건 (2026-07 사람인 수집, 출처 URL 포함)
- data/eval_set.json : 검색 평가용 질문-정답 10문항
- docs/ : 개선 기록 (문제 -> 해결 -> 결과)

## 검색 품질 평가

실제 공고 49건(AI 엔지니어, 데이터 사이언티스트, 타 직군 섞음)으로
질문-정답 평가셋 10문항 만들어서 검색 정확도 측정

```bash
python 09_eval.py --embed-model bge-m3 --show
```

| 임베딩 모델 | hit@1 | hit@3 |
|---|---|---|
| nomic-embed-text | 40% | 50% |
| **bge-m3 (채택)** | **90%** | **100%** |

- nomic은 한국어 질문("영상 편집자 공고 있어?")이 뻔한 정답 공고랑도 매칭 안 되는 문제 있음
- 다국어 임베딩 bge-m3로 교체해서 해결
- 평가셋에 함정 문항 포함 (AI 툴 쓰는 영상편집 공고 vs 진짜 AI 개발 공고 구분되는지)

## 설치 및 실행

1. 패키지 설치

```bash
pip install langchain langchain-ollama langchain-community faiss-cpu streamlit fastapi uvicorn python-dotenv
```

2. Ollama 모델 다운로드

```bash
ollama pull qwen3
ollama pull bge-m3
```

3. 실행

```bash
streamlit run 08_streamlit_app.py      # 화면(UI)
uvicorn 10_api:app --port 8000         # API 서버 (문서: localhost:8000/docs)
```

## Docker로 실행

api 서버 컨테이너로 실행 가능 (로컬 ollama 켜진 상태에서만)
컨테이너 안에서는 localhost가 컨테이너 자신이라 host.docker.internal로 호스트 ollama에 연결

```bash
docker build -t jobfit-api .
docker run -d --name jobfit-container -p 8000:8000 \
  -e OLLAMA_BASE_URL=http://host.docker.internal:11434 jobfit-api
```

실행 후 http://localhost:8000/docs 에서 테스트

## 주요 기능

- **공고 추천** (RAG): 이력서 넣으면 수집된 공고 49건 중 맞는 걸 검색 + 적합도 분석
- **공고 직접 분석**: 관심 공고를 붙여넣으면 내 이력서와 비교해 강점/부족한 점/보완 제안
  - 공고 DB가 필요 없어서 아무 공고나 분석 가능 (개인회원 채용 API 제약 우회)
  - 복사 안 되는 공고는 **스크린샷 이미지 업로드** -> 비전 LLM이 읽어서 분석 (멀티모달)
- 사이드바에 기술 용어 챗봇 (모르는 용어 바로 질문)
- qwen 계열의 `<think>` 추론 블록은 제거하고 최종 답변만 표시

## 데이터에 대한 메모 (실제 공고 49건인 이유)

- 실시간 대량 수집을 위해 워크넷 채용정보 오픈API 연동을 시도함
- 그런데 워크넷 채용정보목록/상세 API는 **기업회원(사업자) 전용**. 개인회원은
  채용행사, 공채속보만 되고 일반 채용목록 조회는 불가
- 사람인/잡코리아 등 민간 API도 파트너/기업 대상이라 개인 프로젝트는 접근 어려움
- 그래서 실제 공고 49건을 수동 수집(출처 URL 기록)해 파이프라인과 검색 품질(hit@1 90%)을 검증
- 실시간 API 자동 수집은 기업회원 자격이 필요해 다음 단계로 남김

## 향후 계획

- 실시간 공고 API 연동 (기업회원 자격 확보 시)
- 이력서 PDF 업로드 기능
- 할루시네이션 방지를 위한 답변 출처 표시
- (완료) 클라우드 배포 -> Streamlit Cloud 라이브 데모
