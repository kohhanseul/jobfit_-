# JobFit - 채용공고 분석 RAG 챗봇

이력서와 채용공고를 비교해 적합도를 분석해주는 AI 챗봇입니다.
단순 키워드 매칭이 아닌 의미 기반 검색(RAG)으로 관련 공고를 찾고,
로컬 LLM(Ollama)을 활용해 API 비용 없이 동작합니다.

## 기술 스택

**Language** Python

**LLM** Ollama (Qwen3) · Gemini API

**Framework** LangChain · Streamlit

**Vector DB** FAISS

**Embedding** bge-m3 (한국어 검색 품질 문제로 nomic-embed-text에서 교체, 아래 평가 참고)

##  프로젝트 구조

- 01_basic.py : LangChain 기본 호출
- 02_prompt.py : 프롬프트 템플릿
- 03_loader.py : 문서 로더 및 청킹
- 04_vectordb.py : FAISS 벡터DB 생성
- 05_rag.py : RAG 체인 구성
- 06_ollama.py : Ollama 기본 호출
- 07_langchain_ollama.py : LangChain + Ollama 연동
- 08_streamlit_app.py : Streamlit UI
- 09_eval.py : 검색 품질 평가 (hit@1, hit@k 측정)
- data/jobs.txt : 실제 채용공고 33건 (2026-07 사람인 수집, 출처 URL 포함)
- data/eval_set.json : 검색 평가용 질문-정답 10문항

## 검색 품질 평가

실제 채용공고 33건(AI 엔지니어, 데이터 사이언티스트, 타 직군 혼합)으로
질문-정답 평가셋 10문항을 만들어 검색 정확도를 측정했습니다.

```bash
python 09_eval.py --embed-model bge-m3 --show
```

| 임베딩 모델 | hit@1 | hit@3 |
|---|---|---|
| nomic-embed-text | 40% | 50% |
| **bge-m3 (채택)** | **90%** | **100%** |

nomic-embed-text는 한국어 질문("영상 편집자 채용 공고 있어?")이 명백한 정답 공고와
매칭되지 않는 문제가 있었고, 다국어 임베딩 bge-m3로 교체해 해결했습니다.
평가셋에는 직군 혼동을 검증하는 함정 문항(AI 툴을 쓰는 영상편집 공고 vs AI 개발 공고)을 포함했습니다.

## 설치 및 실행

1. 패키지 설치

```bash
pip install langchain langchain-ollama langchain-community faiss-cpu streamlit python-dotenv
```

2. Ollama 모델 다운로드

```bash
ollama pull qwen3
ollama pull nomic-embed-text
```

3. 실행

```bash
streamlit run 08_streamlit_app.py
```

## 주요 기능

- 이력서 입력 후 관련 채용공고 자동 검색 및 적합도 분석
- 의미 기반 검색으로 키워드가 없어도 유사 공고 탐색 가능
- 사이드바 기술 용어 챗봇으로 모르는 용어 즉시 질문 가능
- 로컬 LLM 기반으로 API 비용 없이 동작

## 향후 계획

- 사람인 API 연동으로 실시간 채용공고 수집
- 이력서 PDF 업로드 기능
- 클라우드 배포
- 할루시네이션 방지를 위한 답변 출처 표시