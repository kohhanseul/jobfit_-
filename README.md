# JobFit - 채용공고 분석 RAG 챗봇

이력서와 채용공고를 비교해 적합도를 분석해주는 AI 챗봇입니다.
단순 키워드 매칭이 아닌 의미 기반 검색(RAG)으로 관련 공고를 찾고,
로컬 LLM(Ollama)을 활용해 API 비용 없이 동작합니다.

## 기술 스택

**Language** Python

**LLM** Ollama (Qwen3) · Gemini API

**Framework** LangChain · Streamlit

**Vector DB** FAISS

**Embedding** nomic-embed-text

##  프로젝트 구조

- 01_basic.py : LangChain 기본 호출
- 02_prompt.py : 프롬프트 템플릿
- 03_loader.py : 문서 로더 및 청킹
- 04_vectordb.py : FAISS 벡터DB 생성
- 05_rag.py : RAG 체인 구성
- 06_ollama.py : Ollama 기본 호출
- 07_langchain_ollama.py : LangChain + Ollama 연동
- 08_streamlit_app.py : Streamlit UI
- data/jobs.txt : 채용공고 데이터

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