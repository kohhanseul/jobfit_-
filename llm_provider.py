# 로컬(Ollama)과 클라우드(HF Spaces) 모드를 자동 전환
#
# 판단 기준: GROQ_API_KEY 환경변수가 있으면 클라우드 모드, 없으면 로컬 모드
#   - 내 컴퓨터: 키 없음 -> Ollama (오프라인, 무료)
#   - HF Spaces: 키를 비밀 설정에 등록 -> 임베딩은 서버 CPU, LLM은 Groq
#
# 스토리: 상용 유료 API(Gemini)는 안 씀. 오픈소스 모델을 로컬은 Ollama,
#        클라우드는 Groq 무료 티어로 호스팅. 배포 대상에 따라 provider만 바뀜.

import os

# 키 존재 여부가 곧 모드 신호
IS_CLOUD = bool(os.getenv("GROQ_API_KEY"))


def get_embeddings():
    # 검색용 임베딩. 두 모드 다 같은 모델(bge-m3) -> hit@1 90% 유지
    if IS_CLOUD:
        # 클라우드: sentence-transformers로 bge-m3를 서버에서 직접 실행 (API 아님)
        from langchain_huggingface import HuggingFaceEmbeddings
        return HuggingFaceEmbeddings(model_name="BAAI/bge-m3")
    else:
        # 로컬: Ollama의 bge-m3
        from langchain_ollama import OllamaEmbeddings
        return OllamaEmbeddings(model="bge-m3")


def get_llm():
    # 분석용 LLM
    if IS_CLOUD:
        # 클라우드: Groq 무료 API로 오픈 모델 호출 (모델명은 환경변수로 교체 가능)
        # Groq 모델 목록이 바뀌면 console.groq.com 에서 확인 후 GROQ_MODEL 수정
        from langchain_groq import ChatGroq
        model = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
        return ChatGroq(model=model, temperature=0)
    else:
        # 로컬: Ollama의 qwen3
        from langchain_ollama import ChatOllama
        return ChatOllama(model="qwen3", temperature=0)


def mode_label():
    # UI 표시용
    return "클라우드 모드 (bge-m3 + Groq)" if IS_CLOUD else "로컬 모드 (Ollama)"
