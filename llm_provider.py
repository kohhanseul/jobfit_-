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
    # 검색용 임베딩
    #   로컬: bge-m3 (Ollama)                  -> hit@1 90%
    #   클라우드: multilingual-e5-small        -> hit@1 90% (09_eval 재측정)
    #     bge-m3(2.3GB)는 Streamlit Cloud 메모리에 안 들어가서 470MB짜리로 교체
    #     e5 계열은 query:/passage: 프리픽스가 있어야 이 성능이 나옴 -> 아래서 처리
    if IS_CLOUD:
        from langchain_huggingface import HuggingFaceEmbeddings
        from langchain_core.embeddings import Embeddings

        base = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-small")

        class E5Prefixed(Embeddings):
            # 검색은 벡터DB가 embed_query/embed_documents를 대신 불러줌 -> 여기서 프리픽스 자동 부착
            def embed_documents(self, texts):
                return base.embed_documents([f"passage: {t}" for t in texts])

            def embed_query(self, text):
                return base.embed_query(f"query: {text}")

        return E5Prefixed()
    else:
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
