from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage

# =============================================
# 1. Ollama 모델 초기화
# =============================================
# ChatOllama: LangChain에서 Ollama 모델을 사용하기 위한 클래스
# model: ollama list에서 확인한 모델명
# temperature: 0이면 일관된 답변, 1에 가까울수록 창의적인 답변
llm = ChatOllama(model="qwen3", temperature=0)

# =============================================
# 2. 기본 호출
# =============================================
response = llm.invoke([
    HumanMessage(content="AI 엔지니어에게 필요한 역량 3가지를 짧게 알려줘")
])

print("=== LangChain + Ollama 기본 호출 ===")
print(response.content)

# =============================================
# 3. 프롬프트 템플릿 + Ollama 연동
# =============================================
# 02_prompt.py에서 Gemini로 했던 것과 동일한 구조
# 모델만 Ollama로 바꾼 것
prompt = ChatPromptTemplate.from_messages([
    ("system", "너는 채용공고 분석 전문가야. 지원자의 이력서와 채용공고를 비교해서 적합도를 분석해줘."),
    ("human", "채용공고: {job_description}\n\n내 이력서 요약: {resume}")
])

chain = prompt | llm

response2 = chain.invoke({
    "job_description": "AI 엔지니어 모집. Python, 딥러닝 경험 필수. 로컬 LLM 활용 경험 우대.",
    "resume": "전자공학 전공. XGBoost, LSTM, CNN 구현 경험. Gemini API 연동 프로젝트 수행. Ollama 로컬 LLM 구동 경험. 부트캠프 대상 수상."
})

print("\n=== 프롬프트 템플릿 + Ollama ===")
print(response2.content)