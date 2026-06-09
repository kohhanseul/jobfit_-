from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()
#언어모델(gemini-2.5-flash)
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

# 프롬프트 템플릿 정의(틀만 만들고 값만 바꿀수있음)
prompt = ChatPromptTemplate.from_messages([
    ("system", "너는 채용공고 분석 전문가야. 지원자의 이력서와 채용공고를 비교해서 적합도를 분석해줘."),
    ("human", "채용공고: {job_description}\n\n내 이력서 요약: {resume}")
])

# 템플릿에 실제 값 넣기
chain = prompt | llm

response = chain.invoke({
    "job_description": "AI 엔지니어 모집. Python, 딥러닝 경험 필수. LLM 활용 경험 우대.",
    "resume": "전자공학 전공. XGBoost, LSTM, CNN 구현 경험. Gemini API 연동 프로젝트 수행. 부트캠프 대상 수상."
})

print(response.content)