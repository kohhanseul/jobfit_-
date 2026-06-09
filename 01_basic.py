from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage

load_dotenv()
# LangChain 있으면 → 모델만 바꿔도 코드 구조 동일
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

response = llm.invoke([HumanMessage(content="AI 엔지니어에게 필요한 역량 3가지를 알려줘")])#보내는 메시지

print(response.content)#받는 메시지