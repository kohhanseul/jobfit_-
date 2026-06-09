import ollama

# =============================================
# 1. 기본 호출 - 가장 단순한 방식
# =============================================
# ollama.chat(): 로컬에서 실행 중인 LLM에 메시지를 보내고 응답을 받음
# model: 사용할 모델명 (ollama list에서 확인한 이름)
# messages: 대화 내역 (role은 user/assistant/system 세 가지)
response = ollama.chat(
    model="qwen3",
    messages=[
        {
            "role": "user",
            "content": "AI 엔지니어에게 필요한 역량 3가지를 짧게 알려줘"
        }
    ]
)

# response.message.content: 모델이 생성한 텍스트 응답
print("=== Ollama 기본 호출 ===")
print(response.message.content)

# =============================================
# 2. 시스템 프롬프트 추가
# =============================================
# system 역할: LLM에게 역할/성격을 부여하는 메시지
# user보다 먼저 넣어야 효과가 있음
response2 = ollama.chat(
    model="qwen3",
    messages=[
        {
            "role": "system",
            "content": "너는 채용공고 분석 전문가야. 간결하게 답변해줘."
        },
        {
            "role": "user",
            "content": "Python과 딥러닝 경험이 필요한 직무는 어떤 게 있어?"
        }
    ]
)

print("\n=== 시스템 프롬프트 추가 ===")
print(response2.message.content)