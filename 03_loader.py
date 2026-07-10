
#.env 파일에서 API키 읽어오는 라이브러리
from dotenv import load_dotenv
# 파일을 읽어서 랭체인이 처리할 수 있는 형태로 변환하는 라이브러리
from langchain_core.documents import Document

load_dotenv()# 키 불러오기

# 1. 문서 로더 - 텍스트 파일 읽기
with open("data/jobs.txt", "r", encoding="utf-8") as f:
    content = f.read()
# 2. --- 기준으로 직접 나누기
job_list = [job.strip() for job in content.split("---") if job.strip()]

# 3. Document 객체로 변환
documents = [Document(page_content=job) for job in job_list]

print(f"=== 문서 로드 완료 ===")
print(f"청크 수: {len(documents)}")
for i, doc in enumerate(documents):
    print(f"\n[공고 {i+1}]\n{doc.page_content}")