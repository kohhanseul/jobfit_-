# 빌드: docker build -t jobfit-api .
# 실행: docker run -d -p 8000:8000 -e OLLAMA_BASE_URL=http://host.docker.internal:11434 jobfit-api

# 파이썬 3.11 깔린 가벼운 리눅스 베이스
FROM python:3.11-slim

# 컨테이너 안 작업 폴더
WORKDIR /app

# 의존성 먼저 설치 (코드보다 먼저 복사해야 코드만 바꿨을 때 이 단계 캐시로 건너뜀 -> 재빌드 빨라짐)
COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

# 코드, 데이터 복사
COPY 10_api.py .
COPY data/jobs.txt data/jobs.txt

# 8000번 포트 사용 표시
EXPOSE 8000

# host 0.0.0.0 필수 (기본 127.0.0.1이면 컨테이너 밖에서 접속 안 됨)
CMD ["uvicorn", "10_api:app", "--host", "0.0.0.0", "--port", "8000"]
