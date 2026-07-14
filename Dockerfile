# Dockerfile = 도시락 싸는 레시피.
# 위에서 아래로 순서대로 실행되며, 결과물이 이미지(밀봉된 도시락)가 된다.
#
# 빌드:  docker build -t jobfit-api .
#        (-t jobfit-api: 이미지에 이름 붙이기 / 마지막 점: 현재 폴더의 레시피 사용)
# 실행:  docker run -p 8000:8000 -e OLLAMA_BASE_URL=http://host.docker.internal:11434 jobfit-api
#        (-p 8000:8000: 컨테이너 안 8000번 문을 내 컴퓨터 8000번에 연결)

# 1. 바닥 재료: 파이썬 3.11이 미리 깔린 가벼운(slim) 리눅스
FROM python:3.11-slim

# 2. 컨테이너 안에서의 작업 폴더 지정 (없으면 만들어줌)
WORKDIR /app

# 3. 의존성 목록만 먼저 복사해서 설치.
#    코드보다 먼저 하는 이유: 도커는 단계별로 결과를 캐시하는데,
#    코드만 고치고 다시 빌드할 때 이 설치 단계를 건너뛸 수 있어 빨라진다.
COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

# 4. 코드와 데이터를 도시락에 담기
COPY 10_api.py .
COPY data/jobs.txt data/jobs.txt

# 5. 이 도시락은 8000번 문을 쓴다고 표시 (문서 역할)
EXPOSE 8000

# 6. 도시락을 열면(run) 실행할 명령
#    host 0.0.0.0: 컨테이너 밖에서 들어오는 요청도 받겠다는 뜻
#    (기본값 127.0.0.1이면 컨테이너 안에서만 접속 가능해서 밖에서 못 씀)
CMD ["uvicorn", "10_api:app", "--host", "0.0.0.0", "--port", "8000"]
