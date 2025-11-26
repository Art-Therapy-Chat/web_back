# Runpod GPU 환경용 Dockerfile
FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04

# 작업 디렉토리 설정
WORKDIR /web_back

# Python 패키지 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 소스 코드 복사
COPY . .

# 포트 노출
EXPOSE 8888

# 서버 시작 명령
CMD ["uvicorn", "multi_main:app", "--host", "0.0.0.0", "--port", "8888"]
