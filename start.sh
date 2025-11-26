#!/bin/bash
# Runpod 시작 스크립트

echo "🚀 Starting HTP Backend Server..."

# 환경 변수 확인
if [ -z "$OPENAI_API_KEY" ]; then
    echo "⚠️  Warning: OPENAI_API_KEY not set"
fi

# Git pull (optional)
if [ -d ".git" ]; then
    echo "📦 Pulling latest code..."
    git pull origin main
fi

# 서버 시작
echo "✅ Starting uvicorn server on port 8888..."
uvicorn multi_main:app --host 0.0.0.0 --port 8888
