@echo off
REM 빠른 배포 스크립트
echo 🚀 Deploying to GitHub...

cd "c:\Users\helen\Desktop\kt cloud tech up\basic_project\web.ver.2\web_back"

git add .
git status

set /p commit_msg="커밋 메시지: "
git commit -m "%commit_msg%"

git push origin main

echo.
echo ✅ 배포 완료!
echo 📌 Runpod에서 다음 명령어 실행:
echo    cd /web_back
echo    git pull origin main
echo    ./start.sh
pause
