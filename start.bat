@echo off
REM MangaLens — Windows 원클릭 실행 스크립트 (WSL2 경유)
REM 사용법: start.bat [PORT]
setlocal

set "PORT=20399"
if not "%~1"=="" set "PORT=%~1"

REM WSL에서 start.sh 실행
wsl -e bash -c "cd /home/user/test0320 && ./start.sh --port %PORT%"

if %ERRORLEVEL% NEQ 0 (
    echo [Error] 서버 실행에 실패했습니다.
    pause
)
endlocal
