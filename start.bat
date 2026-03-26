@echo off
REM MangaLens — Windows 원클릭 실행 스크립트 (WSL2 경유)
REM 사용법: start.bat [PORT]
setlocal

set "PORT=20399"
if not "%~1"=="" set "PORT=%~1"

REM 현재 .bat 파일의 위치를 WSL 경로로 변환
for %%I in ("%~dp0.") do set "WIN_DIR=%%~fI"
REM Windows 경로를 WSL 경로로 변환 (C:\Users\... → /mnt/c/Users/...)
for /f "usebackq tokens=*" %%A in (`wsl wslpath -u "%WIN_DIR%"`) do set "WSL_DIR=%%A"

REM WSL에서 start.sh 실행
wsl -e bash -c "cd '%WSL_DIR%' && ./start.sh --port %PORT%"

if %ERRORLEVEL% NEQ 0 (
    echo [Error] 서버 실행에 실패했습니다.
    pause
)
endlocal
