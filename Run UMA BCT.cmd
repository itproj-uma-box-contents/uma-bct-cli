@echo off
setlocal
"%SystemRoot%\System32\WindowsPowerShell\v1.0\powershell.exe" -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0run-uma-bct.ps1"
set "uma_bct_status=%ERRORLEVEL%"
if not "%uma_bct_status%"=="0" pause
exit /b %uma_bct_status%
