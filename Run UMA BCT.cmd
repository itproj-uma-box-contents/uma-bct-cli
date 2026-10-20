@echo off
setlocal DisableDelayedExpansion
rem PowerShell handles interactive prompts; a blanket pause blocks JSON and unattended runs.
"%SystemRoot%\System32\WindowsPowerShell\v1.0\powershell.exe" -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0run-uma-bct.ps1" %*
set "uma_bct_status=%ERRORLEVEL%"
exit /b %uma_bct_status%
