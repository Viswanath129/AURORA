@echo off
echo Starting AURORA Dashboard...
cd dashboard
start http://localhost:8000
python -m http.server 8000
pause
