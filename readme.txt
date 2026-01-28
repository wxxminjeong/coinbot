=======================================================
[ENGLISH] Setup & Usage Guide
=======================================================

[1] Environment Setup (.env)
Create a '.env' file in the project root and add the following keys without spaces.

BINANCE_API_KEY=your_binance_api_key
BINANCE_SECRET_KEY=your_binance_secret_key
GEMINI_API_KEY=your_google_gemini_api_key
NGROK_AUTH_TOKEN=your_ngrok_token_optional

-------------------------------------------------------

[2] File Overview
- coinbot.py          : Main trading bot logic (AI-driven)
- log_server.py       : Web server for real-time monitoring
- config.json         : Trading settings (Leverage, Amount, SL/TP)
- ecosystem.config.js : PM2 configuration file for process management

-------------------------------------------------------

[3] PM2 Commands (Terminal)

# 1. Start Bot & Log Server (Run based on ecosystem.config.js)
pm2 start ecosystem.config.js

# 2. Monitor Real-time Logs
pm2 monit
   (or 'pm2 logs')

# 3. Restart All Processes (Apply code changes)
pm2 restart all

# 4. Stop & Remove Processes
pm2 stop all
pm2 delete all

-------------------------------------------------------

[4] Dashboard Access
- Local Network: http://localhost:5000
- External Access: Check the Ngrok URL printed in the terminal logs


=======================================================
[KOREAN] 설치 및 사용 가이드
=======================================================

[1] 환경 변수 설정 (.env)
프로젝트 폴더 안에 '.env' 파일을 만들고 아래 키 값을 공백 없이 입력하세요.

BINANCE_API_KEY=내_바이낸스_API_키
BINANCE_SECRET_KEY=내_바이낸스_시크릿_키
GEMINI_API_KEY=내_구글_제미나이_API_키
NGROK_AUTH_TOKEN=내_NGROK_토큰_입력(선택사항)

-------------------------------------------------------

[2] 파일 구성 확인
- coinbot.py          : 실제 매매를 수행하는 봇 (메인)
- log_server.py       : 로그를 웹으로 보여주는 모니터링 서버
- config.json         : 코인별 레버리지/손절/금액 설정
- ecosystem.config.js : PM2 실행 설정 파일

-------------------------------------------------------

[3] PM2 실행 명령어 (터미널)

# 1. 봇 & 로그서버 실행 (ecosystem.config.js 설정대로 시작)
pm2 start ecosystem.config.js

# 2. 실시간 상태 모니터링
pm2 monit
   (또는 pm2 logs)

# 3. 봇 재시작 (코드 수정 후 적용 시)
pm2 restart all

# 4. 봇 중지 및 프로세스 삭제
pm2 stop all
pm2 delete all

-------------------------------------------------------

[4] 모니터링 접속 방법
- 내부망(집): http://localhost:5000
- 외부망: 터미널에 뜬 Ngrok 주소 확인 (log_server.py 실행 시 출력됨)