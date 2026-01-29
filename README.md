# Binance Futures AI Trading Bot

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Google Gemini](https://img.shields.io/badge/Google%20Gemini-8E75B2?style=for-the-badge&logo=google&logoColor=white)
![Binance](https://img.shields.io/badge/Binance_Futures-FCD535?style=for-the-badge&logo=binance&logoColor=black)
![Ubuntu](https://img.shields.io/badge/Ubuntu_Server-E95420?style=for-the-badge&logo=ubuntu&logoColor=white)

**Intel N100 (Ubuntu Server)** 환경에서 24시간 운용되도록 설계된 바이낸스 선물 자동매매 시스템입니다.
기존의 기계적인 알고리즘 매매에 **Google Gemini LLM**의 추론 능력을 더하여, 시장 데이터를 종합적으로 해석하고 매매 의사결정을 내립니다.

## 🔄 System Architecture

이 프로젝트의 핵심 로직은 **데이터(Data)와 인공지능(AI)의 협업** 프로세스로 구성됩니다.

1.  **Data Collection (데이터 수집)**
    * `CCXT` 라이브러리를 통해 바이낸스 선물 시장의 실시간 가격(OHLCV) 및 거래량 데이터를 수집합니다.
2.  **Technical Analysis (기술적 분석)**
    * 수집된 데이터를 바탕으로 이동평균선(EMA), RSI, MACD, 볼린저 밴드 등 주요 보조지표를 실시간으로 계산합니다.
3.  **AI Reasoning (AI 추론)**
    * 현재의 차트 데이터, 지표 상태, 시장의 추세 정보를 텍스트 프롬프트로 구성하여 **Google Gemini**에게 전달합니다.
    * LLM은 단순 수치 비교를 넘어, 현재 상황이 매수(Long) 혹은 매도(Short)에 적합한지 논리적으로 판단합니다.
4.  **Execution (매매 실행)**
    * AI의 판단 결과와 리스크 관리 규칙(손절/익절 비율)을 결합하여 최종 주문을 바이낸스 서버로 전송합니다.

## ✨ Key Features

* **Hybrid Decision Making:** 정량적 지표(수학적 계산)와 정성적 분석(LLM의 시장 해석)을 결합한 하이브리드 매매 전략.
* **24/7 Automation:** 저전력 개인 서버(N100)에서 중단 없이 돌아가는 완전 자동화 시스템.
* **Real-time Monitoring:** Flask 기반의 웹 대시보드를 통해 실시간 로그 및 포지션 상태 모니터링.
* **Safety First:** API 에러 처리, 네트워크 재연결, 주문 무결성 검증 등 안정성 확보 로직 내장.

## 🛠️ Tech Stack

* **Language:** Python 3.10+
* **Exchange API:** CCXT (Binance Futures)
* **AI Model:** Google Gemini 2.5 Flash
* **Data Processing:** Pandas
* **Server/Infra:** Ubuntu Server, PM2, Ngrok
* **Dashboard:** Flask

## ⚠️ Disclaimer

본 프로젝트는 개인 연구 및 학습 목적으로 개발되었습니다. 가상화폐 투자는 높은 리스크를 수반하며, 본 알고리즘을 사용하여 발생하는 모든 금전적 손실에 대한 책임은 사용자 본인에게 있습니다.
