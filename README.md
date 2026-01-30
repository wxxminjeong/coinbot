# Binance Futures AI Trading Bot

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Google Gemini](https://img.shields.io/badge/Google%20Gemini-8E75B2?style=for-the-badge&logo=google&logoColor=white)
![Binance](https://img.shields.io/badge/Binance_Futures-FCD535?style=for-the-badge&logo=binance&logoColor=black)
![Ubuntu](https://img.shields.io/badge/Ubuntu_Server-E95420?style=for-the-badge&logo=ubuntu&logoColor=white)

**Intel N100 (Ubuntu Server)** 환경에서 구동되는 바이낸스 선물 자동매매 봇입니다.
기본적인 기술적 지표(Technical Indicators)와 **Google Gemini LLM**의 시장 분석을 결합하여 매매를 자동화했습니다.

## 🔄 Workflow

시스템의 주요 동작 과정은 다음과 같습니다.

1.  **Data Collection**
    * `CCXT`를 통해 바이낸스 선물의 실시간 캔들(OHLCV) 및 거래량 데이터를 수집합니다.
2.  **Technical Analysis**
    * 수집된 데이터를 가공하여 이동평균선(EMA), RSI, MACD, 볼린저 밴드 등 보조지표를 산출합니다.
3.  **AI Analysis**
    * 현재 차트 상황과 보조지표 데이터를 프롬프트로 구성하여 **Google Gemini**에 전송합니다.
    * AI는 제공된 데이터를 바탕으로 매수(Long), 매도(Short), 대기(Wait) 여부를 판단합니다.
4.  **Execution**
    * AI의 판단과 사전에 설정된 리스크 관리 규칙(TP/SL)에 따라 주문을 실행합니다.

## ✨ Key Features

* **Hybrid Strategy:** 정량적 지표 계산과 LLM의 상황 판단을 함께 사용하는 하이브리드 방식 적용.
* **Risk Management:** 격리 모드(Isolated Margin) 사용 및 자동화된 익절/손절(TP/SL) 주문 관리.
* **Monitoring:** Flask 기반 웹 대시보드를 통해 실시간 로그 및 포지션 상태 확인 가능.
* **Stability:** API Rate Limit 관리 및 네트워크 연결 끊김에 대한 예외 처리 적용.

## 🛠️ Tech Stack

* **Language:** Python 3.10+
* **Exchange API:** CCXT (Binance Futures)
* **AI Model:** Google Gemini 2.5 Flash
* **Data Processing:** Pandas
* **Server/Infra:** Ubuntu Server, PM2, Ngrok
* **Web Framework:** Flask

## ⚠️ Disclaimer

본 프로젝트는 개인 학습 및 연구 목적으로 개발되었습니다. 가상화폐 투자는 원금 손실의 위험이 있으며, 본 프로그램을 사용하여 발생하는 모든 결과에 대한 책임은 사용자 본인에게 있습니다.
