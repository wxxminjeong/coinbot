import ccxt
import os
from dotenv import load_dotenv
import pprint  # 결과를 보기 좋게 출력하기 위해 사용

# 1. 환경 변수(.env) 로드
load_dotenv()

api_key = os.getenv('BINANCE_API_KEY')
secret_key = os.getenv('BINANCE_SECRET_KEY')

# 키가 제대로 불러와졌는지 확인 (보안을 위해 일부만 출력하거나 길이만 확인)
if not api_key or not secret_key:
    print("❌ .env 파일에서 API Key 또는 Secret Key를 찾을 수 없습니다.")
    exit()

print("🔑 API Key 로드 성공. 접속 시도 중...")

# 2. 바이낸스 객체 생성 (선물 거래용 설정)
exchange = ccxt.binance({
    'apiKey': api_key,
    'secret': secret_key,
    'enableRateLimit': True,
    'options': {
        'defaultType': 'future'  # 중요: 'future'로 설정해야 선물 지갑을 조회합니다.
    }
})

# 3. 잔고 조회 실행
try:
    # 샌드박스(테스트넷) 모드 사용 시 아래 주석 해제
    # exchange.set_sandbox_mode(True) 

    balance = exchange.fetch_balance()
    
    # USDT 잔고만 추출
    usdt_balance = balance['total'].get('USDT', 0)
    free_balance = balance['free'].get('USDT', 0)
    used_balance = balance['used'].get('USDT', 0)

    print("\n" + "="*30)
    print("      💰 바이낸스 선물 잔고 확인      ")
    print("="*30)
    print(f"💵 총 보유 자산 (Total): {usdt_balance:.2f} USDT")
    print(f"✅ 주문 가능 금액 (Free):  {free_balance:.2f} USDT")
    print(f"🔒 사용 중인 증거금 (Used): {used_balance:.2f} USDT")
    print("="*30 + "\n")
    
    print("✅ API 연결 테스트 성공!")

except ccxt.AuthenticationError:
    print("❌ 인증 실패: API Key와 Secret Key 권한을 확인하세요.")
    print("   (IP 제한이 걸려있거나, 선물 거래 권한(Futures Trading)이 체크되어 있는지 확인 필요)")
except Exception as e:
    print(f"❌ 에러 발생: {e}")