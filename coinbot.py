# -*- coding: utf-8 -*-
import ccxt
import os
import time
import json
import logging
import threading
import pandas as pd
import random
import re
from typing import Optional, Dict, List, Any
from dotenv import load_dotenv
from google import genai
from google.genai import types

# ---------------------------------------------------------
# [Configuration]
# ---------------------------------------------------------
load_dotenv()
CONFIG_FILE = "config.json"
MODEL_NAME = "gemini-2.5-flash-lite"
LOG_FILE_NAME = "bot_master.log"
AI_TIMEOUT = 30 

# JSON 파싱을 위한 정규식 미리 컴파일 (성능 최적화)
JSON_PATTERN = re.compile(r"```json\s*(.*?)\s*```|```\s*(.*?)\s*```", re.DOTALL)

BASE_PROMPT = "Act as a Conservative Scalper AI for {symbol} (1m chart)."

# ---------------------------------------------------------
# [Strategy Prompts]
# ---------------------------------------------------------
PROMPTS = {
    "ultra_safe": """
        {base}
        
        **CONTEXT**: 
        - Current Leverage: {leverage}x
        - Target: {target_move}% | Stop Loss: {sl_move}%
        
        **STRATEGY**: "Anti-FOMO Sniper" (Buy the start, NOT the peak)
        
        [Strict Entry Rules]
        1. **Trend Filter (EMA50)**:
           - LONG: Price > EMA50.
           - SHORT: Price < EMA50.
        
        2. **Distance Check (CRITICAL)**:
           - **Don't Chase!** If 'Dist_from_EMA' is > 0.3%, it's too late. Output "wait".
           - We only enter when price is CLOSE to the EMA50 line.

        3. **Volume & Momentum**: 
           - 'Vol_Ratio' > 2.0.
           - MACD supports the direction.

        4. **RSI Safety**:
           - LONG: RSI < 70.
           - SHORT: RSI > 30.
        
        [Decision Logic]
        - **REJECT**: If Price is too far from EMA50 (Checking 'Dist_from_EMA').
        - **REJECT**: If Price vs EMA50 mismatch.
        - **REJECT**: If Volume is weak.
        
        [Final Output]
        - **LONG**: Score >= 95 + Close to EMA.
        - **SHORT**: Score >= 95 + Close to EMA.
        - **WAIT**: If price extended or signals unclear.
        
        Output JSON: {{"decision": "long/short/wait", "reason": "Mention Distance from EMA"}}
    """
}

# ---------------------------------------------------------
# [Logging Setup]
# ---------------------------------------------------------
class AFCLogFilter(logging.Filter):
    def filter(self, record): return "AFC is enabled" not in record.getMessage()

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(message)s', datefmt='%H:%M:%S', handlers=[logging.FileHandler(LOG_FILE_NAME, encoding='utf-8'), logging.StreamHandler()])
logging.getLogger().handlers[0].addFilter(AFCLogFilter())
logging.getLogger().handlers[1].addFilter(AFCLogFilter())
logger = logging.getLogger("BOT")

for lib in ["httpx", "httpcore", "google", "urllib3"]: logging.getLogger(lib).setLevel(logging.ERROR)

# ---------------------------------------------------------
# [Trading Bot Class]
# ---------------------------------------------------------
class TradingBot(threading.Thread):
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        
        try:
            self.symbol = config['symbol']
            self.leverage = config['leverage']
            self.amount = config['amount']
            self.target_roe = config['target_roe']
            self.stop_loss_roe = config['stop_loss_roe']
            self.strategy = config['strategy']
        except KeyError as e:
            logger.error(f"❌ Config Error: Missing {e}")
            raise e

        self.running = True
        self.fix_history = {} 
        
        self.client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
        self.exchange = ccxt.binance({
            'apiKey': os.getenv("BINANCE_API_KEY"),
            'secret': os.getenv("BINANCE_SECRET_KEY"),
            'enableRateLimit': True,
            'options': {'defaultType': 'future'}
        })

    def run(self):
        logger.info(f"🚀 [{self.symbol}] Bot Started | TP:{self.target_roe}% SL:{self.stop_loss_roe}%")
        self._init_exchange_settings()
        
        loop_count = 0
        while self.running:
            try:
                # 1. 포지션 확인 (API Call 1)
                position = self._get_position()
                
                if position:
                    # 포지션이 있으면 관리 모드
                    self._handle_active_position(position, loop_count)
                else:
                    # 포지션이 없으면 탐색 모드
                    self._scan_market_for_entry(loop_count)
                
                # API Throttling (Random 4-6s)
                time.sleep(random.uniform(4, 6))
                loop_count += 1

            except Exception as e:
                self._handle_error(e)

    def _init_exchange_settings(self):
        try:
            self.exchange.load_markets()
            self.exchange.set_margin_mode('isolated', self.symbol)
            self.exchange.set_leverage(self.leverage, self.symbol)
        except Exception: 
            pass

    def _handle_error(self, e: Exception):
        msg = str(e)
        if "Code: -4164" in msg: 
            logger.error(f"❌ [{self.symbol}] Insufficient Balance (Min Notional)")
        elif "503" not in msg and "429" not in msg: 
            logger.error(f"⚠️ [{self.symbol}] System Error: {e}")
        time.sleep(10)

    # --- Position Management ---
    def _handle_active_position(self, position: Dict, loop_count: int):
        # 30초마다 주문 상태 점검 (API 절약)
        if loop_count % 6 == 0:
            self._ensure_orders(position)
            self._log_pnl(position)

    def _log_pnl(self, position: Dict):
        try:
            pnl = float(position['unrealizedPnl'])
            roi = (pnl / float(position['initialMargin'])) * 100
            icon = "🔴" if pnl < 0 else "🟢"
            logger.info(f"{icon} [{self.symbol}] Hold | ROI: {roi:.2f}% | PnL: ${pnl:.4f}")
        except Exception: 
            pass

    # --- Market Analysis (Optimized) ---
    def _scan_market_for_entry(self, loop_count: int):
        # [최적화] 무조건 주문 취소하던 로직 제거. 
        # 데이터를 먼저 보고 진입각이 나올 때만 주문 정리.
        
        market_data = self._get_market_data()
        if not market_data: return

        decision = self._ask_llm(market_data)
        
        if decision in ['long', 'short']:
            # [최적화] 진입 결정이 났을 때만 미체결 주문 정리 (API 효율성 증대)
            if self.exchange.fetch_open_orders(self.symbol):
                self.exchange.cancel_all_orders(self.symbol)
                
            self._execute_entry(decision)
            
        elif loop_count % 3 == 0:
            logger.info(f"👀 [{self.symbol}] Analyzing... (Wait)")

    # --- Data Collection ---
    def _get_market_data(self) -> Optional[str]:
        try:
            # Fetch 300 candles for EMA200
            ohlcv = self.exchange.fetch_ohlcv(self.symbol, '1m', limit=300)
            df = pd.DataFrame(ohlcv, columns=['ts', 'o', 'h', 'l', 'c', 'v'])
            
            close = df['c']
            
            # Indicators
            ema_20 = close.ewm(span=20).mean()
            ema_50 = close.ewm(span=50).mean()
            ema_200 = close.ewm(span=200).mean()
            
            curr_price = close.iloc[-1]
            curr_ema50 = ema_50.iloc[-1]
            curr_ema200 = ema_200.iloc[-1]
            
            # Distance from EMA50
            dist_ema = abs(curr_price - curr_ema50) / curr_ema50 * 100
            
            # Bollinger Bands
            std = close.rolling(20).std()
            upper = ema_20 + (std * 2)
            lower = ema_20 - (std * 2)
            
            # RSI
            delta = close.diff()
            gain = delta.where(delta > 0, 0).rolling(14).mean()
            loss = -delta.where(delta < 0, 0).rolling(14).mean()
            rsi = 100 - (100 / (1 + gain/loss))
            
            # MACD
            exp12 = close.ewm(span=12, adjust=False).mean()
            exp26 = close.ewm(span=26, adjust=False).mean()
            macd_line = exp12 - exp26
            signal_line = macd_line.ewm(span=9, adjust=False).mean()
            macd_hist = macd_line - signal_line
            
            # Volume Ratio
            vol_ma = df['v'].rolling(20).mean().iloc[-1]
            cur_vol = df['v'].iloc[-1]
            vol_ratio = cur_vol / vol_ma if vol_ma > 0 else 0
            
            return f"""
            Price: {curr_price}
            Trends: EMA50={curr_ema50:.4f}, EMA200={curr_ema200:.4f}
            Trend Status: {'BULL' if curr_price > curr_ema200 else 'BEAR'} (EMA200 Base)
            Dist_from_EMA50: {dist_ema:.3f}% (Limit: 0.3%)
            MACD: Hist={macd_hist.iloc[-1]:.4f}
            RSI(14): {rsi.iloc[-1]:.1f}
            Volume Ratio: {vol_ratio:.2f}x
            """
        except Exception: 
            return None

    # --- AI Decision (Optimized) ---
    def _ask_llm(self, data: str) -> str:
        target_price_move = self.target_roe / self.leverage
        sl_price_move = self.stop_loss_roe / self.leverage

        prompt_template = PROMPTS[self.strategy]
        
        final_prompt = prompt_template.format(
            base=BASE_PROMPT.format(symbol=self.symbol),
            leverage=self.leverage,
            target_move=f"{target_price_move:.2f}",
            sl_move=f"{sl_price_move:.2f}"
        ) + f"\nData:\n{data}"
        
        try:
            res = self.client.models.generate_content(
                model=MODEL_NAME,
                contents=[types.Content(role="user", parts=[types.Part.from_text(text=final_prompt)])],
                config=types.GenerateContentConfig(response_mime_type="application/json", temperature=0.1)
            )
            
            text_res = res.text.strip()
            
            # [최적화] 정규식으로 JSON 추출 (더 안전하고 빠름)
            match = JSON_PATTERN.search(text_res)
            if match:
                text_res = match.group(1) or match.group(2)
                
            parsed = json.loads(text_res)
            decision = parsed.get("decision", "wait").lower()
            
            if decision != "wait":
                reason = parsed.get("reason", "No reason provided")
                logger.info(f"💡 [{self.symbol}] AI Insight: {reason}")
            
            return decision
        except Exception as e:
            logger.error(f"AI Error: {e}")
            return "wait"

    # --- Execution ---
    def _execute_entry(self, side: str):
        try:
            self.exchange.set_margin_mode('isolated', self.symbol) 
            ticker = self.exchange.fetch_ticker(self.symbol)
            
            raw_amount = (self.amount * self.leverage / ticker['last'])
            amount = self.exchange.amount_to_precision(self.symbol, raw_amount)
            
            order_func = self.exchange.create_market_buy_order if side == 'long' else self.exchange.create_market_sell_order
            order_func(self.symbol, amount)
            
            time.sleep(2)
            
            pos = self._get_position()
            if pos:
                self.fix_history[self.symbol] = 0
                self._ensure_orders(pos)
                logger.info(f"⚡ [{self.symbol}] Entry {side.upper()} Done")
        except Exception as e:
            logger.error(f"❌ [{self.symbol}] Entry Fail: {e}")

    def _ensure_orders(self, position: Dict):
        # API 과부하 방지 쿨다운
        if time.time() - self.fix_history.get(self.symbol, 0) < 60: return

        try:
            orders = self.exchange.fetch_open_orders(self.symbol)
            side = position['side']
            
            has_tp = any('limit' in o['type'].lower() for o in orders)
            has_sl = any('stop' in o['type'].lower() for o in orders)

            if has_tp and has_sl: return

            logger.warning(f"🔧 [{self.symbol}] Fixing Orders")
            self.fix_history[self.symbol] = time.time()
            self.exchange.cancel_all_orders(self.symbol)
            time.sleep(2)
            
            entry = float(position['entryPrice'])
            amt = float(position['contracts'])
            
            tp_rate = (self.target_roe / self.leverage) / 100
            sl_rate = (self.stop_loss_roe / self.leverage) / 100
            
            if side == 'long':
                tp_price = self.exchange.price_to_precision(self.symbol, entry * (1 + tp_rate))
                sl_price = self.exchange.price_to_precision(self.symbol, entry * (1 - sl_rate))
                
                self.exchange.create_limit_sell_order(self.symbol, amt, tp_price, {'reduceOnly': True})
                self.exchange.create_order(self.symbol, 'STOP_MARKET', 'sell', amt, None, {'stopPrice': sl_price, 'closePosition': True})
            else:
                tp_price = self.exchange.price_to_precision(self.symbol, entry * (1 - tp_rate))
                sl_price = self.exchange.price_to_precision(self.symbol, entry * (1 + sl_rate))
                
                self.exchange.create_limit_buy_order(self.symbol, amt, tp_price, {'reduceOnly': True})
                self.exchange.create_order(self.symbol, 'STOP_MARKET', 'buy', amt, None, {'stopPrice': sl_price, 'closePosition': True})
                
            logger.info(f"✅ [{self.symbol}] Orders Fixed")
        except Exception as e:
            if "-4130" not in str(e): logger.error(f"⚠️ Fix Fail: {e}")

    def _get_position(self) -> Optional[Dict]:
        try:
            positions = self.exchange.fetch_positions([self.symbol])
            for p in positions:
                if float(p['contracts']) != 0: return p
            return None
        except Exception: 
            return None

if __name__ == "__main__":
    try:
        with open(CONFIG_FILE, 'r') as f: config_list = json.load(f)
        
        logger.info(f"🔥 Binance AI Bot Started ({len(config_list)} pairs)")
        
        threads = [TradingBot(cfg) for cfg in config_list]
        for t in threads: 
            t.start()
            time.sleep(random.uniform(0.5, 1.5)) 
            
        for t in threads: t.join()

    except KeyboardInterrupt: logger.info("👋 Exit")
    except Exception as e: logger.error(f"Main Error: {e}")
