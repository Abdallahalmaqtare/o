"""
Pocket Option Data & Analysis Service v4.4 (Async Fix)
=====================================================
Uses 'pocketoptionapi-async' library with proper AsyncPocketOptionClient.
"""
import asyncio
import logging
import numpy as np
from typing import List, Dict, Optional

# Import from the correct module name installed from GitHub
try:
    import pocketoptionapi_async as po
    from pocketoptionapi_async import AsyncPocketOptionClient
except ImportError:
    # Fallback if the name is different or not installed
    po = None
    class AsyncPocketOptionClient:
        def __init__(self, *args, **kwargs):
            raise ImportError("pocketoptionapi-async not found. Please check requirements.txt")

logger = logging.getLogger(__name__)

class TechnicalAnalysis:
    """Pure NumPy implementation of technical indicators."""
    
    @staticmethod
    def ema(data: np.ndarray, period: int) -> np.ndarray:
        if len(data) < period: return np.array([])
        alpha = 2 / (period + 1)
        ema_values = np.zeros_like(data)
        ema_values[0] = data[0]
        for i in range(1, len(data)):
            ema_values[i] = alpha * data[i] + (1 - alpha) * ema_values[i-1]
        return ema_values

    @staticmethod
    def rsi(data: np.ndarray, period: int = 14) -> np.ndarray:
        if len(data) <= period: return np.array([])
        deltas = np.diff(data)
        seed = deltas[:period]
        up = seed[seed >= 0].sum() / period
        down = -seed[seed < 0].sum() / period
        rs = up / down if down != 0 else 100
        rsi_values = np.zeros_like(data)
        rsi_values[:period] = 100. - 100. / (1. + rs)
        for i in range(period, len(data)):
            delta = deltas[i-1]
            up_val, down_val = (delta, 0.) if delta > 0 else (0., -delta)
            up = (up * (period - 1) + up_val) / period
            down = (down * (period - 1) + down_val) / period
            rs = up / down if down != 0 else 100
            rsi_values[i] = 100. - 100. / (1. + rs)
        return rsi_values

    @staticmethod
    def supertrend(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 10, multiplier: float = 3.0):
        if len(close) <= period: return np.array([]), np.array([])
        tr1, tr2, tr3 = high[1:] - low[1:], np.abs(high[1:] - close[:-1]), np.abs(low[1:] - close[:-1])
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        atr = np.zeros_like(close)
        atr[period] = np.mean(tr[:period])
        for i in range(period + 1, len(close)):
            atr[i] = (atr[i-1] * (period - 1) + tr[i-1]) / period
        hl2 = (high + low) / 2
        upper_band, lower_band = hl2 + (multiplier * atr), hl2 - (multiplier * atr)
        f_upper, f_lower = np.copy(upper_band), np.copy(lower_band)
        trend = np.zeros_like(close)
        for i in range(period, len(close)):
            f_upper[i] = upper_band[i] if upper_band[i] < f_upper[i-1] or close[i-1] > f_upper[i-1] else f_upper[i-1]
            f_lower[i] = lower_band[i] if lower_band[i] > f_lower[i-1] or close[i-1] < f_lower[i-1] else f_lower[i-1]
            if close[i] > f_upper[i]: trend[i] = 1
            elif close[i] < f_lower[i]: trend[i] = -1
            else: trend[i] = trend[i-1]
        return trend, atr

    @staticmethod
    def adx(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14):
        if len(close) <= period * 2: return np.array([])
        up_move, down_move = high[1:] - high[:-1], low[:-1] - low[1:]
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
        tr = np.maximum(high[1:] - low[1:], np.maximum(np.abs(high[1:] - close[:-1]), np.abs(low[1:] - close[:-1])))
        s_tr, s_p_dm, s_m_dm = np.zeros(len(close)), np.zeros(len(close)), np.zeros(len(close))
        s_tr[period], s_p_dm[period], s_m_dm[period] = np.sum(tr[:period]), np.sum(plus_dm[:period]), np.sum(minus_dm[:period])
        for i in range(period + 1, len(close)):
            s_tr[i] = s_tr[i-1] - (s_tr[i-1] / period) + tr[i-1]
            s_p_dm[i] = s_p_dm[i-1] - (s_p_dm[i-1] / period) + plus_dm[i-1]
            s_m_dm[i] = s_m_dm[i-1] - (s_m_dm[i-1] / period) + minus_dm[i-1]
        plus_di, minus_di = 100 * (s_p_dm / s_tr), 100 * (s_m_dm / s_tr)
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
        adx_v = np.zeros(len(close))
        adx_v[period * 2 - 1] = np.mean(dx[period:period * 2])
        for i in range(period * 2, len(close)):
            adx_v[i] = (adx_v[i-1] * (period - 1) + dx[i]) / period
        return adx_v

class PocketOptionAnalyzer:
    def __init__(self):
        from config import (
            EMA_FAST, EMA_SLOW, RSI_PERIOD, RSI_CALL_MIN, RSI_PUT_MAX,
            SUPERTREND_PERIOD, SUPERTREND_MULTIPLIER, ADX_PERIOD, ADX_MIN_THRESHOLD
        )
        self.ema_fast_p, self.ema_slow_p = EMA_FAST, EMA_SLOW
        self.rsi_p, self.rsi_call_min, self.rsi_put_max = RSI_PERIOD, RSI_CALL_MIN, RSI_PUT_MAX
        self.st_p, self.st_m = SUPERTREND_PERIOD, SUPERTREND_MULTIPLIER
        self.adx_p, self.adx_min = ADX_PERIOD, ADX_MIN_THRESHOLD

    def analyze(self, candles: List[Dict]) -> Optional[Dict]:
        if not candles or len(candles) < 50: return None
        closes = np.array([c.get('close', c.get('c')) for c in candles if c], dtype=float)
        highs = np.array([c.get('high', c.get('h')) for c in candles if c], dtype=float)
        lows = np.array([c.get('low', c.get('l')) for c in candles if c], dtype=float)
        
        ta = TechnicalAnalysis()
        ema_f, ema_s, rsi = ta.ema(closes, self.ema_fast_p), ta.ema(closes, self.ema_slow_p), ta.rsi(closes, self.rsi_p)
        trend, _ = ta.supertrend(highs, lows, closes, self.st_p, self.st_m)
        adx = ta.adx(highs, lows, closes, self.adx_p)
        
        if any(len(x) == 0 for x in [ema_f, ema_s, rsi, trend, adx]): return None
        
        curr_c, curr_ef, curr_es, curr_r, curr_t, curr_a = closes[-1], ema_f[-1], ema_s[-1], rsi[-1], trend[-1], adx[-1]
        inds = {"ema_fast": round(curr_ef, 5), "ema_slow": round(curr_es, 5), "rsi": round(curr_r, 2), "supertrend": "UP" if curr_t == 1 else "DOWN", "adx": round(curr_a, 2)}
        
        if curr_c > curr_ef > curr_es and curr_r >= self.rsi_call_min and curr_t == 1 and curr_a >= self.adx_min:
            return {"direction": "CALL", "indicators": inds}
        if curr_c < curr_ef < curr_es and curr_r <= self.rsi_put_max and curr_t == -1 and curr_a >= self.adx_min:
            return {"direction": "PUT", "indicators": inds}
        return None

class PocketOptionDataService:
    def __init__(self, ssid: str, is_demo: bool = True):
        self.ssid, self.is_demo = ssid, is_demo
        self.client = AsyncPocketOptionClient(ssid=ssid)
        self._connected = False

    async def connect(self):
        if not self._connected:
            try:
                if not self.client.is_connected():
                    await self.client.connect()
                self._connected = self.client.is_connected()
                return self._connected
            except Exception as e:
                logger.error(f"Connection error: {e}")
                return False
        return True

    async def get_candles(self, asset: str, timeframe_seconds: int = 900, count: int = 100):
        if not await self.connect(): return []
        try:
            candles = await self.client.get_candles(asset, timeframe_seconds, count)
            if hasattr(candles, 'to_dict'): # if it's a dataframe
                return candles.to_dict('records')
            return candles
        except Exception as e:
            logger.error(f"Error fetching candles for {asset}: {e}")
            return []

_data_service = None
_analyzer = PocketOptionAnalyzer()

def init_data_service(ssid: str, is_demo: bool = True):
    global _data_service
    _data_service = PocketOptionDataService(ssid, is_demo)

def get_data_service(): return _data_service
def get_analyzer(): return _analyzer

ASSET_MAP = {"EURUSD": ["EURUSD_otc"], "USDGBP": ["USDGBP_otc"], "AUDCAD": ["AUDCAD_otc"]}
def get_asset_names(pair: str) -> List[str]:
    pair = pair.upper().replace("/", "")
    return ASSET_MAP.get(pair, [f"{pair}_otc", pair])
