#!/usr/bin/env python3
"""
Strict Institutional Volatility Tracker (Upstox) - STATEFUL INTRADAY EDITION
+ Dynamic Anchor Time: Locks the exact minute a 1X block fires.
+ Stateful Tracking: Tracks the anchor's survival through the rest of the day.
+ Forensic Graveyard: Lists exactly WHY every filtered stock was rejected.
"""
import os
import sys
import re
import argparse
import urllib.parse
import json
import gzip
import io
import time
import threading
from bisect import bisect_left
from datetime import datetime, timedelta, timezone
import concurrent.futures

import requests
from requests.adapters import HTTPAdapter
import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings("ignore")

# ==============================================================================
# 0. ENGINE CONSTANTS & CONFIGURATION 
# ==============================================================================
TRADING_MODE = "STOCK_FNO"   # Options: "STOCK_FNO", "CASH_EQUITY", "INDEX_OPTIONS"

EXPIRY_OFFSET = 0          
STRIKES_FROM_ATM = 5       
OPT_MIN_PRICE = 30
OPT_MIN_VOLUME = 10000

HA_ATR_MULTIPLIERS = [1, 2, 3, 5]
ATR_BASIS_PERIOD = 14
ATR_BASIS_TF = "15min"
MIN_ATR_PCT = 0.001

TOP_N_BUYERS = 15
TOP_N_SELLERS = 15
MIN_PERFECT_BLOCKS = 1

COLOR_GREEN_BG = '\033[42m\033[30m'
COLOR_RED_BG = '\033[41m\033[97m'
COLOR_RESET = '\033[0m'
COLOR_BOLD = '\033[1m'
COLOR_CYAN = '\033[96m'
COLOR_RED_FG = '\033[91m'
COLOR_YELLOW = '\033[93m'
ANSI_RE = re.compile(r'\x1b\[[0-9;]*m')

MIN_PRICE = 100
MAX_PRICE = 5000
MIN_DAILY_VOLUME = 100000
BACKTRACE_DAYS = 30        

RSI_PERIOD = 14
BB_PERIOD = 20
BB_STD = 1
ADX_PERIOD = 14
ADX_THRESHOLD = 20

WORKERS = 16                       
INCLUDE_NON_EQ_SERIES = False      
PREFILTER_PRICE_SLACK = 0.25
PREFILTER_MIN_TODAY_VOLUME = 0     

RATE_CAPS = ((1.0, 22), (60.0, 220), (1800.0, 900))
API_HOST = "https://api.upstox.com"
IST = timezone(timedelta(hours=5, minutes=30))
SESSION_OPEN_MIN = 9 * 60 + 15
SESSION_CLOSE_MIN = 15 * 60 + 30

# ==============================================================================
# HELPERS
# ==============================================================================
def now_ist():
    return datetime.now(IST).replace(tzinfo=None)

def market_is_open():
    n = now_ist()
    m = n.hour * 60 + n.minute
    return n.weekday() < 5 and SESSION_OPEN_MIN <= m < SESSION_CLOSE_MIN

class BudgetLimiter:
    def __init__(self, caps):
        self.caps = caps
        self.horizon = max(s for s, _ in caps)
        self.lock = threading.Lock()
        self.stamps = []
        self.block_until = 0.0
        self.total_calls = 0

    def acquire(self):
        while True:
            with self.lock:
                now = time.time()
                cut = bisect_left(self.stamps, now - self.horizon)
                if cut: del self.stamps[:cut]
                n = len(self.stamps)
                wait = max(0.0, self.block_until - now)
                for span, cap in self.caps:
                    if n >= cap and n - bisect_left(self.stamps, now - span) >= cap:
                        wait = max(wait, self.stamps[n - cap] + span - now)
                if wait <= 0:
                    self.stamps.append(now); self.total_calls += 1; return
            time.sleep(min(wait, 1.0) + 0.005)

class FetchStats:
    def __init__(self):
        self.lock = threading.Lock()
        self.failed = 0
        self.auth_failed = False
    def mark_auth_failed(self):
        with self.lock: self.auth_failed = True

STATS = FetchStats()
LIMITERS = {"quotes": BudgetLimiter(RATE_CAPS), "history": BudgetLimiter(RATE_CAPS), "intraday": BudgetLimiter(RATE_CAPS)}
_TLS = threading.local()

def _limiter_for(url):
    if "market-quote/quotes" in url: return LIMITERS["quotes"]
    if "/intraday/" in url: return LIMITERS["intraday"]
    return LIMITERS["history"]

def _session():
    s = getattr(_TLS, "s", None)
    if s is None:
        s = requests.Session()
        s.mount("https://", HTTPAdapter(pool_connections=2, pool_maxsize=2))
        _TLS.s = s
    return s

class Progress:
    def __init__(self, label, total):
        self.label, self.total, self.n, self.step = label, total, 0, max(1, total // 20)
        self.lock = threading.Lock()
    def tick(self):
        with self.lock:
            self.n += 1
            if self.n == self.total or self.n % self.step == 0:
                print(f"\r   {self.label}: {self.n}/{self.total}", end="", file=sys.stderr, flush=True)
    def done(self): print("", file=sys.stderr)

# ==============================================================================
# 1. UPSTOX API
# ==============================================================================
def _get(url, params=None, retries=4):
    token = os.environ.get("UPSTOX_ACCESS_TOKEN")
    if not token or STATS.auth_failed: return 401, None
    headers = {'Accept': 'application/json', 'Authorization': f'Bearer {token}'}
    limiter = _limiter_for(url)
    for attempt in range(retries):
        limiter.acquire()
        try:
            r = _session().get(url, headers=headers, params=params, timeout=20)
            code = r.status_code
            if code == 200: return 200, r.json()
            if code == 401: STATS.mark_auth_failed(); return 401, None
        except Exception: pass
        time.sleep(0.5 * (attempt + 1))
    return 0, None

def _to_frame(candles):
    if not candles: return None
    df = pd.DataFrame(candles).iloc[:, :6]
    df.columns = ['Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume']
    ts = df['Timestamp'].astype(str)
    try: df['Datetime'] = pd.to_datetime(ts.str.slice(0, 19), format="%Y-%m-%dT%H:%M:%S")
    except ValueError: df['Datetime'] = pd.to_datetime(ts, utc=True).dt.tz_convert(IST).dt.tz_localize(None)
    for c in ('Open', 'High', 'Low', 'Close', 'Volume'): df[c] = df[c].astype(float)
    return df.drop(columns='Timestamp').drop_duplicates(subset='Datetime').sort_values('Datetime').reset_index(drop=True)

def _candles(url):
    status, js = _get(url)
    return (200, _to_frame((js.get('data') or {}).get('candles') or [])) if status == 200 and js else (status, None)

def _url_range_daily(key, start, end): 
    return f"{API_HOST}/v2/historical-candle/{urllib.parse.quote(key)}/day/{end:%Y-%m-%d}/{start:%Y-%m-%d}"
def _url_range_1m(key, start, end): 
    return f"{API_HOST}/v2/historical-candle/{urllib.parse.quote(key)}/1minute/{end:%Y-%m-%d}/{start:%Y-%m-%d}"
def _url_intraday(key): 
    return f"{API_HOST}/v2/historical-candle/intraday/{urllib.parse.quote(key)}/1minute"

def _date_chunks(start, end, span=7):
    cur = end
    while cur >= start:
        c_start = max(start, cur - timedelta(days=span - 1))
        yield c_start, cur
        cur = c_start - timedelta(days=1)

def fetch_quotes(items, batch=200):
    batches = [items[i:i + batch] for i in range(0, len(items), batch)]
    def one(b):
        status, js = _get(f"{API_HOST}/v2/market-quote/quotes", params={"instrument_key": ",".join(x['key'] for x in b)})
        if status != 200 or not js: return {}
        by_ts = {f"{x['key'].split('|')[0]}:{x['symbol']}": x['key'] for x in b}
        out = {}
        for k, v in (js.get('data') or {}).items():
            key = v.get('instrument_token') or by_ts.get(k)
            if key: out[key] = {'ltp': v.get('last_price') or 0.0, 'vol': v.get('volume') or 0}
        return out
    res = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(WORKERS, max(1, len(batches)))) as ex:
        for part in ex.map(one, batches): res.update(part)
    return res

def fetch_today(key):
    if now_ist().weekday() >= 5: return None
    status, df = _candles(_url_intraday(key))
    if status != 200 or df is None: return None
    df = df[df['Datetime'].dt.date == now_ist().date()]     
    return df if not df.empty else None

def prepare_master(dfs):
    master = pd.concat([d[['Datetime', 'Open', 'High', 'Low', 'Close', 'Volume']] for d in dfs], ignore_index=True).drop_duplicates(subset='Datetime').sort_values('Datetime').reset_index(drop=True)
    dt = master['Datetime']
    day = dt.dt.normalize()
    master['Min'] = dt.dt.hour.values * 60 + dt.dt.minute.values
    master['SessId'] = day.values.astype('datetime64[D]').astype('int64')
    master['Session'] = day.dt.date
    return master

def _download_master(name):
    for _ in range(3):
        try:
            resp = requests.get(f"https://assets.upstox.com/market-quote/instruments/exchange/{name}.json.gz", timeout=90)
            if resp.status_code == 200: return json.load(gzip.GzipFile(fileobj=io.BytesIO(resp.content)))
        except Exception: time.sleep(1.0)
    return None

def _equity_universe(mode):
    nse = _download_master("NSE")
    if not nse: return []
    def ts_of(i): return i.get("tradingsymbol", i.get("trading_symbol"))
    def plain(i): return (i.get("segment") == "NSE_EQ" and ts_of(i) and i.get("instrument_key") and (INCLUDE_NON_EQ_SERIES or (i.get("instrument_type") or "EQ") == "EQ"))
    fno = {i.get("underlying_symbol") for i in nse if i.get("segment") == "NSE_FO" and i.get("underlying_symbol")}
    rows = [i for i in nse if plain(i) and (ts_of(i) in fno if mode == "STOCK_FNO" else ts_of(i) not in fno)]
    return list({i["instrument_key"]: {"symbol": ts_of(i), "key": i["instrument_key"]} for i in rows}.values())

def get_dynamic_universe(mode): return _equity_universe(mode) if mode in ("STOCK_FNO", "CASH_EQUITY") else [] 

# ==============================================================================
# 2. THE DECOUPLED HA-ATR ENGINE 
# ==============================================================================
def compute_base_atr(m):
    hi, lo, cl = m['High'].values, m['Low'].values, m['Close'].values
    sid, mins = m['SessId'].values, m['Min'].values
    tf_minutes = int(ATR_BASIS_TF.replace("min", ""))
    bucket = sid * 100 + mins // tf_minutes
    starts = np.concatenate(([0], np.flatnonzero(np.diff(bucket)) + 1))
    ends = np.concatenate((starts[1:], [len(bucket)])) - 1
    b_high, b_low, b_close = np.maximum.reduceat(hi, starts), np.minimum.reduceat(lo, starts), cl[ends]
    if len(b_close) < 5: return max(cl[-1] * MIN_ATR_PCT, 0.01)
    prev_close = np.concatenate(([np.nan], b_close[:-1]))
    tr = np.fmax(np.fmax(b_high - b_low, np.abs(b_high - prev_close)), np.abs(b_low - prev_close))
    return max(float(tr[-ATR_BASIS_PERIOD:].mean()), b_close[-1] * MIN_ATR_PCT, 0.01)

def split_sessions(m):
    sid = m['SessId'].values
    cuts = np.flatnonzero(np.diff(sid)) + 1
    starts = np.concatenate(([0], cuts)); ends = np.concatenate((cuts, [len(sid)]))
    o, h, l, c, dt = (m[k].tolist() for k in ('Open', 'High', 'Low', 'Close', 'Datetime'))
    return [(o[s:e], h[s:e], l[s:e], c[s:e], dt[s:e]) for s, e in zip(starts, ends)]

def build_isolated_range_bars(sessions, target_range, base_atr=None):
    B_O, B_H, B_L, B_C, B_CLOSED, B_TIME = [], [], [], [], [], []
    seg_start = 0

    for opens, highs, lows, closes, dts in sessions:
        if not closes: continue
        before = len(B_C)
        curr_O, curr_H, curr_L, curr_C = opens[0], highs[0], lows[0], closes[0]
        curr_T = dts[0]

        for hi, lo, cl, t in zip(highs, lows, closes, dts):
            if hi > curr_H: curr_H = hi
            if lo < curr_L: curr_L = lo
            curr_C = cl
            curr_T = t

            if curr_H - curr_L >= target_range:
                B_O.append(curr_O); B_H.append(curr_H); B_L.append(curr_L); B_C.append(curr_C); B_TIME.append(curr_T)
                B_CLOSED.append(True)
                curr_O = curr_H = curr_L = curr_C
                
        if curr_H > curr_L:
            B_O.append(curr_O); B_H.append(curr_H); B_L.append(curr_L); B_C.append(curr_C); B_TIME.append(curr_T)
            B_CLOSED.append(False) 
        if len(B_C) > before: seg_start = before

    if not B_C: return None

    o = B_O[seg_start:]; h = B_H[seg_start:]; l = B_L[seg_start:]; c = B_C[seg_start:]; t = B_TIME[seg_start:]
    closed = B_CLOSED[seg_start:]
    ha_close = [(o[i] + h[i] + l[i] + c[i]) / 4 for i in range(len(c))]
    ha_open = (o[0] + c[0]) / 2
    for i in range(1, len(c)): ha_open = (ha_open + ha_close[i - 1]) / 2.0
    ha_trend = ['Green' if ha_close[i] >= ha_open else 'Red' for i in range(len(c))]

    return {'High': np.asarray(h), 'Low': np.asarray(l), 'Close': np.asarray(c), 'Closed': closed, 'HA_Trend': ha_trend, 'Time': t}

def _ewm(x, alpha):
    xs = x.tolist(); out = [0.0] * len(xs); prev = out[0] = xs[0]
    keep = 1.0 - alpha
    for i in range(1, len(xs)): out[i] = prev = keep * prev + alpha * xs[i]
    return np.asarray(out)

def _last_bb(series):
    w = series[-BB_PERIOD:]
    return float(w.mean()), float(w.std(ddof=0)) if len(w) > 1 else 0.0

def _evaluate_kinetic_step(close, high, low, idx):
    if idx < 5: return "Neutral", "Neutral", "Neutral", False, False
    
    n = idx + 1
    c_slice = close[:n]
    h_slice = high[:n]
    l_slice = low[:n]
    
    delta = np.diff(c_slice, prepend=c_slice[0])
    gain = _ewm(np.where(delta > 0, delta, 0.0), 1 / RSI_PERIOD)
    loss = _ewm(np.where(delta < 0, -delta, 0.0), 1 / RSI_PERIOD)
    rsi = 100 - (100 / (1 + (gain / (loss + 1e-8))))
    r_mean, r_std = _last_bb(rsi)

    macd = _ewm(c_slice, 2 / 13) - _ewm(c_slice, 2 / 27)
    hist = macd - _ewm(macd, 2 / 10)
    h_mean, h_std = _last_bb(hist)

    up, down = np.zeros(n), np.zeros(n)
    up[1:] = h_slice[1:] - h_slice[:-1]; down[1:] = l_slice[:-1] - l_slice[1:]
    plus_dm = np.where((up > down) & (up > 0), up, 0.0)
    minus_dm = np.where((down > up) & (down > 0), down, 0.0)
    
    tr = (h_slice - l_slice).copy()
    tr[1:] = np.maximum(tr[1:], np.maximum(np.abs(h_slice[1:] - c_slice[:-1]), np.abs(l_slice[1:] - c_slice[:-1])))

    a = 1 / ADX_PERIOD; atr = _ewm(tr, a)
    plus_di = 100 * (_ewm(plus_dm, a) / (atr + 1e-8))
    minus_di = 100 * (_ewm(minus_dm, a) / (atr + 1e-8))
    adx = _ewm(100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-8), a)
    
    p_mean, p_std = _last_bb(plus_di)
    m_mean, m_std = _last_bb(minus_di)

    rsi_sig = "Buy" if r_std > 0 and rsi[-1] > r_mean + BB_STD * r_std else "Sell" if r_std > 0 and rsi[-1] < r_mean - BB_STD * r_std else "Neutral"
    macd_sig = "Buy" if h_std > 0 and hist[-1] > h_mean + BB_STD * h_std else "Sell" if h_std > 0 and hist[-1] < h_mean - BB_STD * h_std else "Neutral"

    bull_agg = p_std > 0 and plus_di[-1] > p_mean + BB_STD * p_std
    bear_agg = m_std > 0 and minus_di[-1] > m_mean + BB_STD * m_std

    adx_sig = "Buy" if bull_agg and plus_di[-1] > minus_di[-1] and adx[-1] >= ADX_THRESHOLD else "Sell" if bear_agg and minus_di[-1] > plus_di[-1] and adx[-1] >= ADX_THRESHOLD else "Neutral"
    
    return rsi_sig, macd_sig, adx_sig, bull_agg, bear_agg

def evaluate_anchor(bars, bb_upper, bb_lower, kc_upper, kc_lower, close_1m, dt_1m):
    """
    Steps through today's 1X blocks. Returns the anchor if it survives.
    Otherwise, returns (None, RejectionReasonString) so the UI can display why it failed.
    """
    if bars is None or len(bars['Close']) < 5: 
        return None, "Not enough fully closed 1X range blocks today."
    
    close, high, low = bars['Close'], bars['High'], bars['Low']
    closed = bars['Closed']
    times = bars['Time']
    ha_trend = bars['HA_Trend']
    
    anchor = None
    warzone_kills = 0
    
    # 1. FIND THE ANCHOR
    for i in range(len(close)):
        if not closed[i]: continue 
        
        f_rsi, f_macd, f_adx, f_bull_power, f_bear_power = _evaluate_kinetic_step(close, high, low, i)
        
        try:
            m_idx = next(idx for idx, t in enumerate(dt_1m) if t >= times[i])
        except StopIteration:
            m_idx = len(close_1m) - 1
            
        bb_kc_bull_fire = bb_upper[m_idx] > kc_upper[m_idx]
        bb_kc_bear_fire = bb_lower[m_idx] < kc_lower[m_idx]

        is_bull_raw = f_rsi == "Buy" and f_macd == "Buy" and f_adx == "Buy" and ha_trend[i] == 'Green' and bb_kc_bull_fire
        is_bear_raw = f_rsi == "Sell" and f_macd == "Sell" and f_adx == "Sell" and ha_trend[i] == 'Red' and bb_kc_bear_fire

        is_bull = is_bull_raw
        is_bear = is_bear_raw

        # Warzone Audit
        if is_bull_raw and f_bear_power: is_bull = False; warzone_kills += 1
        if is_bear_raw and f_bull_power: is_bear = False; warzone_kills += 1
        
        if is_bull:
            anchor = {"dir": "BULL", "time": times[i], "low": low[i], "high": high[i], "idx": i, "m_idx": m_idx}
            break
        if is_bear:
            anchor = {"dir": "BEAR", "time": times[i], "low": low[i], "high": high[i], "idx": i, "m_idx": m_idx}
            break

    if not anchor: 
        if warzone_kills > 0:
            return None, f"Killed by Inverted Warzone Chop ({warzone_kills} attempts)."
        return None, "Failed Kinetic Alignment / BB never pierced KC."
    
    # 2. VERIFY SURVIVAL
    for i in range(anchor['m_idx'], len(close_1m)):
        if anchor['dir'] == "BULL" and close_1m[i] < anchor['low']:
            b_time = pd.to_datetime(dt_1m[i]).strftime('%H:%M')
            a_time = pd.to_datetime(anchor['time']).strftime('%H:%M')
            return None, f"Anchor at {a_time} breached its Low ({anchor['low']:.2f}) at {b_time}."
        if anchor['dir'] == "BEAR" and close_1m[i] > anchor['high']:
            b_time = pd.to_datetime(dt_1m[i]).strftime('%H:%M')
            a_time = pd.to_datetime(anchor['time']).strftime('%H:%M')
            return None, f"Anchor at {a_time} breached its High ({anchor['high']:.2f}) at {b_time}."
            
    f_rsi, f_macd, f_adx, _, _ = _evaluate_kinetic_step(close, high, low, len(close)-1)

    return {
        "dir": anchor['dir'],
        "time": pd.to_datetime(anchor['time']).strftime("%H:%M"),
        "rsi": f_rsi, "macd": f_macd, "adx": f_adx
    }, "Survived"


def compute_row(symbol, master_1m):
    close, high, low, dt = master_1m['Close'].values, master_1m['High'].values, master_1m['Low'].values, master_1m['Datetime'].values
    
    sma20 = pd.Series(close).rolling(20, min_periods=1).mean().values
    std20 = pd.Series(close).rolling(20, min_periods=1).std(ddof=0).values
    
    tr = np.zeros_like(close)
    tr[0] = high[0] - low[0]
    tr[1:] = np.maximum(high[1:] - low[1:], np.maximum(np.abs(high[1:] - close[:-1]), np.abs(low[1:] - close[:-1])))
    atr20 = pd.Series(tr).rolling(20, min_periods=1).mean().values
    
    bb_upper = sma20 + 2.0 * std20
    bb_lower = sma20 - 2.0 * std20
    kc_upper = sma20 + 1.5 * atr20
    kc_lower = sma20 - 1.5 * atr20
    
    base_atr = compute_base_atr(master_1m)
    sessions = split_sessions(master_1m)
    
    row = {'Symbol': symbol, 'LTP': float(close[-1]), 'LastSession': master_1m['Session'].iloc[-1],
           'ActiveAnchor': None, 'AnchorDir': "NONE", 'State': "NONE", 'Blocks': 0, 'RejectReason': ""}
    
    bars_1x = build_isolated_range_bars(sessions, base_atr * 1, base_atr=base_atr)
    anchor, reject_reason = evaluate_anchor(bars_1x, bb_upper, bb_lower, kc_upper, kc_lower, close, dt)
    
    row['RejectReason'] = reject_reason

    if anchor:
        row['ActiveAnchor'] = anchor['time']
        row['AnchorDir'] = anchor['dir']
        row['BB_RSI_1X'] = anchor['rsi']
        row['BB_MACD_1X'] = anchor['macd']
        row['ADX_1X'] = anchor['adx']
        row['Blocks'] += 1
        
        if anchor['dir'] == "BULL":
            row['State'] = "[ACTIVE BUY]" if bb_upper[-1] > kc_upper[-1] else "[COILING]"
        else:
            row['State'] = "[ACTIVE SELL]" if bb_lower[-1] < kc_lower[-1] else "[COILING]"
            
        for mult in HA_ATR_MULTIPLIERS[1:]:
            gtag = f"{mult}X"
            bars = build_isolated_range_bars(sessions, base_atr * mult, base_atr=base_atr)
            if bars and len(bars['Close']) >= 5:
                f_rsi, f_macd, f_adx, _, _ = _evaluate_kinetic_step(bars['Close'], bars['High'], bars['Low'], len(bars['Close'])-1)
                row[f'BB_RSI_{gtag}'], row[f'BB_MACD_{gtag}'], row[f'ADX_{gtag}'] = f_rsi, f_macd, f_adx
                if anchor['dir'] == "BULL" and f_rsi == "Buy" and f_macd == "Buy" and f_adx == "Buy": row['Blocks'] += 1
                if anchor['dir'] == "BEAR" and f_rsi == "Sell" and f_macd == "Sell" and f_adx == "Sell": row['Blocks'] += 1

    return row

# ==============================================================================
# 3. PIPELINE EXECUTOR & UI DRAWING
# ==============================================================================
def format_cell(text, width=10):
    if not text: return " " * width
    spaces = width - len(str(text))
    left_pad, right_pad = " " * (spaces // 2), " " * (spaces - (spaces // 2))
    colored_text = f"{COLOR_GREEN_BG}{text}{COLOR_RESET}" if text == "Buy" else f"{COLOR_RED_BG}{text}{COLOR_RESET}" if text == "Sell" else str(text)
    return f"{left_pad}{colored_text}{right_pad}"

def _history_worker_daily(args):
    item, start, end, progress = args
    try:
        frames = []
        for c_start, c_end in _date_chunks(start, end, span=365):
            status, df = _candles(_url_range_daily(item['key'], c_start, c_end))
            if status == 200 and df is not None: frames.append(df)
        if not frames: return None
        master = prepare_master(frames)
        close, vol = master['Close'].iloc[-1], master['Volume'].iloc[-1]
        ok = MIN_PRICE <= close <= MAX_PRICE and vol >= MIN_DAILY_VOLUME
        return item if ok else None
    except Exception: return None
    finally: progress.tick()

def process_stock_1m(args):
    item, cutoff_dt, is_live, progress = args
    try:
        target_dt = cutoff_dt.date()
        start_dt = target_dt - timedelta(days=7) 
        frames = []
        for c_start, c_end in _date_chunks(start_dt, target_dt, span=7):
            status, df = _candles(_url_range_1m(item['key'], c_start, c_end))
            if status == 200 and df is not None: frames.append(df)
            
        if is_live:
            today_df = fetch_today(item['key'])
            if today_df is not None: frames.append(today_df)
            
        if not frames: return None
        master_1m = prepare_master(frames)
        master_1m = master_1m[master_1m['Datetime'] <= cutoff_dt] 
        
        if len(master_1m) < 30: return None
        return compute_row(item['symbol'], master_1m)
    except Exception: return None
    finally: progress.tick()

def run_screener(mode=TRADING_MODE, days=BACKTRACE_DAYS, target_date_str=None, target_time_str="15:30"):
    t_start = time.time()
    
    if target_date_str:
        try:
            cutoff_dt = datetime.strptime(f"{target_date_str} {target_time_str}", "%Y-%m-%d %H:%M")
            target_dt = cutoff_dt.date()
            is_live = False
        except ValueError:
            print(f"{COLOR_RED_FG}[!] Invalid date format.{COLOR_RESET}"); return
    else:
        target_dt = now_ist().date()
        cutoff_dt = now_ist()
        is_live = True

    print(f"\n{COLOR_CYAN}📡 Initializing Tracker [{mode}] | Time Machine: {cutoff_dt.strftime('%Y-%m-%d %H:%M')} (Live: {is_live}){COLOR_RESET}")

    universe_raw = get_dynamic_universe(mode)
    if STATS.auth_failed or not universe_raw: return

    daily_start = target_dt - timedelta(days=days * 2 + 6)
    prog = Progress("daily_prefilter", len(universe_raw))
    with concurrent.futures.ThreadPoolExecutor(max_workers=WORKERS) as ex:
        candidates = [r for r in ex.map(_history_worker_daily, [(it, daily_start, target_dt, prog) for it in universe_raw]) if r is not None]
    prog.done()

    if STATS.auth_failed or not candidates: return

    prog = Progress("1m_blocks", len(candidates))
    with concurrent.futures.ThreadPoolExecutor(max_workers=WORKERS) as ex:
        results = [r for r in ex.map(process_stock_1m, [(item, cutoff_dt, is_live, prog) for item in candidates]) if r is not None]
    prog.done()

    if not results: return

    bulls = [r for r in results if r['AnchorDir'] == "BULL"]
    bears = [r for r in results if r['AnchorDir'] == "BEAR"]
    rejected = [r for r in results if r['AnchorDir'] == "NONE"]

    bulls.sort(key=lambda r: (-r['Blocks'], r['State'] != "[ACTIVE BUY]"))
    bears.sort(key=lambda r: (-r['Blocks'], r['State'] != "[ACTIVE SELL]"))
    
    bulls, bears = bulls[:TOP_N_BUYERS], bears[:TOP_N_SELLERS]

    print(f"{COLOR_BOLD}\n=== STATEFUL INSTITUTIONAL VOLATILITY TRACKER [{mode}] ==={COLOR_RESET}")
    print(f"Time Machine: {cutoff_dt.strftime('%Y-%m-%d %H:%M')} | Scanned: {len(results)}\n")

    def print_basket(title, icon, data_list):
        if not data_list: return
        print(f"\n{COLOR_BOLD}{icon} {title}{COLOR_RESET}")
        header_str = f" {COLOR_CYAN}{'Script':<15} {'LTP':<8} |"
        for mult in HA_ATR_MULTIPLIERS:
            header_str += f"  {'BB-RSI ' + str(mult) + 'X':^11} {'BB-MACD ' + str(mult) + 'X':^12} {'BB-DI ' + str(mult) + 'X':^9} |"
        header_str += f" {'Anchor':^8} | {'Tracker State':^15}"
        print(header_str + COLOR_RESET)
        print("-" * len(ANSI_RE.sub("", header_str)))

        for row in data_list:
            row_str = f" {row['Symbol']:<15} {row['LTP']:<8.2f} |"
            for mult in HA_ATR_MULTIPLIERS:
                gtag = f"{mult}X"
                row_str += (f"  {format_cell(row.get(f'BB_RSI_{gtag}'), 11)}"
                            f" {format_cell(row.get(f'BB_MACD_{gtag}'), 12)}"
                            f" {format_cell(row.get(f'ADX_{gtag}'), 9)} |")
            
            state = row['State']
            state_text = f"{COLOR_GREEN_BG} {state} {COLOR_RESET}" if "BUY]" in state else f"{COLOR_RED_BG} {state} {COLOR_RESET}" if "SELL]" in state else f"{COLOR_YELLOW} {state} {COLOR_RESET}"
            
            row_str += f" {row['ActiveAnchor']:^8} | {state_text}"
            print(row_str)

    print_basket("TOP BULL SETUPS (Valid Anchors Surviving)", "🔥", bulls)
    print_basket("TOP BEAR SETUPS (Valid Anchors Surviving)", "🩸", bears)
    
    if rejected:
        print(f"\n{COLOR_BOLD}🚫 THE GRAVEYARD (Filtered / Rejected Stocks){COLOR_RESET}")
        header_str = f" {COLOR_CYAN}{'Script':<15} {'LTP':<8} | {'Forensic Rejection Reason'}"
        print(header_str + COLOR_RESET)
        print("-" * 90)
        
        # Sort so breached anchors appear first, then warzones, then flatlines
        def sort_reason(r):
            reason = r['RejectReason']
            if "breached" in reason: return 0
            if "Warzone" in reason: return 1
            if "Kinetic" in reason: return 2
            return 3
            
        rejected.sort(key=lambda x: (sort_reason(x), x['Symbol']))
        
        for row in rejected:
            print(f" {row['Symbol']:<15} {row['LTP']:<8.2f} | {COLOR_YELLOW}{row['RejectReason']}{COLOR_RESET}")
    
    total_calls = sum(l.total_calls for l in LIMITERS.values())
    print(f"\n⏱️ Tracker sync completed in {(time.time() - t_start):.2f} seconds ({total_calls} API calls).\n")

def parse_args():
    p = argparse.ArgumentParser(description="Strict institutional volatility tracker (Upstox)")
    p.add_argument("--mode", choices=["STOCK_FNO", "CASH_EQUITY", "INDEX_OPTIONS"], default=TRADING_MODE)
    p.add_argument("--days", type=int, default=BACKTRACE_DAYS, help="trading sessions of history")
    p.add_argument("--date", type=str, default=None, help="Target date (YYYY-MM-DD)")
    p.add_argument("--time", type=str, default="15:30", help="Target time (HH:MM). Defaults to 15:30.")
    return p.parse_args()

if __name__ == "__main__":
    if not os.environ.get("UPSTOX_ACCESS_TOKEN"):
        print(f"{COLOR_RED_FG}[!] Missing UPSTOX_ACCESS_TOKEN.{COLOR_RESET}"); sys.exit(1)
    args = parse_args()
    run_screener(args.mode, args.days, args.date, args.time)
