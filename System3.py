#!/usr/bin/env python3
"""
Strict Institutional Volatility Tracker (Upstox) - APEX TRIPWIRE EDITION
+ Continuous Kinetic Math: Indicators calculated smoothly on 1-min arrays to prevent EMA time-warping.
+ Directional ATR Tripwires: Wicks are ignored. Indicators are only sampled when Close pushes 1 ATR.
+ Trailing Stop-Loss Floor: Trade survives indefinitely until Close drops 1 full ATR from the peak + confirmed reversal.
+ Gap & Grind Fix: Intraday % Change explicitly calculated and sorted to keep Top Gainers at the top.
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

REQUIRE_BB_KC_PIERCE = False

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
MIN1_HISTORY_DAYS = 15

RSI_PERIOD = 14
BB_PERIOD = 20
BB_STD = 1.0
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
# 2. CONTINUOUS KINETIC TRIPWIRE ENGINE 
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

def _ewm(x, alpha):
    xs = x.tolist(); out = [0.0] * len(xs); prev = out[0] = xs[0]
    keep = 1.0 - alpha
    for i in range(1, len(xs)): out[i] = prev = keep * prev + alpha * xs[i]
    return np.asarray(out)

def _evaluate_kinetic_arrays(close, high, low):
    n = len(close)
    delta = np.diff(close, prepend=close[0])
    gain = _ewm(np.where(delta > 0, delta, 0.0), 1 / RSI_PERIOD)
    loss = _ewm(np.where(delta < 0, -delta, 0.0), 1 / RSI_PERIOD)
    rsi = 100 - (100 / (1 + (gain / (loss + 1e-8))))
    
    rsi_mean = pd.Series(rsi).rolling(BB_PERIOD, min_periods=1).mean().values
    rsi_std = pd.Series(rsi).rolling(BB_PERIOD, min_periods=1).std(ddof=0).values
    
    macd = _ewm(close, 2 / 13) - _ewm(close, 2 / 27)
    hist = macd - _ewm(macd, 2 / 10)
    h_mean = pd.Series(hist).rolling(BB_PERIOD, min_periods=1).mean().values
    h_std = pd.Series(hist).rolling(BB_PERIOD, min_periods=1).std(ddof=0).values
    
    up, down = np.zeros(n), np.zeros(n)
    up[1:] = high[1:] - high[:-1]; down[1:] = low[:-1] - low[1:]
    plus_dm = np.where((up > down) & (up > 0), up, 0.0)
    minus_dm = np.where((down > up) & (down > 0), down, 0.0)
    
    tr = (high - low).copy()
    tr[1:] = np.maximum(tr[1:], np.maximum(np.abs(high[1:] - close[:-1]), np.abs(low[1:] - close[:-1])))
    
    a = 1 / ADX_PERIOD; atr = _ewm(tr, a)
    plus_di = 100 * (_ewm(plus_dm, a) / (atr + 1e-8))
    minus_di = 100 * (_ewm(minus_dm, a) / (atr + 1e-8))
    
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-8)
    adx = _ewm(dx, a)
    
    p_mean = pd.Series(plus_di).rolling(BB_PERIOD, min_periods=1).mean().values
    p_std = pd.Series(plus_di).rolling(BB_PERIOD, min_periods=1).std(ddof=0).values
    m_mean = pd.Series(minus_di).rolling(BB_PERIOD, min_periods=1).mean().values
    m_std = pd.Series(minus_di).rolling(BB_PERIOD, min_periods=1).std(ddof=0).values
    
    return {
        'rsi': rsi, 'r_mean': rsi_mean, 'r_std': rsi_std,
        'hist': hist, 'h_mean': h_mean, 'h_std': h_std,
        'plus_di': plus_di, 'p_mean': p_mean, 'p_std': p_std,
        'minus_di': minus_di, 'm_mean': m_mean, 'm_std': m_std,
        'adx': adx
    }

def get_kinetics(kin_1m, idx):
    rsi = kin_1m['rsi'][idx]; r_mean = kin_1m['r_mean'][idx]; r_std = kin_1m['r_std'][idx]
    hist = kin_1m['hist'][idx]; h_mean = kin_1m['h_mean'][idx]; h_std = kin_1m['h_std'][idx]
    p_di = kin_1m['plus_di'][idx]; p_mean = kin_1m['p_mean'][idx]; p_std = kin_1m['p_std'][idx]
    m_di = kin_1m['minus_di'][idx]; m_mean = kin_1m['m_mean'][idx]; m_std = kin_1m['m_std'][idx]
    adx = kin_1m['adx'][idx]
    
    bull_rsi = r_std > 0 and rsi > r_mean + BB_STD * r_std
    bear_rsi = r_std > 0 and rsi < r_mean - BB_STD * r_std
    bull_macd = h_std > 0 and hist > h_mean + BB_STD * h_std
    bear_macd = h_std > 0 and hist < h_mean - BB_STD * h_std
    bull_di = p_std > 0 and p_di > p_mean + BB_STD * p_std and p_di > m_di and adx >= ADX_THRESHOLD
    bear_di = m_std > 0 and m_di > m_mean + BB_STD * m_std and m_di > p_di and adx >= ADX_THRESHOLD
    
    raw_bull_power = p_std > 0 and p_di > p_mean + BB_STD * p_std
    raw_bear_power = m_std > 0 and m_di > m_mean + BB_STD * m_std
    
    bull_score = (1 if bull_rsi else 0) + (1 if bull_macd else 0) + (1 if bull_di else 0)
    bear_score = (1 if bear_rsi else 0) + (1 if bear_macd else 0) + (1 if bear_di else 0)
    
    return bull_score, bear_score, raw_bull_power, raw_bear_power, bull_rsi, bear_rsi, bull_macd, bear_macd, bull_di, bear_di

def get_tripwire_state(close, kin_1m, base_atr, mult, today_start):
    curr_open = close[today_start]
    last_rsi, last_macd, last_adx = "Neutral", "Neutral", "Neutral"
    blocks = 0
    for i in range(today_start, len(close)):
        target = base_atr * mult
        if close[i] - curr_open >= target:
            bull_score, bear_score, _, _, b_rsi, br_rsi, b_macd, br_macd, b_di, br_di = get_kinetics(kin_1m, i)
            last_rsi = "Buy" if b_rsi else "Sell" if br_rsi else "Neutral"
            last_macd = "Buy" if b_macd else "Sell" if br_macd else "Neutral"
            last_adx = "Buy" if b_di else "Sell" if br_di else "Neutral"
            blocks += 1
            curr_open = close[i]
        elif curr_open - close[i] >= target:
            bull_score, bear_score, _, _, b_rsi, br_rsi, b_macd, br_macd, b_di, br_di = get_kinetics(kin_1m, i)
            last_rsi = "Buy" if b_rsi else "Sell" if br_rsi else "Neutral"
            last_macd = "Buy" if b_macd else "Sell" if br_macd else "Neutral"
            last_adx = "Buy" if b_di else "Sell" if br_di else "Neutral"
            blocks += 1
            curr_open = close[i]
    return last_rsi, last_macd, last_adx, blocks

def evaluate_anchor_tripwire(close, dt_1m, kin_1m, base_atr, bb_upper, bb_lower, kc_upper, kc_lower, today_start):
    if today_start >= len(close): return None, "No data for today."
    
    i = today_start
    curr_open = close[today_start]
    survived_anchor = None
    last_reject_reason = "Failed Kinetic Alignment."
    warzone_kills = 0
    
    while i < len(close):
        anchor = None
        
        # 1. Search for Anchor (Wait for 1 ATR Directional Close)
        while i < len(close):
            bull_score, bear_score, r_bull, r_bear, _, _, _, _, _, _ = get_kinetics(kin_1m, i)
            
            if REQUIRE_BB_KC_PIERCE:
                bb_kc_bull_fire = bb_upper[i] > kc_upper[i]
                bb_kc_bear_fire = bb_lower[i] < kc_lower[i]
            else:
                bb_kc_bull_fire = bb_kc_bear_fire = True

            if close[i] - curr_open >= base_atr:
                # 1 ATR Bullish Tripwire snapped! Take Kinetic Snapshot.
                if bull_score >= 2 and bb_kc_bull_fire:
                    if r_bear: # Warzone
                        warzone_kills += 1
                        curr_open = close[i]
                    else:
                        anchor = {"dir": "BULL", "idx": i, "time": dt_1m[i], "price": close[i]}
                        break
                else:
                    curr_open = close[i]
                    
            elif curr_open - close[i] >= base_atr:
                # 1 ATR Bearish Tripwire snapped! Take Kinetic Snapshot.
                if bear_score >= 2 and bb_kc_bear_fire:
                    if r_bull: # Warzone
                        warzone_kills += 1
                        curr_open = close[i]
                    else:
                        anchor = {"dir": "BEAR", "idx": i, "time": dt_1m[i], "price": close[i]}
                        break
                else:
                    curr_open = close[i]
                    
            i += 1
            
        if not anchor:
            break # Reached the end of the day without finding anything
            
        # 2. Track Anchor Survival (Trailing Stop-Loss Floor)
        survived = True
        peak_price = anchor['price']
        j = anchor['idx'] + 1
        
        while j < len(close):
            bull_score, bear_score, _, _, _, _, _, _, _, _ = get_kinetics(kin_1m, j)
            
            if anchor['dir'] == "BULL":
                peak_price = max(peak_price, close[j])
                # Downward SL Tripwire: 1 ATR drop from highest peak
                if peak_price - close[j] >= base_atr: 
                    if bear_score >= 2:
                        b_time = pd.to_datetime(dt_1m[j]).strftime('%H:%M')
                        a_time = pd.to_datetime(anchor['time']).strftime('%H:%M')
                        last_reject_reason = f"Anchor at {a_time} killed at {b_time} (Kinetic SL after 1 ATR pullback)."
                        survived = False
                        curr_open = close[j] # Restart search
                        i = j + 1
                        break
                    else:
                        peak_price = close[j] # Survived the dip! Reset peak to avoid spamming
                        
            elif anchor['dir'] == "BEAR":
                peak_price = min(peak_price, close[j])
                # Upward SL Tripwire: 1 ATR rally from lowest trough
                if close[j] - peak_price >= base_atr: 
                    if bull_score >= 2:
                        b_time = pd.to_datetime(dt_1m[j]).strftime('%H:%M')
                        a_time = pd.to_datetime(anchor['time']).strftime('%H:%M')
                        last_reject_reason = f"Anchor at {a_time} killed at {b_time} (Kinetic SL after 1 ATR rally)."
                        survived = False
                        curr_open = close[j] # Restart search
                        i = j + 1
                        break
                    else:
                        peak_price = close[j] # Survived the rally! Reset trough
            j += 1
            
        if survived:
            survived_anchor = anchor
            break
            
    if survived_anchor:
        return survived_anchor, "Survived"
        
    if warzone_kills > 0 and "killed at" not in last_reject_reason:
        return None, f"Killed by Inverted Warzone Chop ({warzone_kills} attempts)."
        
    return None, last_reject_reason

def compute_row(symbol, master_1m):
    close, high, low, dt = master_1m['Close'].values, master_1m['High'].values, master_1m['Low'].values, master_1m['Datetime'].values
    
    # Locate today's start index
    today = pd.to_datetime(dt[-1]).date()
    today_start_idx = master_1m.index[master_1m['Datetime'].dt.date == today][0]
    
    # Calculate Day % Change (Current Close vs Today's Open)
    day_open_price = master_1m['Open'].iloc[today_start_idx]
    intraday_pct = ((close[-1] - day_open_price) / day_open_price) * 100
    
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
    
    kin_1m = _evaluate_kinetic_arrays(close, high, low)
    base_atr = compute_base_atr(master_1m)
    
    row = {'Symbol': symbol, 'LTP': float(close[-1]), 'DayChangePct': intraday_pct,
           'ActiveAnchor': None, 'AnchorDir': "NONE", 'State': "NONE", 'Blocks': 0, 'RejectReason': ""}
    
    anchor, reject_reason = evaluate_anchor_tripwire(close, dt, kin_1m, base_atr, bb_upper, bb_lower, kc_upper, kc_lower, today_start_idx)
    row['RejectReason'] = reject_reason

    if anchor:
        row['ActiveAnchor'] = pd.to_datetime(anchor['time']).strftime("%H:%M")
        row['AnchorDir'] = anchor['dir']
        
        if anchor['dir'] == "BULL":
            row['State'] = "[ACTIVE BUY]" if bb_upper[-1] > kc_upper[-1] else "[COILING]"
        else:
            row['State'] = "[ACTIVE SELL]" if bb_lower[-1] < kc_lower[-1] else "[COILING]"
            
        for mult in HA_ATR_MULTIPLIERS:
            rsi_st, macd_st, adx_st, blk_count = get_tripwire_state(close, kin_1m, base_atr, mult, today_start_idx)
            gtag = f"{mult}X"
            row[f'BB_RSI_{gtag}'] = rsi_st
            row[f'BB_MACD_{gtag}'] = macd_st
            row[f'ADX_{gtag}'] = adx_st
            if mult == 1:
                row['Blocks'] = blk_count

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
    item, cutoff_dt, is_live, history_days, progress = args
    try:
        target_dt = cutoff_dt.date()
        start_dt = target_dt - timedelta(days=history_days)
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
        results = [r for r in ex.map(process_stock_1m, [(item, cutoff_dt, is_live, MIN1_HISTORY_DAYS, prog) for item in candidates]) if r is not None]
    prog.done()

    if not results: return

    bulls = [r for r in results if r['AnchorDir'] == "BULL"]
    bears = [r for r in results if r['AnchorDir'] == "BEAR"]
    rejected = [r for r in results if r['AnchorDir'] == "NONE"]

    # --- THE SORTING TRAP FIXED --- 
    # Top % Gainers with ACTIVE BUY status rise to the very top.
    bulls.sort(key=lambda r: (r['State'] != "[ACTIVE BUY]", -r['DayChangePct']))
    bears.sort(key=lambda r: (r['State'] != "[ACTIVE SELL]", r['DayChangePct']))
    
    bulls, bears = bulls[:TOP_N_BUYERS], bears[:TOP_N_SELLERS]

    print(f"{COLOR_BOLD}\n=== STATEFUL INSTITUTIONAL VOLATILITY TRACKER [{mode}] ==={COLOR_RESET}")
    print(f"Time Machine: {cutoff_dt.strftime('%Y-%m-%d %H:%M')} | Scanned: {len(results)}\n")

    def print_basket(title, icon, data_list):
        if not data_list: return
        print(f"\n{COLOR_BOLD}{icon} {title}{COLOR_RESET}")
        header_str = f" {COLOR_CYAN}{'Script':<15} {'LTP':<8} {'Day%':<6} |"
        for mult in HA_ATR_MULTIPLIERS:
            header_str += f"  {'BB-RSI ' + str(mult) + 'X':^11} {'BB-MACD ' + str(mult) + 'X':^12} {'BB-DI ' + str(mult) + 'X':^9} |"
        header_str += f" {'Anchor':^8} | {'Tracker State':^15}"
        print(header_str + COLOR_RESET)
        print("-" * len(ANSI_RE.sub("", header_str)))

        for row in data_list:
            day_pct = f"{row['DayChangePct']:>5.2f}%"
            row_str = f" {row['Symbol']:<15} {row['LTP']:<8.2f} {day_pct:<6} |"
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
        header_str = f" {COLOR_CYAN}{'Script':<15} {'LTP':<8} {'Day%':<6} | {'Forensic Rejection Reason'}"
        print(header_str + COLOR_RESET)
        print("-" * 95)
        
        def sort_reason(r):
            reason = r['RejectReason']
            if "Kinetic SL" in reason: return 0
            if "Warzone" in reason: return 1
            if "Alignment" in reason: return 2
            return 3
            
        rejected.sort(key=lambda x: (sort_reason(x), -x['DayChangePct']))
        
        for row in rejected:
            day_pct = f"{row['DayChangePct']:>5.2f}%"
            print(f" {row['Symbol']:<15} {row['LTP']:<8.2f} {day_pct:<6} | {COLOR_YELLOW}{row['RejectReason']}{COLOR_RESET}")
    
    total_calls = sum(l.total_calls for l in LIMITERS.values())
    print(f"\n⏱️ Tracker sync completed in {(time.time() - t_start):.2f} seconds ({total_calls} API calls).\n")

def parse_args():
    p = argparse.ArgumentParser(description="Strict institutional volatility tracker (Upstox)")
    p.add_argument("--mode", choices=["STOCK_FNO", "CASH_EQUITY", "INDEX_OPTIONS"], default=TRADING_MODE)
    p.add_argument("--days", type=int, default=BACKTRACE_DAYS, help="trading sessions of history (daily prefilter only)")
    p.add_argument("--date", type=str, default=None, help="Target date (YYYY-MM-DD)")
    p.add_argument("--time", type=str, default="15:30", help="Target time (HH:MM). Defaults to 15:30.")
    p.add_argument("--history-days", type=int, default=MIN1_HISTORY_DAYS,
                    help=f"Calendar days of 1-minute history to fetch for indicator math (default: {MIN1_HISTORY_DAYS}).")
    p.add_argument("--disable-bb-kc-gate", action="store_true",
                    help="Drop the 'Bollinger Band already pierced Keltner Channel' requirement.")
    return p.parse_args()

if __name__ == "__main__":
    if not os.environ.get("UPSTOX_ACCESS_TOKEN"):
        print(f"{COLOR_RED_FG}[!] Missing UPSTOX_ACCESS_TOKEN.{COLOR_RESET}"); sys.exit(1)
    args = parse_args()
    MIN1_HISTORY_DAYS = args.history_days
    if args.disable_bb_kc_gate:
        REQUIRE_BB_KC_PIERCE = False
    run_screener(args.mode, args.days, args.date, args.time)
