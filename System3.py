#!/usr/bin/env python3
"""
Strict Institutional Volatility Screener (Upstox) - STATELESS INTRADAY EDITION
+ Zero Disk Caching (100% Live API fetches)
+ 1X Mandatory Anchor Rule (No hollow core breakouts)
+ Two-Tier Prime Sorting: Confluence (Blocks) -> Freshness (BB/KC % Crossover)
+ High-Res Intraday Optimizations (1-min candles, 15-min ATR baseline)
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

# --- OPTIONS CHAIN CONFIGURATION ---
EXPIRY_OFFSET = 0          
STRIKES_FROM_ATM = 5       
OPT_MIN_PRICE = 30
OPT_MIN_VOLUME = 10000

# --- DECOUPLED HA-ATR ENGINE MULTIPLIERS ---
HA_ATR_MULTIPLIERS = [1, 2, 3, 5]
ATR_BASIS_PERIOD = 14
ATR_BASIS_TF = "15min"      # 15-minute baseline for Intraday Institutional Volume
MIN_ATR_PCT = 0.001

# --- CLEAN-SURGE / DIRTY-BLOCK AUDIT ---
# While a range block is being built from 1-minute candles, track the worst move
# against the emerging trend (peak->trough for an up-move, trough->peak for a
# down-move). If that internal reverse move reaches this fraction of the Base
# ATR before the block closes, the block is "Dirty/Exhausted" and its Buy/Sell
# signal is killed, regardless of how the block eventually closed.
DIRTY_MOVE_ATR_FRACTION = 0.4

# --- OUTPUT LIMITS & CONFLUENCE ---
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

# --- EQUITY UNIVERSE FILTERING ---
MIN_PRICE = 100
MAX_PRICE = 5000
MIN_DAILY_VOLUME = 100000
BACKTRACE_DAYS = 30        # Reduced for pure intraday speed

# --- INDICATOR PERIODS ---
RSI_PERIOD = 14
BB_PERIOD = 20
BB_STD = 1
ADX_PERIOD = 14
ADX_THRESHOLD = 20

# --- SPEED KNOBS ---
WORKERS = 16                       
INCLUDE_NON_EQ_SERIES = False      
PREFILTER_PRICE_SLACK = 0.25
PREFILTER_MIN_TODAY_VOLUME = 0     
PREFILTER_MIN_ABS_MOVE_PCT = 0.0   

# --- API RATE LIMITS ---
RATE_CAPS = ((1.0, 22), (60.0, 220), (1800.0, 900))

API_HOST = "https://api.upstox.com"
IST = timezone(timedelta(hours=5, minutes=30))
SESSION_OPEN_MIN = 9 * 60 + 15
SESSION_CLOSE_MIN = 15 * 60 + 30


# ==============================================================================
# HELPERS: clock, budgeted rate limiter, HTTP, progress
# ==============================================================================
def now_ist():
    return datetime.now(IST).replace(tzinfo=None)

def market_is_open():
    n = now_ist()
    m = n.hour * 60 + n.minute
    return n.weekday() < 5 and SESSION_OPEN_MIN <= m < SESSION_CLOSE_MIN

class BudgetLimiter:
    """In-memory sliding window rate limiter. Loses history on script exit."""
    def __init__(self, caps):
        self.caps = caps
        self.horizon = max(s for s, _ in caps)
        self.lock = threading.Lock()
        self.stamps = []
        self.block_until = 0.0
        self.last_notice = 0.0
        self.total_calls = 0

    def used(self, span):
        with self.lock:
            return len(self.stamps) - bisect_left(self.stamps, time.time() - span)

    def penalize(self, seconds):
        with self.lock:
            self.block_until = max(self.block_until, time.time() + seconds)

    def acquire(self):
        while True:
            notice = None
            with self.lock:
                now = time.time()
                cut = bisect_left(self.stamps, now - self.horizon)
                if cut:
                    del self.stamps[:cut]
                n = len(self.stamps)
                wait = max(0.0, self.block_until - now)
                for span, cap in self.caps:
                    if n >= cap and n - bisect_left(self.stamps, now - span) >= cap:
                        wait = max(wait, self.stamps[n - cap] + span - now)
                if wait <= 0:
                    self.stamps.append(now)
                    self.total_calls += 1
                    return
                if wait > 5 and now - self.last_notice > 20:
                    self.last_notice = now
                    notice = (n, wait)
            if notice:
                print(f"\n   ⏳ Upstox rate-limit window is full ({notice[0]} calls in the last 30 min); "
                      f"waiting {notice[1]:.0f}s to stay under the cap...", file=sys.stderr, flush=True)
            time.sleep(min(wait, 1.0) + 0.005)

class FetchStats:
    def __init__(self):
        self.lock = threading.Lock()
        self.failed = 0
        self.auth_failed = False

    def fail(self):
        with self.lock:
            self.failed += 1

    def mark_auth_failed(self):
        with self.lock:
            self.auth_failed = True

STATS = FetchStats()
LIMITERS = {
    "quotes": BudgetLimiter(RATE_CAPS),
    "history": BudgetLimiter(RATE_CAPS),
    "intraday": BudgetLimiter(RATE_CAPS),
}
LIMITER = LIMITERS["history"]   
_TLS = threading.local()

def _limiter_for(url):
    if "market-quote/quotes" in url:
        return LIMITERS["quotes"]
    if "/intraday/" in url:
        return LIMITERS["intraday"]
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
        self.label, self.total, self.n = label, total, 0
        self.step = max(1, total // 20)
        self.lock = threading.Lock()

    def tick(self):
        with self.lock:
            self.n += 1
            if self.n == self.total or self.n % self.step == 0:
                print(f"\r   {self.label}: {self.n}/{self.total}", end="", file=sys.stderr, flush=True)

    def done(self):
        print("", file=sys.stderr)

# ==============================================================================
# 1. UPSTOX API: quotes, candles, universe & dynamic options strikes
# ==============================================================================
def _get(url, params=None, retries=4):
    token = os.environ.get("UPSTOX_ACCESS_TOKEN")
    if not token or STATS.auth_failed:
        return 401, None
    headers = {'Accept': 'application/json', 'Authorization': f'Bearer {token}'}
    limiter = _limiter_for(url)
    for attempt in range(retries):
        limiter.acquire()
        try:
            r = _session().get(url, headers=headers, params=params, timeout=20)
        except requests.RequestException:
            time.sleep(0.5 * (attempt + 1))
            continue
        code = r.status_code
        if code == 200:
            try:
                return 200, r.json()
            except ValueError:
                time.sleep(0.5 * (attempt + 1))
                continue
        if code == 401:
            STATS.mark_auth_failed()
            return 401, None
        if code == 429 or code >= 500:
            limiter.penalize(min(3.0 * (attempt + 1), 15.0))
            continue
        return code, None
    STATS.fail()
    return 0, None

def _to_frame(candles):
    if not candles:
        return None
    df = pd.DataFrame(candles).iloc[:, :6]
    df.columns = ['Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume']
    ts = df['Timestamp'].astype(str)
    try:
        if not ts.str.endswith('+05:30').all():
            raise ValueError("non-IST offset")
        df['Datetime'] = pd.to_datetime(ts.str.slice(0, 19), format="%Y-%m-%dT%H:%M:%S")
    except ValueError:
        df['Datetime'] = pd.to_datetime(ts, utc=True).dt.tz_convert(IST).dt.tz_localize(None)
    for c in ('Open', 'High', 'Low', 'Close', 'Volume'):
        df[c] = df[c].astype(float)
    df = (df.drop(columns='Timestamp').drop_duplicates(subset='Datetime')
            .sort_values('Datetime').reset_index(drop=True))
    return df if not df.empty else None

def _candles(url):
    status, js = _get(url)
    if status != 200 or not js:
        return status, None
    return 200, _to_frame((js.get('data') or {}).get('candles') or [])

def _url_range(key, start, end):       
    return (f"{API_HOST}/v2/historical-candle/{urllib.parse.quote(key)}/day/"
            f"{end:%Y-%m-%d}/{start:%Y-%m-%d}")

def _url_intraday(key):
    # INTRADAY OPTIMIZATION: Switched to /1minute for precise range bar resolution
    return f"{API_HOST}/v2/historical-candle/intraday/{urllib.parse.quote(key)}/1minute"

def _date_chunks(start, end, span=365):
    cur = end
    while cur >= start:
        c_start = max(start, cur - timedelta(days=span - 1))
        yield c_start, cur
        cur = c_start - timedelta(days=1)

def fetch_quotes(items, batch=200):
    batches = [items[i:i + batch] for i in range(0, len(items), batch)]
    def one(b):
        status, js = _get(f"{API_HOST}/v2/market-quote/quotes",
                          params={"instrument_key": ",".join(x['key'] for x in b)})
        if status != 200 or not js:
            return {}
        by_ts = {f"{x['key'].split('|')[0]}:{x['symbol']}": x['key'] for x in b}
        out = {}
        for k, v in (js.get('data') or {}).items():
            key = v.get('instrument_token') or by_ts.get(k)
            if key:
                out[key] = {'ltp': v.get('last_price') or 0.0, 'vol': v.get('volume') or 0,
                            'net': v.get('net_change') or 0.0}
        return out

    res = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(WORKERS, max(1, len(batches)))) as ex:
        for part in ex.map(one, batches):
            res.update(part)
    return res

def fetch_today(key):
    if now_ist().weekday() >= 5:
        return None
    status, df = _candles(_url_intraday(key))
    if status != 200 or df is None:
        return None
    df = df[df['Datetime'].dt.date == now_ist().date()]     
    return df if not df.empty else None

BASE_COLS = ['Datetime', 'Open', 'High', 'Low', 'Close', 'Volume']

def prepare_master(dfs):
    master = (pd.concat([d[BASE_COLS] for d in dfs], ignore_index=True)
              .drop_duplicates(subset='Datetime')
              .sort_values('Datetime').reset_index(drop=True))
    dt = master['Datetime']
    day = dt.dt.normalize()
    master['Min'] = dt.dt.hour.values * 60 + dt.dt.minute.values
    master['SessId'] = day.values.astype('datetime64[D]').astype('int64')
    master['Session'] = day.dt.date
    return master

def keep_last_sessions(master, days):
    sessions = sorted(master['Session'].unique())[-days:]
    return master[master['Session'].isin(sessions)].reset_index(drop=True)

def load_history(item, start, end, days):
    frames = []
    for c_start, c_end in _date_chunks(start, end):
        status, df = _candles(_url_range(item['key'], c_start, c_end))
        if status != 200:
            return None                     
        if df is not None:
            frames.append(df)
    if not frames:
        return None
    hist = keep_last_sessions(prepare_master(frames), days)
    if hist.empty:
        return None
    return hist

# ---------------- universe -----------------------------------------------------
INDEX_CONFIG = {
    "NIFTY": {"spot_key": "NSE_INDEX|Nifty 50", "step": 50, "match": ["NIFTY", "NIFTY 50"]},
    "BANKNIFTY": {"spot_key": "NSE_INDEX|Nifty Bank", "step": 100, "match": ["BANKNIFTY", "NIFTY BANK"]},
    "FINNIFTY": {"spot_key": "NSE_INDEX|Nifty Fin Service", "step": 50, "match": ["FINNIFTY", "NIFTY FIN SERVICE"]},
    "MIDCPNIFTY": {"spot_key": "NSE_INDEX|NIFTY MID SELECT", "step": 25, "match": ["MIDCPNIFTY", "NIFTY MID SELECT"]},
    "SENSEX": {"spot_key": "BSE_INDEX|SENSEX", "step": 100, "match": ["SENSEX", "BSE SENSEX"]},
}

def _download_master(name):
    url = f"https://assets.upstox.com/market-quote/instruments/exchange/{name}.json.gz"
    for attempt in range(3):
        try:
            resp = requests.get(url, timeout=90)
            if resp.status_code == 200:
                return json.load(gzip.GzipFile(fileobj=io.BytesIO(resp.content)))
        except Exception as e:
            print(f"{COLOR_RED_FG}[API Error] {name} master attempt {attempt + 1}: {e}{COLOR_RESET}")
        time.sleep(1.0 * (attempt + 1))
    return None

def _equity_universe(mode):
    print("🔄 Downloading NSE instrument master Live...")
    nse = _download_master("NSE")
    if not nse:
        return []

    def ts_of(i):
        return i.get("tradingsymbol", i.get("trading_symbol"))

    def plain(i):
        return (i.get("segment") == "NSE_EQ" and ts_of(i) and i.get("instrument_key")
                and (INCLUDE_NON_EQ_SERIES or (i.get("instrument_type") or "EQ") == "EQ"))

    fno = {i.get("underlying_symbol") for i in nse if i.get("segment") == "NSE_FO" and i.get("underlying_symbol")}
    if mode == "STOCK_FNO":
        rows = [i for i in nse if plain(i) and ts_of(i) in fno]
    else:
        rows = [i for i in nse if plain(i) and ts_of(i) not in fno]
    universe = list({i["instrument_key"]: {"symbol": ts_of(i), "key": i["instrument_key"]} for i in rows}.values())
    return universe

def _option_master():
    print("🔄 Downloading NSE & BSE instrument masters Live...")
    nse, bse = _download_master("NSE"), _download_master("BSE")
    if not nse or not bse:
        return {}
    name_to_idx = {}
    for idx, cfg in INDEX_CONFIG.items():
        for m in cfg["match"]:
            name_to_idx[m] = idx

    out = {idx: [] for idx in INDEX_CONFIG}
    for item in nse + bse:                                       
        if str(item.get("instrument_type", "")).upper() not in ("CE", "PE"):
            continue
        idx = name_to_idx.get(str(item.get("name", "")).upper()) or \
              name_to_idx.get(str(item.get("underlying_symbol", "")).upper())
        if not idx:
            continue
        strike_val = item.get("strike", item.get("strike_price"))
        try:
            strike = int(float(strike_val)) if strike_val is not None else 0
        except (ValueError, TypeError):
            continue
        if strike > 0:
            out[idx].append({"strike": strike, "expiry": item.get("expiry"),
                             "ts": str(item.get("tradingsymbol", item.get("trading_symbol", ""))),
                             "key": item.get("instrument_key")})
    return out

def _parse_expiry(e):
    try:
        if isinstance(e, (int, float)) or (isinstance(e, str) and e.isdigit()):
            return datetime.fromtimestamp(int(e) / 1000.0, IST).replace(tzinfo=None)
        e_str = str(e).split('T')[0]
        for fmt in ("%Y-%m-%d", "%d-%b-%Y"):
            try:
                return datetime.strptime(e_str, fmt)
            except ValueError:
                pass
    except Exception:
        pass
    return datetime.max

def _options_universe():
    opt_master = _option_master()
    if not opt_master:
        print(f"{COLOR_RED_FG}[API Error] Could not build the option master.{COLOR_RESET}")
        return []

    spot_items = [{"symbol": n, "key": c["spot_key"]} for n, c in INDEX_CONFIG.items()]
    quotes = fetch_quotes(spot_items)
    print(f"🎯 Calculating ATM Strikes & Constructing Options Chain for {len(INDEX_CONFIG)} Indices...\n")

    universe, today = [], now_ist().date()
    for idx_name, cfg in INDEX_CONFIG.items():
        spot_price = (quotes.get(cfg["spot_key"]) or {}).get('ltp') or 0.0
        if spot_price <= 0:                                  
            df = fetch_today(cfg["spot_key"])
            if df is None:
                end = now_ist().date() - timedelta(days=1)
                df = load_history({"key": cfg["spot_key"]}, end - timedelta(days=6), end, 2)
            spot_price = float(df['Close'].iloc[-1]) if df is not None and not df.empty else 0.0
        if spot_price <= 0:
            print(f"   [!] {idx_name}: Could not fetch live spot price. Skipping.")
            continue

        step = cfg["step"]
        atm = int(round(spot_price / step) * step)
        targets = {int(atm + i * step) for i in range(-STRIKES_FROM_ATM, STRIKES_FROM_ATM + 1)}
        idx_opts = opt_master.get(idx_name, [])
        if not idx_opts:
            print(f"   [!] {idx_name}: Could not find option chain in JSON. Skipping.")
            continue

        exp = {}
        for o in idx_opts:
            if o.get("expiry") is not None and o["expiry"] not in exp:
                exp[o["expiry"]] = _parse_expiry(o["expiry"])
        live = sorted((dt, e) for e, dt in exp.items() if dt == datetime.max or dt.date() >= today)
        if len(live) <= EXPIRY_OFFSET:
            print(f"   [!] {idx_name}: Required expiry offset not available. Found {len(live)}. Skipping.")
            continue
        target_dt, target_expiry = live[EXPIRY_OFFSET]
        expiry_str = target_dt.strftime("%d-%b-%Y") if target_dt != datetime.max else str(target_expiry)

        matched = [o for o in idx_opts if o["expiry"] == target_expiry and o["strike"] in targets]
        print(f"   => {idx_name:<10} | Spot: {spot_price:<8.2f} | ATM: {atm:<6} | Expiry: {expiry_str} "
              f"| Grabbed {len(matched)} CE/PE Contracts")
        universe.extend({"symbol": o["ts"], "key": o["key"]} for o in matched)
    print("\n")
    return universe

def get_dynamic_universe(mode):
    if mode == "INDEX_OPTIONS":
        return _options_universe()
    if mode in ("STOCK_FNO", "CASH_EQUITY"):
        return _equity_universe(mode)
    return []

# ==============================================================================
# 2. THE DECOUPLED HA-ATR ENGINE 
# ==============================================================================
def compute_base_atr(m):
    hi, lo, cl = m['High'].values, m['Low'].values, m['Close'].values
    sid, mins = m['SessId'].values, m['Min'].values

    # Uses the configured ATR_BASIS_TF (now 15min)
    tf_minutes = int(ATR_BASIS_TF.replace("min", ""))
    bucket = sid * 100 + mins // tf_minutes
    starts = np.concatenate(([0], np.flatnonzero(np.diff(bucket)) + 1))
    ends = np.concatenate((starts[1:], [len(bucket)])) - 1
    b_high = np.maximum.reduceat(hi, starts)
    b_low = np.minimum.reduceat(lo, starts)
    b_close = cl[ends]
    if len(b_close) < 5:
        return max(cl[-1] * MIN_ATR_PCT, 0.01)

    prev_close = np.concatenate(([np.nan], b_close[:-1]))
    tr = np.fmax(np.fmax(b_high - b_low, np.abs(b_high - prev_close)), np.abs(b_low - prev_close))
    atr = float(tr[-ATR_BASIS_PERIOD:].mean())
    return max(atr, b_close[-1] * MIN_ATR_PCT, 0.01)

def split_sessions(m):
    sid = m['SessId'].values
    cuts = np.flatnonzero(np.diff(sid)) + 1
    starts = np.concatenate(([0], cuts))
    ends = np.concatenate((cuts, [len(sid)]))
    o, h, l, c = (m[k].tolist() for k in ('Open', 'High', 'Low', 'Close'))
    return [(o[s:e], h[s:e], l[s:e], c[s:e]) for s, e in zip(starts, ends)]

def build_isolated_range_bars(sessions, target_range, base_atr=None,
                               dirty_fraction=None):
    """
    Builds range bars from 1-minute (opens, highs, lows, closes) tuples per session.

    While each block accumulates, this also audits the *path* price took to get
    there (not just the fact that it closed):
      - block_max_high / low_since_max: tracks the running peak of the block and
        the lowest low seen since that peak -> "pullback" = how hard sellers hit
        the move on the way up.
      - block_min_low / high_since_min: mirror image for the downside -> "bounce"
        = how hard buyers hit the move on the way down.
    If pullback (or bounce) reaches dirty_fraction * base_atr before the block
    closes, that block is flagged Dirty for that direction. A block can be dirty
    for Bulls, dirty for Bears, both, or neither.
    """
    if dirty_fraction is None:
        dirty_fraction = DIRTY_MOVE_ATR_FRACTION      # re-read live so --dirty-fraction works

    B_O, B_H, B_L, B_C = [], [], [], []
    B_DIRTY_BULL, B_DIRTY_BEAR = [], []
    seg_start = 0
    dirty_thresh = (base_atr * dirty_fraction) if base_atr else None

    for opens, highs, lows, closes in sessions:
        if not closes:
            continue
        before = len(B_C)
        curr_O, curr_H, curr_L, curr_C = opens[0], highs[0], lows[0], closes[0]

        # Internal path trackers for the block currently being built
        blk_max_high, blk_low_since_max = curr_H, curr_L
        blk_min_low, blk_high_since_min = curr_L, curr_H
        dirty_bull = dirty_bear = False

        def _reset_trackers(o0, h0, l0):
            nonlocal blk_max_high, blk_low_since_max, blk_min_low, blk_high_since_min
            nonlocal dirty_bull, dirty_bear
            blk_max_high, blk_low_since_max = h0, l0
            blk_min_low, blk_high_since_min = l0, h0
            dirty_bull = dirty_bear = False

        for hi, lo, cl in zip(highs, lows, closes):
            if hi > curr_H:
                curr_H = hi
            if lo < curr_L:
                curr_L = lo
            curr_C = cl

            # --- audit the path for THIS candle before it can be folded away ---
            if hi >= blk_max_high:
                blk_max_high = hi
                blk_low_since_max = lo
            else:
                blk_low_since_max = min(blk_low_since_max, lo)
            if lo <= blk_min_low:
                blk_min_low = lo
                blk_high_since_min = hi
            else:
                blk_high_since_min = max(blk_high_since_min, hi)

            if dirty_thresh is not None:
                if (blk_max_high - blk_low_since_max) >= dirty_thresh:
                    dirty_bull = True
                if (blk_high_since_min - blk_min_low) >= dirty_thresh:
                    dirty_bear = True

            if curr_H - curr_L >= target_range:
                B_O.append(curr_O); B_H.append(curr_H); B_L.append(curr_L); B_C.append(curr_C)
                B_DIRTY_BULL.append(dirty_bull); B_DIRTY_BEAR.append(dirty_bear)
                curr_O = curr_H = curr_L = curr_C
                _reset_trackers(curr_O, curr_H, curr_L)
        if curr_H > curr_L:
            B_O.append(curr_O); B_H.append(curr_H); B_L.append(curr_L); B_C.append(curr_C)
            B_DIRTY_BULL.append(dirty_bull); B_DIRTY_BEAR.append(dirty_bear)
        if len(B_C) > before:
            seg_start = before

    if not B_C:
        return None

    o = B_O[seg_start:]; h = B_H[seg_start:]; l = B_L[seg_start:]; c = B_C[seg_start:]
    ha_close = [(o[i] + h[i] + l[i] + c[i]) / 4 for i in range(len(c))]
    ha_open = (o[0] + c[0]) / 2
    for i in range(1, len(c)):
        ha_open = (ha_open + ha_close[i - 1]) / 2.0
    ha_trend = 'Green' if ha_close[-1] >= ha_open else 'Red'

    return {'High': np.asarray(B_H), 'Low': np.asarray(B_L), 'Close': np.asarray(B_C),
            'HA_Trend': ha_trend,
            'Dirty_Bull': B_DIRTY_BULL[seg_start:][-1] if B_DIRTY_BULL[seg_start:] else False,
            'Dirty_Bear': B_DIRTY_BEAR[seg_start:][-1] if B_DIRTY_BEAR[seg_start:] else False}

def _ewm(x, alpha):
    xs = x.tolist()
    out = [0.0] * len(xs)
    prev = xs[0]
    out[0] = prev
    keep = 1.0 - alpha
    for i in range(1, len(xs)):
        prev = keep * prev + alpha * xs[i]
        out[i] = prev
    return np.asarray(out)

def _last_bb(series):
    w = series[-BB_PERIOD:]
    mean = float(w.mean())
    std = float(w.std(ddof=0)) if len(w) > 1 else 0.0
    return mean, std

def calculate_strict_signals(bars):
    if bars is None or len(bars['Close']) < 5:
        return "", "", "", "NONE"
    close, high, low = bars['Close'], bars['High'], bars['Low']
    n = len(close)

    delta = np.diff(close, prepend=close[0])
    gain = _ewm(np.where(delta > 0, delta, 0.0), 1 / RSI_PERIOD)
    loss = _ewm(np.where(delta < 0, -delta, 0.0), 1 / RSI_PERIOD)
    rsi = 100 - (100 / (1 + (gain / (loss + 1e-8))))
    r_mean, r_std = _last_bb(rsi)

    macd = _ewm(close, 2 / 13) - _ewm(close, 2 / 27)
    hist = macd - _ewm(macd, 2 / 10)
    h_mean, h_std = _last_bb(hist)

    up = np.zeros(n); down = np.zeros(n)
    up[1:] = high[1:] - high[:-1]
    down[1:] = low[:-1] - low[1:]
    plus_dm = np.where((up > down) & (up > 0), up, 0.0)
    minus_dm = np.where((down > up) & (down > 0), down, 0.0)
    
    tr = (high - low).copy()
    tr[1:] = np.maximum(tr[1:], np.maximum(np.abs(high[1:] - close[:-1]), np.abs(low[1:] - close[:-1])))

    a = 1 / ADX_PERIOD
    atr = _ewm(tr, a)
    plus_di = 100 * (_ewm(plus_dm, a) / (atr + 1e-8))
    minus_di = 100 * (_ewm(minus_dm, a) / (atr + 1e-8))
    adx = _ewm(100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-8), a)
    p_mean, p_std = _last_bb(plus_di)
    m_mean, m_std = _last_bb(minus_di)

    bb_rsi = "Neutral"
    if r_std > 0:
        if rsi[-1] > r_mean + BB_STD * r_std: bb_rsi = "Buy"
        elif rsi[-1] < r_mean - BB_STD * r_std: bb_rsi = "Sell"

    bb_macd = "Neutral"
    if h_std > 0:
        if hist[-1] > h_mean + BB_STD * h_std: bb_macd = "Buy"
        elif hist[-1] < h_mean - BB_STD * h_std: bb_macd = "Sell"

    adx_sig = "Neutral"
    if p_std > 0 and plus_di[-1] > p_mean + BB_STD * p_std and plus_di[-1] > minus_di[-1] and adx[-1] >= ADX_THRESHOLD:
        adx_sig = "Buy"
    elif m_std > 0 and minus_di[-1] > m_mean + BB_STD * m_std and minus_di[-1] > plus_di[-1] and adx[-1] >= ADX_THRESHOLD:
        adx_sig = "Sell"

    ha_trend = bars['HA_Trend']
    is_bull = bb_rsi == "Buy" and bb_macd == "Buy" and adx_sig == "Buy" and ha_trend == 'Green'
    is_bear = bb_rsi == "Sell" and bb_macd == "Sell" and adx_sig == "Sell" and ha_trend == 'Red'

    # CLEAN-SURGE AUDIT: a block that technically closed Bull/Bear but only did so
    # after a chaotic internal fight (price reversed >= DIRTY_MOVE_ATR_FRACTION of
    # Base ATR against the trend mid-block) is "Dirty/Exhausted" -> kill the signal.
    dirty_killed = False
    if is_bull and bars.get('Dirty_Bull'):
        is_bull = False
        dirty_killed = True
    if is_bear and bars.get('Dirty_Bear'):
        is_bear = False
        dirty_killed = True

    if is_bull:
        return bb_rsi, bb_macd, adx_sig, "BULL", dirty_killed
    if is_bear:
        return bb_rsi, bb_macd, adx_sig, "BEAR", dirty_killed
    return "", "", "", "NONE", dirty_killed


def compute_row(symbol, master_1m):
    close = master_1m['Close'].values
    high = master_1m['High'].values
    low = master_1m['Low'].values
    
    sma20 = pd.Series(close).rolling(20, min_periods=1).mean().values
    std20 = pd.Series(close).rolling(20, min_periods=1).std(ddof=0).values
    
    tr = np.zeros_like(close)
    tr[0] = high[0] - low[0]
    tr[1:] = np.maximum(high[1:] - low[1:], np.maximum(np.abs(high[1:] - close[:-1]), np.abs(low[1:] - close[:-1])))
    atr20 = pd.Series(tr).rolling(20, min_periods=1).mean().values
    
    bb_upper = sma20[-1] + 2.0 * std20[-1]
    bb_lower = sma20[-1] - 2.0 * std20[-1]
    kc_upper = sma20[-1] + 1.5 * atr20[-1]
    kc_lower = sma20[-1] - 1.5 * atr20[-1]
    
    # PERCENTAGE DELTA MATH (Normalizing output to a percentage scale instead of raw rupees)
    bull_bb_kc_delta = float((bb_upper - kc_upper) / kc_upper * 100) if kc_upper > 0 else 0.0
    bear_bb_kc_delta = float((kc_lower - bb_lower) / kc_lower * 100) if kc_lower > 0 else 0.0

    base_atr = compute_base_atr(master_1m)
    sessions = split_sessions(master_1m)
    row = {'Symbol': symbol, 'LTP': float(master_1m['Close'].iloc[-1]), 'LastSession': master_1m['Session'].iloc[-1],
           'Score': 0, 'PerfectBullBlocks': 0, 'PerfectBearBlocks': 0, 'DirtyKilled1X': False,
           'Bull_BB_KC_Delta': bull_bb_kc_delta, 'Bear_BB_KC_Delta': bear_bb_kc_delta}
    
    for mult in HA_ATR_MULTIPLIERS:
        gtag = f"{mult}X"
        bars = build_isolated_range_bars(sessions, base_atr * mult, base_atr=base_atr)
        bb_rsi, bb_macd, adx_sig, alignment, dirty_killed = calculate_strict_signals(bars)
        row[f'BB_RSI_{gtag}'], row[f'BB_MACD_{gtag}'], row[f'ADX_{gtag}'] = bb_rsi, bb_macd, adx_sig
        if alignment == "BULL":
            row['PerfectBullBlocks'] += 1
            row['Score'] += 1
        elif alignment == "BEAR":
            row['PerfectBearBlocks'] += 1
            row['Score'] -= 1
        if mult == 1 and dirty_killed:
            row['DirtyKilled1X'] = True
    
    return row

# ==============================================================================
# 3. PIPELINE EXECUTOR & UI DRAWING
# ==============================================================================
def format_cell(text, width=10):
    if not text: return " " * width
    spaces = width - len(str(text))
    left_pad, right_pad = " " * (spaces // 2), " " * (spaces - (spaces // 2))
    colored_text = (f"{COLOR_GREEN_BG}{text}{COLOR_RESET}" if text == "Buy"
                    else f"{COLOR_RED_BG}{text}{COLOR_RESET}" if text == "Sell" else str(text))
    return f"{left_pad}{colored_text}{right_pad}"


def prefilter_by_quotes(universe):
    quotes = fetch_quotes(universe)
    if len(quotes) < 0.5 * len(universe):
        print(f"{COLOR_YELLOW}⚠️ Quote pre-filter unavailable; falling back to full scan.{COLOR_RESET}")
        return universe
    lo, hi = MIN_PRICE * (1 - PREFILTER_PRICE_SLACK), MAX_PRICE * (1 + PREFILTER_PRICE_SLACK)
    live = market_is_open()
    keep = []
    for it in universe:
        q = quotes.get(it['key'])
        if not q or not (lo <= q['ltp'] <= hi):
            continue
        if live:
            if PREFILTER_MIN_TODAY_VOLUME and q['vol'] < PREFILTER_MIN_TODAY_VOLUME:
                continue
            if PREFILTER_MIN_ABS_MOVE_PCT:
                prev = q['ltp'] - q['net']
                if prev > 0 and abs(q['net']) / prev * 100 < PREFILTER_MIN_ABS_MOVE_PCT:
                    continue
        keep.append(it)
    return keep


def _history_worker(args):
    item, start, end, days, mode, progress = args
    try:
        hist = load_history(item, start, end, days)
        if hist is None or hist.empty:
            return None
        close, vol = hist['Close'].iloc[-1], hist['Volume'].sum()
        if mode == "INDEX_OPTIONS":
            ok = close >= OPT_MIN_PRICE and vol >= OPT_MIN_VOLUME
        else:
            ok = MIN_PRICE <= close <= MAX_PRICE and vol >= MIN_DAILY_VOLUME
        return (item, hist) if ok else None
    except Exception as e:
        print(f"\n{COLOR_YELLOW}[history] {item['symbol']}: {e}{COLOR_RESET}", file=sys.stderr)
        return None
    finally:
        progress.tick()


def process_stock(args):
    item, hist, days, progress = args
    try:
        today_df = fetch_today(item['key'])
        frames = [hist] + ([today_df] if today_df is not None else [])
        master_1m = keep_last_sessions(prepare_master(frames), days)
        if len(master_1m) < 30:
            return None
        return compute_row(item['symbol'], master_1m)
    except Exception as e:
        print(f"\n{COLOR_YELLOW}[process] {item['symbol']}: {e}{COLOR_RESET}", file=sys.stderr)
        return None
    finally:
        progress.tick()


def _eta(calls):
    per_min = min(cap for span, cap in RATE_CAPS if span == 60.0)
    return f"~{max(1, round(calls / per_min * 60))}s" if calls < per_min else f"~{calls / per_min:.1f} min"


def run_screener(mode=TRADING_MODE, days=BACKTRACE_DAYS, min_blocks=MIN_PERFECT_BLOCKS):
    t_start = time.time()
    days = max(days, 2)
    print(f"\n{COLOR_CYAN}📡 Initializing Screener Pipeline [{mode}] via UPSTOX API...{COLOR_RESET}")

    universe_raw = get_dynamic_universe(mode)
    if STATS.auth_failed:
        print(f"{COLOR_RED_FG}[Auth Error] Upstox rejected the token (401). Refresh UPSTOX_ACCESS_TOKEN.{COLOR_RESET}")
        return
    if not universe_raw:
        print(f"{COLOR_RED_FG}[!] Failed to generate the Universe list. Exiting.{COLOR_RESET}")
        return

    end = now_ist().date() - timedelta(days=1)
    start = end - timedelta(days=days * 2 + 6)

    candidates = universe_raw
    if mode != "INDEX_OPTIONS":
        candidates = prefilter_by_quotes(universe_raw)
        print(f"⚡ Quote pre-filter: {len(universe_raw)} -> {len(candidates)} instruments "
              f"({-(-len(universe_raw) // 100)} API calls)")
        if STATS.auth_failed:
            print(f"{COLOR_RED_FG}[Auth Error] Upstox rejected the token (401). Refresh UPSTOX_ACCESS_TOKEN.{COLOR_RESET}")
            return

    what = "Options Strikes for Minimum Premium & Liquidity" if mode == "INDEX_OPTIONS" \
        else "stocks for Volume & Price constraints"
    print(f"🔄 Fetching Live History for {len(candidates)} {what}...  [{_eta(len(candidates))}]")
    prog = Progress("history", len(candidates))
    work = [(it, start, end, days, mode, prog) for it in candidates]
    with concurrent.futures.ThreadPoolExecutor(max_workers=WORKERS) as ex:
        stage1 = [r for r in ex.map(_history_worker, work) if r is not None]
    prog.done()

    if STATS.auth_failed:
        print(f"{COLOR_RED_FG}[Auth Error] Upstox rejected the token (401). Refresh UPSTOX_ACCESS_TOKEN.{COLOR_RESET}")
        return
    if not stage1:
        print(f"{COLOR_YELLOW}No instruments passed the universe filter.{COLOR_RESET}")
        return
    liq = "highly liquid Option Strikes" if mode == "INDEX_OPTIONS" else "qualified assets"
    print(f"✅ Target Universe ready ({len(stage1)} {liq}). Fetching live candles "
          f"[{len(stage1)} calls, {_eta(len(stage1))}; {LIMITER.used(1800)}/{RATE_CAPS[-1][1]} used in last 30 min]...\n")

    prog = Progress("signals", len(stage1))
    work = [(item, hist, days, prog) for item, hist in stage1]
    with concurrent.futures.ThreadPoolExecutor(max_workers=WORKERS) as ex:
        results = [r for r in ex.map(process_stock, work) if r is not None]
    prog.done()

    if not results:
        print(f"{COLOR_YELLOW}No usable data returned.{COLOR_RESET}")
        return

    latest_session = max(r['LastSession'] for r in results)         
    dashboard_data = [r for r in results if r['LastSession'] == latest_session]

    # THE 1X ANCHOR RULE (Mandatory alignment on lowest timeframe)
    bulls = [r for r in dashboard_data if r['PerfectBullBlocks'] >= min_blocks 
             and r.get('BB_RSI_1X') == 'Buy' 
             and r.get('BB_MACD_1X') == 'Buy' 
             and r.get('ADX_1X') == 'Buy']
             
    bears = [r for r in dashboard_data if r['PerfectBearBlocks'] >= min_blocks 
             and r.get('BB_RSI_1X') == 'Sell' 
             and r.get('BB_MACD_1X') == 'Sell' 
             and r.get('ADX_1X') == 'Sell']

    # TWO-TIER PRIME SORTING: 1. Perfect Blocks (Descending) -> 2. Freshness % (Ascending, >0.00 first)
    def order_bull(r):
        d = r['Bull_BB_KC_Delta']
        tier2 = d if d > 0 else float('inf') # Force non-breakouts (inside squeeze) to bottom
        return (-r['PerfectBullBlocks'], tier2)

    def order_bear(r):
        d = r['Bear_BB_KC_Delta']
        tier2 = d if d > 0 else float('inf')
        return (-r['PerfectBearBlocks'], tier2)

    bulls.sort(key=order_bull)
    bears.sort(key=order_bear)
    
    bulls, bears = bulls[:TOP_N_BUYERS], bears[:TOP_N_SELLERS]

    print(f"{COLOR_BOLD}=== STRICT INSTITUTIONAL VOLATILITY DASHBOARD [{mode}] ==={COLOR_RESET}")
    print(f"Session: {latest_session} | Refreshed: {now_ist():%H:%M:%S} IST | Scanned: {len(dashboard_data)}\n")

    sym_title = "Options Strike" if mode == "INDEX_OPTIONS" else "Script"
    prem = "PREMIUM " if mode == "INDEX_OPTIONS" else ""

    def print_basket(title, icon, data_list, dist_key):
        if not data_list: return
        print(f"\n{COLOR_BOLD}{icon} {title}{COLOR_RESET}")
        header_str = f" {COLOR_CYAN}{sym_title:<22} {'LTP':<8} |"
        for mult in HA_ATR_MULTIPLIERS:
            gtag = f"{mult}X"
            header_str += f"  {'BB-RSI ' + gtag:^11} {'BB-MACD ' + gtag:^12} {'BB-DI ' + gtag:^9} |"
        header_str += f" {'BB-KC Δ%':>8}"
        print(header_str + COLOR_RESET)
        print("-" * len(ANSI_RE.sub("", header_str)))
        for row in data_list:
            row_str = f" {row['Symbol']:<22} {row['LTP']:<8.2f} |"
            for mult in HA_ATR_MULTIPLIERS:
                gtag = f"{mult}X"
                row_str += (f"  {format_cell(row.get(f'BB_RSI_{gtag}'), 11)}"
                            f" {format_cell(row.get(f'BB_MACD_{gtag}'), 12)}"
                            f" {format_cell(row.get(f'ADX_{gtag}'), 9)} |")
            val = row[dist_key]
            val_str = f"{val:>7.2f}%" if val > 0 else f"{val:>7.2f}%"
            row_str += f" {val_str}"
            print(row_str)

    print_basket(f"TOP {prem}BUYERS (Freshest BB/KC Breakouts Sorted First)", "🔥", bulls, 'Bull_BB_KC_Delta')
    print_basket(f"TOP {prem}SELLERS (Freshest BB/KC Breakdowns Sorted First)", "🩸", bears, 'Bear_BB_KC_Delta')
    
    dirty_vetoed = sum(1 for r in dashboard_data if r.get('DirtyKilled1X'))
    if not bulls and not bears:
        print(f"{COLOR_YELLOW}No instrument has a perfectly aligned block (with active 1X Anchor) right now.{COLOR_RESET}")
        if dirty_vetoed:
            print(f"{COLOR_YELLOW}   ↳ {dirty_vetoed}/{len(dashboard_data)} instrument(s) had a perfect 1X alignment "
                  f"that was vetoed as Dirty/Exhausted (internal reverse >= {DIRTY_MOVE_ATR_FRACTION:.2f}x Base ATR). "
                  f"Try a looser --dirty-fraction if this feels too strict.{COLOR_RESET}")
    elif dirty_vetoed:
        print(f"{COLOR_YELLOW}ℹ️  {dirty_vetoed} additional instrument(s) had a perfect 1X alignment but were "
              f"vetoed as Dirty/Exhausted.{COLOR_RESET}")
    if STATS.failed:
        print(f"\n{COLOR_YELLOW}⚠️ {STATS.failed} request(s) failed after retries; results may be incomplete.{COLOR_RESET}")

    total_calls = sum(l.total_calls for l in LIMITERS.values())
    print(f"\n⏱️ Scan completed in {(time.time() - t_start):.2f} seconds "
          f"({total_calls} API calls this run).\n")

def parse_args():
    p = argparse.ArgumentParser(description="Strict institutional volatility screener (Upstox)")
    p.add_argument("--mode", choices=["STOCK_FNO", "CASH_EQUITY", "INDEX_OPTIONS"], default=TRADING_MODE)
    p.add_argument("--days", type=int, default=BACKTRACE_DAYS, help="trading sessions of history (min 2)")
    p.add_argument("--min-blocks", type=int, default=MIN_PERFECT_BLOCKS)
    p.add_argument("--dirty-fraction", type=float, default=DIRTY_MOVE_ATR_FRACTION,
                    help="Fraction of Base ATR an internal reverse move must reach, mid-block, "
                         "to mark that block Dirty/Exhausted and kill its signal (default: "
                         f"{DIRTY_MOVE_ATR_FRACTION}). Note this is measured against the flat "
                         "Base ATR, so it bites hardest on the 1X anchor block (whose entire "
                         "range IS 1x Base ATR) and barely at all on 5X blocks. Raise it "
                         "(e.g. 0.6-0.8) if the 1X anchor is vetoing almost everything.")
    return p.parse_args()

if __name__ == "__main__":
    if not os.environ.get("UPSTOX_ACCESS_TOKEN"):
        print(f"{COLOR_RED_FG}[!] Missing UPSTOX_ACCESS_TOKEN environment variable.{COLOR_RESET}")
        sys.exit(1)
    args = parse_args()
    DIRTY_MOVE_ATR_FRACTION = args.dirty_fraction
    run_screener(args.mode, args.days, args.min_blocks)
