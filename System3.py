#!/usr/bin/env python3
"""
Strict Institutional Volatility Tracker (Upstox) - System 3.0 (100% Parameterized Edition)
+ Zero Hardcoded Logic: Every indicator period, multiplier, stop ratchet, and time gate lives in Section 0.
+ Modes: STOCK_FNO, CASH_EQUITY, INDEX_OPTIONS
+ Gap-Stitched Continuous Baseline: Eliminates overnight gap distortion while keeping 09:15 warmup instant.
+ 3-Lock Master Trigger with Configurable Quorum Gating & Rolling Shock Window.
+ Smart Trailing & Profit Ratchet: Disables soft stop at +1.5R and locks breakeven floor at +1.8R.
+ Auction Filter: Blocks new anchors at/after LAST_NEW_ANCHOR_MIN to eliminate closing-auction noise.
"""
import os
import sys
import re
import argparse
import traceback
import urllib.parse
import gzip
import json
import time
import threading
from bisect import bisect_left
from datetime import datetime, timedelta, timezone, time as dtime
import concurrent.futures

import smtplib
from email.message import EmailMessage

import requests
from requests.adapters import HTTPAdapter
import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings("ignore")

# ==============================================================================
# 0. ENGINE CONSTANTS & CONFIGURATION (SINGLE SOURCE OF TRUTH)
# ==============================================================================

# --- 0.1 EXECUTION & UNIVERSE FILTERING ---
TRADING_MODE = "STOCK_FNO"              # Choices: "STOCK_FNO", "CASH_EQUITY", "INDEX_OPTIONS"
WORKERS = 16                            # Concurrent thread pool workers
INCLUDE_NON_EQ_SERIES = False           # Include non-EQ segments in cash/FnO scan
MIN_PRICE = 100.0                       # Equity prefilter minimum price
MAX_PRICE = 5000.0                      # Equity prefilter maximum price
MIN_DAILY_VOLUME = 100000               # Equity prefilter minimum daily volume
BACKTRACE_DAYS = 30                     # Base lookback days for daily prefilter
PREFILTER_LOOKBACK_MULT = 2             # Multiplier applied to BACKTRACE_DAYS for calendar fetch
PREFILTER_LOOKBACK_PAD_DAYS = 6         # Extra padding days for prefilter calendar fetch
MIN1_HISTORY_DAYS = 19                  # Calendar days of 1-minute history fetched per symbol
MAX_PINNED_MATCHES = 8                  # Max symbols pinned via --track substring match

# --- 0.2 SESSION & TIME BOUNDARIES ---
IST_OFFSET_HOURS = 5
IST_OFFSET_MINUTES = 30
IST = timezone(timedelta(hours=IST_OFFSET_HOURS, minutes=IST_OFFSET_MINUTES))
SESSION_OPEN_MIN = 9 * 60 + 15          # 09:15 IST session start (in minutes from midnight)
SESSION_CLOSE_MIN = 15 * 60 + 30        # 15:30 IST session close (in minutes from midnight)
LAST_NEW_ANCHOR_MIN = 15 * 60 + 15      # 15:15 IST cutoff: blocks new anchors during closing auction
DEFAULT_SNAPSHOT_TIME = "15:30"         # Default time string when --date is used without --time
CHECKPOINT_INTERVAL_MIN = 5             # Intraday evaluation checkpoint step (in minutes)
BACKTRACE_CHECKPOINTS = True            # Evaluate across all intraday checkpoints
DAY_PCT_BASIS = "PREV_CLOSE"            # Choices: "PREV_CLOSE", "OPEN"
MAX_SESSION_LOOKBACK_DAYS = 14          # Max days to step back when resolving holidays/weekends
SESSION_BUCKET_MULT = 10000             # Multiplier for isolating session buckets in ATR aggregation

# --- 0.3 LOCK 1: PRICE VOLATILITY BANDS (BOLLINGER vs. KELTNER) ---
PRICE_BB_PERIOD = 20                    # Price Bollinger Band rolling window (bars)
PRICE_BB_STD_MULT = 2.0                 # Price Bollinger Band standard deviation multiplier
PRICE_KC_PERIOD = 20                    # Price Keltner Channel ATR rolling window (bars)
PRICE_KC_ATR_MULT = 1.5                 # Price Keltner Channel ATR multiplier
ROLLING_MIN_PERIODS = 1                 # Minimum bars required for rolling stats
ISOLATE_SESSION_BANDS = False           # False = Gap-Stitched Continuous Baseline (catches 09:15 movers); True = hard reset at 09:15

# --- 0.4 KINETIC INDICATOR PERIODS ---
RSI_PERIOD = 14                         # 1-minute RSI Wilder smoothing period
MACD_FAST_PERIOD = 12                   # 1-minute MACD fast EMA period (alpha = 2 / (N + 1))
MACD_SLOW_PERIOD = 26                   # 1-minute MACD slow EMA period (alpha = 2 / (N + 1))
MACD_SIGNAL_PERIOD = 9                  # 1-minute MACD signal EMA period (alpha = 2 / (N + 1))
ADX_PERIOD = 14                         # 1-minute DMI / ADX Wilder smoothing period
IND_BB_PERIOD = 20                      # Rolling window (bars) for indicator Bollinger Bands
VEL_ATR_PERIOD = 14                     # Wilder smoothing period for 1-minute indicator Velocity ATR
EPSILON = 1e-8                          # Zero-division guard for indicator ratios

# --- 0.5 LOCK 2 & LOCK 3: STRUCTURE & KINETIC SHOCK TRIPWIRES ---
IND_BB_PIERCE_STD = 1.0                 # Lock 2: Indicator StdDev multiplier required for structure breakout
STRUCT_QUORUM = 3                       # Lock 2: Min indicators (out of 4: MACD, OBV, DI, ADX) required
KINETIC_SHOCK_ATR_MULT = 1.5            # Lock 3: Velocity must exceed this multiple of its own Velocity ATR
SHOCK_QUORUM = 3                        # Lock 3: Min indicators (out of 4: MACD, OBV, DI, ADX) required
SHOCK_WINDOW_BARS = 3                   # Lock 3: Rolling 1-minute bars to align sequential indicator shocks
SHOCK_FLOW_MIN_THRESHOLD = 0.0          # Lock 3: Current bar directional velocity floor (vh or vo > threshold)
GRID_MULTIPLIERS = [1, 2, 3, 5]         # Multipliers displayed in the ATR & BB dashboard grids

# --- 0.6 BASE ATR, TRAILING STOPS, PROFIT RATCHET & ADVERSE DRIFT ---
ATR_BASIS_PERIOD = 14                   # Number of historical timeframe bars averaged for Base ATR
ATR_BASIS_TF = "15min"                  # Timeframe string for Base ATR calculation
ATR_MIN_BARS = 5                        # Minimum completed timeframe bars required to validate Base ATR
MIN_ATR_PCT = 0.005                     # Dynamic minimum Base ATR floor as a fraction of price (0.005 = 0.5%)
ABSOLUTE_MIN_ATR = 0.01                 # Absolute numerical floor for Base ATR

HARD_STOP_ATR_MULT = 2.5                # Unconditional hard trailing stop distance (in Base ATRs from peak)
SOFT_STOP_ATR_MULT = 1.0                # Soft kinetic trailing stop distance (in Base ATRs from peak)
SOFT_STOP_OPPOSING_SCORE = 4            # Opposing indicator count (0-4) required to fire Soft Stop
OPPOSING_VEL_ATR_MULT = 1.0             # Opposing velocity ATR multiple required to add +1 to opposing score
OPPOSING_BB_STD_MULT = 1.0              # Opposing indicator BB StdDev multiple required to add +1 to score
WARZONE_OPPOSING_SCORE = 2              # Opposing score threshold that blocks anchor creation as Warzone Chop

SOFT_STOP_DISABLE_R = 1.5               # R-Multiple peak above which Soft Stop disables so 2.5 ATR Hard Stop rides the trend
PROFIT_LOCK_TRIGGER_R = 1.8             # R-Multiple peak required to activate the Breakeven+ Profit Ratchet
PROFIT_LOCK_FLOOR_R = 0.3               # Minimum R-Multiple floor locked in once PROFIT_LOCK_TRIGGER_R is hit

ADVERSE_DRIFT_R_FLOOR = -0.5            # Explicit R-Multiple floor for live purge at snapshot time
ADVERSE_DRIFT_STOP_RATIO = 1.0          # Ratio of HARD_STOP_ATR_MULT used when ADVERSE_DRIFT_R_FLOOR is None
BANKED_R_PROTECTION_FLOOR = 0.5         # Cumulative R threshold that protects re-entries from adverse drift purge

# --- 0.7 INDEX OPTIONS CONFIGURATION ---
EXPIRY_SELECTION = "CURRENT"            # Choices: "CURRENT", "NEXT"
STRIKES_ABOVE_ATM = 5                   # Number of strikes above ATM to scan
STRIKES_BELOW_ATM = 5                   # Number of strikes below ATM to scan
OPT_MIN_PRICE = 10.0                    # Minimum option LTP at snapshot
OPT_MIN_VOLUME = 5000                   # Minimum cumulative option volume at snapshot
LIVE_QUOTE_FALLBACK_MIN = 3             # Minutes from current clock within which live quote spot fallback is allowed
SPOT_1M_LOOKBACK_DAYS = 5               # Calendar days lookback for 1m index spot resolution
SPOT_DAILY_LOOKBACK_DAYS = 10           # Calendar days lookback for daily index spot resolution
INDEX_CONFIG = {
    "NIFTY":     {"spot_key": "NSE_INDEX|Nifty 50",          "aliases": {"NIFTY", "NIFTY 50"}},
    "BANKNIFTY": {"spot_key": "NSE_INDEX|Nifty Bank",        "aliases": {"BANKNIFTY", "NIFTY BANK"}},
    "FINNIFTY":  {"spot_key": "NSE_INDEX|Nifty Fin Service", "aliases": {"FINNIFTY", "NIFTY FIN SERVICE"}},
    "SENSEX":    {"spot_key": "BSE_INDEX|SENSEX",            "aliases": {"SENSEX", "BSE SENSEX", "BSESN"}},
}
ALIAS_TO_INDEX = {a: idx for idx, cfg in INDEX_CONFIG.items() for a in cfg["aliases"]}
PROBE_KEY = INDEX_CONFIG["NIFTY"]["spot_key"]

# --- 0.8 DASHBOARD DISPLAY & FORMATTING ---
TOP_N_BUYERS = 50                       # Max rows in Top Bull Setups
TOP_N_SELLERS = 50                      # Max rows in Top Bear Setups
TOP_N_GRAVEYARD = 100                   # Max rows in The Graveyard (0 = unlimited)
MIN_SYMBOL_WIDTH = 12                   # Minimum width of the Script column
MAX_SYMBOL_WIDTH = 30                   # Maximum width of the Script column
BANKED_R_DISPLAY_EPSILON = 0.01         # Minimum absolute Banked R to display the (* / BankedR) badge

COLOR_RESET = '\033[0m'
COLOR_BOLD = '\033[1m'
COLOR_DIM = '\033[90m'
COLOR_CYAN = '\033[96;1m'
COLOR_GREEN_FG = '\033[92;1m'
COLOR_RED_FG = '\033[91;1m'
COLOR_YELLOW = '\033[93;1m'
ANSI_RE = re.compile(r'\x1b\[[0-9;]*m')

ACTIVE_BULL_STATES = ("[ BB > KC ]", "[ BB < KC ]")
ACTIVE_BEAR_STATES = ("[ BB > KC ]", "[ BB < KC ]")

# --- 0.9 UPSTOX API, RATE LIMITER & SMTP INFRASTRUCTURE ---
API_HOST = "https://api.upstox.com"
MASTER_BASE_URL = "https://assets.upstox.com/market-quote/instruments/exchange"
RATE_CAPS = ((1.0, 22), (60.0, 220), (1800.0, 900))
LIMITER_MAX_SLEEP_SEC = 1.0
LIMITER_JITTER_SEC = 0.005
HTTP_POOL_CONNECTIONS = 2
HTTP_POOL_MAXSIZE = 2
API_MAX_RETRIES = 4
API_TIMEOUT_SEC = 20
API_RATE_LIMIT_BACKOFF_SEC = 2.0
API_DEFAULT_BACKOFF_SEC = 0.5
URL_LOG_CLIP_LEN = 90
MAX_ERROR_SAMPLES = 5
PROGRESS_STEPS = 20
QUOTE_BATCH_SIZE = 200
CHUNK_SPAN_1M_DAYS = 7
CHUNK_SPAN_DAILY_DAYS = 365
MASTER_DOWNLOAD_RETRIES = 3
MASTER_DOWNLOAD_TIMEOUT_SEC = 90
MASTER_RETRY_SLEEP_SEC = 1.0

DEFAULT_SMTP_SERVER = "smtp.gmail.com"
DEFAULT_SMTP_PORT = "465"
SMTP_SSL_PORT = 465
SMTP_TIMEOUT_SEC = 20

# ==============================================================================
# HELPERS & RATE LIMITERS
# ==============================================================================
def now_ist():
    return datetime.now(IST).replace(tzinfo=None)

def _close_dt(day):
    return datetime.combine(day, dtime(SESSION_CLOSE_MIN // 60, SESSION_CLOSE_MIN % 60))

def _open_dt(day):
    return datetime.combine(day, dtime(SESSION_OPEN_MIN // 60, SESSION_OPEN_MIN % 60))

def _norm_sym(s):
    return re.sub(r'[\s_\-]+', '', str(s).upper())

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
            time.sleep(min(wait, LIMITER_MAX_SLEEP_SEC) + LIMITER_JITTER_SEC)

class FetchStats:
    def __init__(self):
        self.lock = threading.Lock()
        self.failed = 0
        self.samples = []
        self.auth_failed = False

    def fail(self, msg):
        with self.lock:
            self.failed += 1
            if len(self.samples) < MAX_ERROR_SAMPLES:
                self.samples.append(msg)

    def mark_auth_failed(self):
        with self.lock:
            self.auth_failed = True

class ErrorLog:
    def __init__(self):
        self.lock = threading.Lock()
        self.items = []

    def add(self, where, exc):
        with self.lock:
            self.items.append((where, f"{type(exc).__name__}: {exc}", traceback.format_exc()))

STATS = FetchStats()
ERRORS = ErrorLog()
LIMITERS = {
    "quotes": BudgetLimiter(RATE_CAPS),
    "history": BudgetLimiter(RATE_CAPS),
    "intraday": BudgetLimiter(RATE_CAPS),
}
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
        s.mount("https://", HTTPAdapter(pool_connections=HTTP_POOL_CONNECTIONS, pool_maxsize=HTTP_POOL_MAXSIZE))
        _TLS.s = s
    return s

class Progress:
    def __init__(self, label, total):
        self.label = label
        self.total = total
        self.n = 0
        self.step = max(1, total // PROGRESS_STEPS)
        self.lock = threading.Lock()

    def tick(self):
        with self.lock:
            self.n += 1
            if self.n == self.total or self.n % self.step == 0:
                print(f"\r   {self.label}: {self.n}/{self.total}", end="", file=sys.stderr, flush=True)

    def done(self):
        print("", file=sys.stderr)

# ==============================================================================
# 1. UPSTOX API
# ==============================================================================
def _get(url, params=None, retries=None):
    max_retries = API_MAX_RETRIES if retries is None else retries
    token = os.environ.get("UPSTOX_ACCESS_TOKEN")
    if not token or STATS.auth_failed:
        return 401, None
    headers = {'Accept': 'application/json', 'Authorization': f'Bearer {token}'}
    limiter = _limiter_for(url)
    short = url.split("/v2/")[-1][:URL_LOG_CLIP_LEN]
    last = "no attempt made"
    for attempt in range(max_retries):
        limiter.acquire()
        try:
            r = _session().get(url, headers=headers, params=params, timeout=API_TIMEOUT_SEC)
            code = r.status_code
            if code == 200:
                return 200, r.json()
            if code == 401:
                STATS.mark_auth_failed()
                return 401, None
            last = f"HTTP {code}"
            if code in (400, 404):
                STATS.fail(f"{last} for {short}")
                return code, None
            backoff = API_RATE_LIMIT_BACKOFF_SEC if code == 429 else API_DEFAULT_BACKOFF_SEC
            time.sleep(backoff * (attempt + 1))
            continue
        except Exception as e:
            last = f"{type(e).__name__}: {e}"
        time.sleep(API_DEFAULT_BACKOFF_SEC * (attempt + 1))
    STATS.fail(f"{last} for {short}")
    return 0, None

def _to_frame(candles):
    if not candles:
        return None
    df = pd.DataFrame(candles).iloc[:, :6]
    df.columns = ['Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume']
    ts = df['Timestamp'].astype(str)
    try:
        df['Datetime'] = pd.to_datetime(ts.str.slice(0, 19), format="%Y-%m-%dT%H:%M:%S")
    except ValueError:
        df['Datetime'] = pd.to_datetime(ts, utc=True).dt.tz_convert(IST).dt.tz_localize(None)
    for c in ('Open', 'High', 'Low', 'Close', 'Volume'):
        df[c] = df[c].astype(float)
    return df.drop(columns='Timestamp').drop_duplicates(subset='Datetime').sort_values('Datetime').reset_index(drop=True)

def _candles(url):
    status, js = _get(url)
    if status != 200 or not js:
        return status, None
    try:
        return 200, _to_frame((js.get('data') or {}).get('candles') or [])
    except Exception as e:
        STATS.fail(f"candle parse error ({type(e).__name__}: {e}) for {url.split('/v2/')[-1][:URL_LOG_CLIP_LEN]}")
        return 200, None

def _url_range_daily(key, start, end):
    return f"{API_HOST}/v2/historical-candle/{urllib.parse.quote(key)}/day/{end:%Y-%m-%d}/{start:%Y-%m-%d}"

def _url_range_1m(key, start, end):
    return f"{API_HOST}/v2/historical-candle/{urllib.parse.quote(key)}/1minute/{end:%Y-%m-%d}/{start:%Y-%m-%d}"

def _url_intraday(key):
    return f"{API_HOST}/v2/historical-candle/intraday/{urllib.parse.quote(key)}/1minute"

def _date_chunks(start, end, span=None):
    chunk_span = CHUNK_SPAN_1M_DAYS if span is None else span
    cur = end
    while cur >= start:
        c_start = max(start, cur - timedelta(days=chunk_span - 1))
        yield c_start, cur
        cur = c_start - timedelta(days=1)

def generate_checkpoints(cutoff_dt, interval_min=None):
    interval = max(1, int(interval_min or CHECKPOINT_INTERVAL_MIN))
    session_open = _open_dt(cutoff_dt.date())
    if cutoff_dt <= session_open:
        return [cutoff_dt]
    checkpoints, t = [], session_open + timedelta(minutes=interval)
    while t <= cutoff_dt:
        checkpoints.append(t)
        t += timedelta(minutes=interval)
    if not checkpoints or checkpoints[-1] < cutoff_dt:
        checkpoints.append(cutoff_dt)
    return checkpoints

def fetch_quotes(items, batch=None):
    batch_sz = QUOTE_BATCH_SIZE if batch is None else batch
    batches = [items[i:i + batch_sz] for i in range(0, len(items), batch_sz)]

    def one(b):
        status, js = _get(f"{API_HOST}/v2/market-quote/quotes", params={"instrument_key": ",".join(x['key'] for x in b)})
        if status != 200 or not js:
            return {}
        by_ts = {f"{x['key'].split('|')[0]}:{x['symbol']}": x['key'] for x in b}
        data, out = js.get('data') or {}, {}
        for k, v in data.items():
            key = v.get('instrument_token') or by_ts.get(k)
            if not key and len(b) == 1 and len(data) == 1:
                key = b[0]['key']
            if key:
                out[key] = {'ltp': v.get('last_price') or 0.0, 'vol': v.get('volume') or 0}
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

def _session_traded(d, today):
    if d.weekday() >= 5:
        return False
    if d == today:
        df = fetch_today(PROBE_KEY)
        return df is not None and not df.empty
    status, df = _candles(_url_range_1m(PROBE_KEY, d, d + timedelta(days=1)))
    return bool(status == 200 and df is not None and (df['Datetime'].dt.date == d).any())

def resolve_session(cutoff_dt):
    notes = []
    now = now_ist().replace(second=0, microsecond=0)
    if cutoff_dt > now:
        notes.append(f"snapshot {cutoff_dt:%Y-%m-%d %H:%M} is in the future -- using current time {now:%Y-%m-%d %H:%M} instead")
        cutoff_dt = now
    target = cutoff_dt.date()

    if cutoff_dt <= _open_dt(target):
        why = f"{cutoff_dt:%Y-%m-%d %H:%M} is at/before the {_open_dt(target):%H:%M} open"
    elif target.weekday() >= 5:
        why = f"{target} is a weekend"
    elif _session_traded(target, now.date()):
        return min(cutoff_dt, _close_dt(target)), notes
    elif STATS.auth_failed:
        return cutoff_dt, notes
    else:
        why = f"no trading data for {target} (market holiday?)"

    d = target - timedelta(days=1)
    for _ in range(MAX_SESSION_LOOKBACK_DAYS):
        if STATS.auth_failed:
            return cutoff_dt, notes
        if _session_traded(d, now.date()):
            notes.append(f"{why} -> using the last completed session: {d} {SESSION_CLOSE_MIN // 60}:{SESSION_CLOSE_MIN % 60:02d}")
            return _close_dt(d), notes
        d -= timedelta(days=1)
    notes.append(f"{why}, and no trading session was found in the previous {MAX_SESSION_LOOKBACK_DAYS} days -- continuing with requested time")
    return cutoff_dt, notes

def prepare_master(dfs):
    master = (
        pd.concat([d[['Datetime', 'Open', 'High', 'Low', 'Close', 'Volume']] for d in dfs], ignore_index=True)
        .drop_duplicates(subset='Datetime')
        .sort_values('Datetime')
        .reset_index(drop=True)
    )
    dt = master['Datetime']
    day = dt.dt.normalize()
    master['Min'] = dt.dt.hour.values * 60 + dt.dt.minute.values
    master['SessId'] = day.values.astype('datetime64[D]').astype('int64')
    master['Session'] = day.dt.date
    return master

def _unwrap_master(data):
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        if isinstance(data.get("data"), list):
            return data["data"]
        for v in data.values():
            if isinstance(v, list):
                return v
    return []

def _download_master(name, _is_fallback=False):
    url = f"{MASTER_BASE_URL}/{name}.json.gz"
    for attempt in range(1, MASTER_DOWNLOAD_RETRIES + 1):
        try:
            resp = requests.get(url, timeout=MASTER_DOWNLOAD_TIMEOUT_SEC)
            if resp.status_code == 200:
                try:
                    rows = _unwrap_master(json.loads(gzip.decompress(resp.content).decode('utf-8')))
                    if rows:
                        return rows
                except Exception:
                    pass
            elif resp.status_code == 404:
                break
        except Exception:
            pass
        time.sleep(MASTER_RETRY_SLEEP_SEC)
    if not _is_fallback and name.upper() in ("NSE", "BSE"):
        full = _download_master("complete", _is_fallback=True)
        rows = [
            r for r in full
            if str(r.get("exchange", "")).upper() == name.upper()
            or str(r.get("segment", "")).upper().startswith(name.upper())
        ]
        if rows:
            return rows
    return []

def resolve_pins(cands, needles, label="symbol"):
    pinned = set()
    norm = [(_norm_sym(sym), sym, key) for sym, key in cands]
    for raw in needles:
        n = _norm_sym(raw)
        if not n:
            continue
        hits = [c for c in norm if c[0] == n] or [c for c in norm if n in c[0]]
        if not hits:
            continue
        hits.sort(key=lambda c: c[2])
        pinned.update(h[1] for h in hits[:MAX_PINNED_MATCHES])
    return pinned

def _equity_universe(mode, needles=()):
    nse = _download_master("NSE")
    if not nse:
        return []

    def ts_of(i):
        return i.get("tradingsymbol", i.get("trading_symbol"))

    def plain(i):
        return (
            i.get("segment") == "NSE_EQ"
            and ts_of(i)
            and i.get("instrument_key")
            and (INCLUDE_NON_EQ_SERIES or (i.get("instrument_type") or "EQ") == "EQ")
        )

    fno = {i.get("underlying_symbol") for i in nse if i.get("segment") == "NSE_FO" and i.get("underlying_symbol")}
    everything = {i["instrument_key"]: {"symbol": ts_of(i), "key": i["instrument_key"]} for i in nse if plain(i)}
    in_mode = {k: v for k, v in everything.items() if ((v["symbol"] in fno) if mode == "STOCK_FNO" else (v["symbol"] not in fno))}
    pinned = set()
    if needles:
        pinned = resolve_pins([(v["symbol"], (len(v["symbol"]), v["symbol"])) for v in everything.values()], needles, "equity symbol")
    out = {k: dict(v, pinned=v["symbol"] in pinned) for k, v in in_mode.items()}
    for k, v in everything.items():
        if v["symbol"] in pinned and k not in out:
            out[k] = dict(v, pinned=True)
    return list(out.values())

def _resolve_index_spot_price(key, name, cutoff_dt, use_intraday):
    target_dt = cutoff_dt.date()
    if use_intraday:
        df = fetch_today(key)
        if df is not None and not df.empty:
            sub = df[df['Datetime'] < cutoff_dt]
            if not sub.empty:
                return float(sub['Close'].iloc[-1]), "today's 1-min close at snapshot"
        if now_ist() - cutoff_dt <= timedelta(minutes=LIVE_QUOTE_FALLBACK_MIN):
            q = fetch_quotes([{"key": key, "symbol": name}], batch=1)
            ltp = q.get(key, {}).get('ltp', 0.0)
            if ltp and ltp > 0:
                return float(ltp), "live quote"
    status, df = _candles(_url_range_1m(key, target_dt - timedelta(days=SPOT_1M_LOOKBACK_DAYS), target_dt + timedelta(days=1)))
    if status == 200 and df is not None and not df.empty:
        sub = df[df['Datetime'] < cutoff_dt]
        if not sub.empty:
            return float(sub['Close'].iloc[-1]), "1-min close at snapshot"
    status, df = _candles(_url_range_daily(key, target_dt - timedelta(days=SPOT_DAILY_LOOKBACK_DAYS), target_dt + timedelta(days=1)))
    if status == 200 and df is not None and not df.empty:
        limit = target_dt if cutoff_dt >= _close_dt(target_dt) else target_dt - timedelta(days=1)
        sub = df[df['Datetime'].dt.date <= limit]
        if not sub.empty:
            return float(sub['Close'].iloc[-1]), "daily close (fallback)"
    return 0.0, "no candles returned"

def _field(row, *names):
    for n in names:
        v = row.get(n)
        if v is None or v == "" or (isinstance(v, float) and v != v):
            continue
        return v
    return None

def _expiry_to_date(exp):
    try:
        if isinstance(exp, (int, float, np.integer, np.floating)) or (isinstance(exp, str) and exp.strip().isdigit()):
            v = float(exp)
            if v > 1e11:
                v /= 1000.0
            return datetime.fromtimestamp(v, IST).date()
        s = str(exp).strip().split('T')[0].split(' ')[0]
        for fmt in ("%Y-%m-%d", "%d-%b-%Y", "%d-%m-%Y", "%d/%m/%Y"):
            try:
                return datetime.strptime(s, fmt).date()
            except ValueError:
                pass
    except Exception:
        pass
    return None

def _index_options_universe(cutoff_dt, use_intraday, needles=()):
    target_dt = cutoff_dt.date()
    spot = {}
    for idx, cfg in INDEX_CONFIG.items():
        try:
            price, _ = _resolve_index_spot_price(cfg["spot_key"], idx, cutoff_dt, use_intraday)
        except Exception:
            price = 0.0
        if price > 0:
            spot[idx] = price
    if not spot and not needles:
        return []

    rows = _download_master("NSE") + _download_master("BSE")
    if not rows:
        return []

    by_idx = {idx: [] for idx in INDEX_CONFIG}
    for row in rows:
        segment = str(_field(row, "segment") or "").upper()
        if segment and segment not in ("NSE_FO", "BSE_FO"):
            continue
        ts = str(_field(row, "trading_symbol", "tradingsymbol") or "").upper()
        itype = str(_field(row, "option_type", "instrument_type") or "").upper()
        if itype not in ("CE", "PE"):
            m = re.search(r'\b(CE|PE)\b', ts) or re.search(r'\d(CE|PE)$', ts)
            if not m:
                continue
            itype = m.group(1)
        cands = [str(_field(row, "underlying_symbol") or ""), str(_field(row, "name") or ""), ts.split()[0] if ts else ""]
        idx = next((ALIAS_TO_INDEX[c.upper().strip()] for c in cands if c.upper().strip() in ALIAS_TO_INDEX), None)
        if idx is None:
            continue
        exp_date = _expiry_to_date(_field(row, "expiry"))
        if exp_date is None:
            continue
        strike_raw = _field(row, "strike_price", "strike")
        try:
            strike = float(strike_raw) if strike_raw is not None else 0.0
        except Exception:
            strike = 0.0
        if strike <= 0:
            m = re.search(r'(\d+(?:\.\d+)?)\s+(?:CE|PE)\b', ts) or re.search(r'(\d+)(?:CE|PE)$', ts)
            strike = float(m.group(1)) if m else 0.0
        if strike <= 0 or not _field(row, "instrument_key"):
            continue
        by_idx[idx].append((exp_date, strike, itype, row))

    universe, seen = [], set()
    if needles:
        flat = []
        for idx, contracts in by_idx.items():
            for exp_date, strike, itype, row in contracts:
                if exp_date < target_dt:
                    continue
                key = _field(row, "instrument_key")
                sym = str(_field(row, "trading_symbol", "tradingsymbol") or key)
                flat.append((sym, (exp_date, strike, itype, sym), key))
        sym_to_key = {sym: key for sym, _, key in flat}
        for sym in sorted(resolve_pins([(s, k) for s, k, _ in flat], needles, "option contract")):
            key = sym_to_key[sym]
            if key not in seen:
                seen.add(key)
                universe.append({"key": key, "symbol": sym, "pinned": True})

    for idx, contracts in by_idx.items():
        if idx not in spot or not contracts:
            continue
        expiries = sorted({c[0] for c in contracts if c[0] >= target_dt})
        if not expiries:
            continue
        expiry_idx = 1 if str(EXPIRY_SELECTION).upper() == "NEXT" else 0
        target_exp = expiries[min(expiry_idx, len(expiries) - 1)]
        chain = [c for c in contracts if c[0] == target_exp]
        strikes = np.array(sorted({c[1] for c in chain}))
        atm_i = int(np.abs(strikes - spot[idx]).argmin())
        lo = max(0, atm_i - STRIKES_BELOW_ATM)
        hi = min(len(strikes), atm_i + STRIKES_ABOVE_ATM + 1)
        window = set(strikes[lo:hi].tolist())
        for exp_date, strike, itype, row in chain:
            key = _field(row, "instrument_key")
            if strike in window and key not in seen:
                seen.add(key)
                universe.append({"key": key, "symbol": str(_field(row, "trading_symbol", "tradingsymbol") or key), "pinned": False})
    return universe

def get_dynamic_universe(mode, cutoff_dt, use_intraday, needles=()):
    if mode in ("STOCK_FNO", "CASH_EQUITY"):
        return _equity_universe(mode, needles)
    if mode == "INDEX_OPTIONS":
        return _index_options_universe(cutoff_dt, use_intraday, needles)
    return []

# ==============================================================================
# 2. CONTINUOUS KINETIC TRIPWIRE ENGINE
# ==============================================================================
def compute_base_atr(m):
    if m is None or len(m) == 0:
        return None, 0
    hi, lo, cl = m['High'].values, m['Low'].values, m['Close'].values
    sid = m['SessId'].values
    mins = np.clip(m['Min'].values, SESSION_OPEN_MIN, SESSION_CLOSE_MIN - 1)
    tf_minutes = max(1, int(str(ATR_BASIS_TF).replace("min", "")))
    bucket = sid * SESSION_BUCKET_MULT + (mins - SESSION_OPEN_MIN) // tf_minutes
    starts = np.concatenate(([0], np.flatnonzero(np.diff(bucket)) + 1))
    ends = np.concatenate((starts[1:], [len(bucket)])) - 1
    b_high, b_low, b_close = np.maximum.reduceat(hi, starts), np.minimum.reduceat(lo, starts), cl[ends]
    prev_close = np.concatenate(([np.nan], b_close[:-1]))
    tr = np.fmax(np.fmax(b_high - b_low, np.abs(b_high - prev_close)), np.abs(b_low - prev_close))
    atr = float(tr[-ATR_BASIS_PERIOD:].mean())
    return atr, len(b_close)

def _ewm(x, alpha):
    xs = x.tolist()
    out = [0.0] * len(xs)
    prev = out[0] = xs[0]
    keep = 1.0 - alpha
    for i in range(1, len(xs)):
        out[i] = prev = keep * prev + alpha * xs[i]
    return np.asarray(out)

def _evaluate_kinetic_arrays(open_, close, high, low, vol, sess_starts, sid):
    """
    Computes 1m kinetics using global period constants on gap-neutralized continuous
    series so morning gaps do not distort bands AND 09:15 breakouts fire without warmup lag.
    """
    n = len(close)
    delta = np.diff(close, prepend=close[0])
    delta = np.where(sess_starts, close - open_, delta)

    alpha_rsi = 1.0 / float(RSI_PERIOD)
    gain = _ewm(np.where(delta > 0, delta, 0.0), alpha_rsi)
    loss = _ewm(np.where(delta < 0, -delta, 0.0), alpha_rsi)
    rsi = 100.0 - (100.0 / (1.0 + (gain / (loss + EPSILON))))

    alpha_fast = 2.0 / (float(MACD_FAST_PERIOD) + 1.0)
    alpha_slow = 2.0 / (float(MACD_SLOW_PERIOD) + 1.0)
    alpha_sig = 2.0 / (float(MACD_SIGNAL_PERIOD) + 1.0)

    intraday_close_chain = np.cumsum(delta) + open_[0]
    macd = _ewm(intraday_close_chain, alpha_fast) - _ewm(intraday_close_chain, alpha_slow)
    hist = macd - _ewm(macd, alpha_sig)

    up, down = np.zeros(n), np.zeros(n)
    up[1:] = high[1:] - high[:-1]
    down[1:] = low[:-1] - low[1:]
    up = np.where(sess_starts, high - open_, up)
    down = np.where(sess_starts, open_ - low, down)
    plus_dm = np.where((up > down) & (up > 0), up, 0.0)
    minus_dm = np.where((down > up) & (down > 0), down, 0.0)

    tr = (high - low).copy()
    tr[1:] = np.maximum(tr[1:], np.maximum(np.abs(high[1:] - close[:-1]), np.abs(low[1:] - close[:-1])))
    tr = np.where(sess_starts, high - low, tr)

    alpha_adx = 1.0 / float(ADX_PERIOD)
    atr = _ewm(tr, alpha_adx)
    plus_di = 100.0 * (_ewm(plus_dm, alpha_adx) / (atr + EPSILON))
    minus_di = 100.0 * (_ewm(minus_dm, alpha_adx) / (atr + EPSILON))
    dx = 100.0 * np.abs(plus_di - minus_di) / (plus_di + minus_di + EPSILON)
    adx = _ewm(dx, alpha_adx)

    direction = np.where(delta > 0, 1, np.where(delta < 0, -1, 0))
    obv = np.cumsum(direction * vol)

    if ISOLATE_SESSION_BANDS:
        df = pd.DataFrame({
            'rsi': rsi, 'hist': hist, 'plus_di': plus_di,
            'minus_di': minus_di, 'adx': adx, 'obv': obv, 'sid': sid
        })
        grouped = df.groupby('sid')
        def roll_mean(x): return x.rolling(IND_BB_PERIOD, min_periods=ROLLING_MIN_PERIODS).mean()
        def roll_std(x): return x.rolling(IND_BB_PERIOD, min_periods=ROLLING_MIN_PERIODS).std(ddof=0)
        rsi_mean, rsi_std = grouped['rsi'].transform(roll_mean).values, grouped['rsi'].transform(roll_std).values
        h_mean, h_std = grouped['hist'].transform(roll_mean).values, grouped['hist'].transform(roll_std).values
        p_mean, p_std = grouped['plus_di'].transform(roll_mean).values, grouped['plus_di'].transform(roll_std).values
        m_mean, m_std = grouped['minus_di'].transform(roll_mean).values, grouped['minus_di'].transform(roll_std).values
        a_mean, a_std = grouped['adx'].transform(roll_mean).values, grouped['adx'].transform(roll_std).values
        o_mean, o_std = grouped['obv'].transform(roll_mean).values, grouped['obv'].transform(roll_std).values
    else:
        # Continuous rolling stats across gap-neutralized indicator arrays (instant 09:15 readiness)
        rsi_s, hist_s = pd.Series(rsi), pd.Series(hist)
        pdi_s, mdi_s = pd.Series(plus_di), pd.Series(minus_di)
        adx_s, obv_s = pd.Series(adx), pd.Series(obv)
        rsi_mean, rsi_std = rsi_s.rolling(IND_BB_PERIOD, min_periods=ROLLING_MIN_PERIODS).mean().values, rsi_s.rolling(IND_BB_PERIOD, min_periods=ROLLING_MIN_PERIODS).std(ddof=0).values
        h_mean, h_std = hist_s.rolling(IND_BB_PERIOD, min_periods=ROLLING_MIN_PERIODS).mean().values, hist_s.rolling(IND_BB_PERIOD, min_periods=ROLLING_MIN_PERIODS).std(ddof=0).values
        p_mean, p_std = pdi_s.rolling(IND_BB_PERIOD, min_periods=ROLLING_MIN_PERIODS).mean().values, pdi_s.rolling(IND_BB_PERIOD, min_periods=ROLLING_MIN_PERIODS).std(ddof=0).values
        m_mean, m_std = mdi_s.rolling(IND_BB_PERIOD, min_periods=ROLLING_MIN_PERIODS).mean().values, mdi_s.rolling(IND_BB_PERIOD, min_periods=ROLLING_MIN_PERIODS).std(ddof=0).values
        a_mean, a_std = adx_s.rolling(IND_BB_PERIOD, min_periods=ROLLING_MIN_PERIODS).mean().values, adx_s.rolling(IND_BB_PERIOD, min_periods=ROLLING_MIN_PERIODS).std(ddof=0).values
        o_mean, o_std = obv_s.rolling(IND_BB_PERIOD, min_periods=ROLLING_MIN_PERIODS).mean().values, obv_s.rolling(IND_BB_PERIOD, min_periods=ROLLING_MIN_PERIODS).std(ddof=0).values

    vel_hist = np.diff(hist, prepend=hist[0])
    vel_obv = np.diff(obv, prepend=obv[0])
    vel_pdi = np.diff(plus_di, prepend=plus_di[0])
    vel_mdi = np.diff(minus_di, prepend=minus_di[0])
    vel_adx = np.diff(adx, prepend=adx[0])

    alpha_vel = 1.0 / float(VEL_ATR_PERIOD)
    atr_hist = _ewm(np.abs(vel_hist), alpha_vel)
    atr_obv = _ewm(np.abs(vel_obv), alpha_vel)
    atr_pdi = _ewm(np.abs(vel_pdi), alpha_vel)
    atr_mdi = _ewm(np.abs(vel_mdi), alpha_vel)
    atr_adx = _ewm(np.abs(vel_adx), alpha_vel)

    w = max(1, int(SHOCK_WINDOW_BARS))
    win_vh_max = pd.Series(vel_hist).rolling(w, min_periods=ROLLING_MIN_PERIODS).max().values
    win_vh_min = pd.Series(vel_hist).rolling(w, min_periods=ROLLING_MIN_PERIODS).min().values
    win_vo_max = pd.Series(vel_obv).rolling(w, min_periods=ROLLING_MIN_PERIODS).max().values
    win_vo_min = pd.Series(vel_obv).rolling(w, min_periods=ROLLING_MIN_PERIODS).min().values
    win_vp_max = pd.Series(vel_pdi).rolling(w, min_periods=ROLLING_MIN_PERIODS).max().values
    win_vm_max = pd.Series(vel_mdi).rolling(w, min_periods=ROLLING_MIN_PERIODS).max().values
    win_va_max = pd.Series(vel_adx).rolling(w, min_periods=ROLLING_MIN_PERIODS).max().values

    return {
        'rsi': rsi, 'r_mean': rsi_mean, 'r_std': rsi_std,
        'hist': hist, 'h_mean': h_mean, 'h_std': h_std,
        'obv': obv, 'o_mean': o_mean, 'o_std': o_std,
        'plus_di': plus_di, 'p_mean': p_mean, 'p_std': p_std,
        'minus_di': minus_di, 'm_mean': m_mean, 'm_std': m_std,
        'adx': adx, 'a_mean': a_mean, 'a_std': a_std,
        'vel_hist': vel_hist, 'atr_hist': atr_hist,
        'vel_obv': vel_obv, 'atr_obv': atr_obv,
        'vel_pdi': vel_pdi, 'atr_pdi': atr_pdi,
        'vel_mdi': vel_mdi, 'atr_mdi': atr_mdi,
        'vel_adx': vel_adx, 'atr_adx': atr_adx,
        'win_vh_max': win_vh_max, 'win_vh_min': win_vh_min,
        'win_vo_max': win_vo_max, 'win_vo_min': win_vo_min,
        'win_vp_max': win_vp_max, 'win_vm_max': win_vm_max,
        'win_va_max': win_va_max
    }

def get_kinetics(kin_1m, idx):
    hist, h_mean, h_std = kin_1m['hist'][idx], kin_1m['h_mean'][idx], kin_1m['h_std'][idx]
    obv, o_mean, o_std = kin_1m['obv'][idx], kin_1m['o_mean'][idx], kin_1m['o_std'][idx]
    p_di, p_mean, p_std = kin_1m['plus_di'][idx], kin_1m['p_mean'][idx], kin_1m['p_std'][idx]
    m_di, m_mean, m_std = kin_1m['minus_di'][idx], kin_1m['m_mean'][idx], kin_1m['m_std'][idx]
    adx, a_mean, a_std = kin_1m['adx'][idx], kin_1m['a_mean'][idx], kin_1m['a_std'][idx]

    vh, ah = kin_1m['vel_hist'][idx], kin_1m['atr_hist'][idx]
    vo, ao = kin_1m['vel_obv'][idx], kin_1m['atr_obv'][idx]
    vp, ap = kin_1m['vel_pdi'][idx], kin_1m['atr_pdi'][idx]
    vm, am = kin_1m['vel_mdi'][idx], kin_1m['atr_mdi'][idx]
    va, aa = kin_1m['vel_adx'][idx], kin_1m['atr_adx'][idx]

    w_vh_max, w_vh_min = kin_1m['win_vh_max'][idx], kin_1m['win_vh_min'][idx]
    w_vo_max, w_vo_min = kin_1m['win_vo_max'][idx], kin_1m['win_vo_min'][idx]
    w_vp_max, w_vm_max, w_va_max = kin_1m['win_vp_max'][idx], kin_1m['win_vm_max'][idx], kin_1m['win_va_max'][idx]

    # --- LOCK 2: Structure Breakout (Quorum of 4) ---
    b_struct_cnt = (
        int(h_std > 0 and hist > h_mean + IND_BB_PIERCE_STD * h_std) +
        int(o_std > 0 and obv > o_mean + IND_BB_PIERCE_STD * o_std) +
        int(p_std > 0 and p_di > p_mean + IND_BB_PIERCE_STD * p_std and p_di > m_di) +
        int(a_std > 0 and adx > a_mean + IND_BB_PIERCE_STD * a_std and p_di > m_di)
    )
    br_struct_cnt = (
        int(h_std > 0 and hist < h_mean - IND_BB_PIERCE_STD * h_std) +
        int(o_std > 0 and obv < o_mean - IND_BB_PIERCE_STD * o_std) +
        int(m_std > 0 and m_di > m_mean + IND_BB_PIERCE_STD * m_std and m_di > p_di) +
        int(a_std > 0 and adx > a_mean + IND_BB_PIERCE_STD * a_std and m_di > p_di)
    )

    bull_struct = (b_struct_cnt >= STRUCT_QUORUM) and (p_di > m_di)
    bear_struct = (br_struct_cnt >= STRUCT_QUORUM) and (m_di > p_di)

    # --- LOCK 3: Kinetic Shock (Quorum of 4 within Rolling Window) ---
    b_shock_cnt = (
        int(w_vh_max > KINETIC_SHOCK_ATR_MULT * ah) +
        int(w_vo_max > KINETIC_SHOCK_ATR_MULT * ao) +
        int(w_vp_max > KINETIC_SHOCK_ATR_MULT * ap) +
        int(w_va_max > KINETIC_SHOCK_ATR_MULT * aa)
    )
    br_shock_cnt = (
        int(w_vh_min < -KINETIC_SHOCK_ATR_MULT * ah) +
        int(w_vo_min < -KINETIC_SHOCK_ATR_MULT * ao) +
        int(w_vm_max > KINETIC_SHOCK_ATR_MULT * am) +
        int(w_va_max > KINETIC_SHOCK_ATR_MULT * aa)
    )

    bull_shock = (b_shock_cnt >= SHOCK_QUORUM) and (vh > SHOCK_FLOW_MIN_THRESHOLD or vo > SHOCK_FLOW_MIN_THRESHOLD)
    bear_shock = (br_shock_cnt >= SHOCK_QUORUM) and (vh < -SHOCK_FLOW_MIN_THRESHOLD or vo < -SHOCK_FLOW_MIN_THRESHOLD)

    # --- GRANULAR OPPOSING SCORE (Parameterized) ---
    bull_score = (
        int(p_std > 0 and p_di > p_mean + OPPOSING_BB_STD_MULT * p_std) +
        int(vh > OPPOSING_VEL_ATR_MULT * ah) +
        int(vo > OPPOSING_VEL_ATR_MULT * ao) +
        int(h_std > 0 and hist > h_mean + OPPOSING_BB_STD_MULT * h_std)
    )
    bear_score = (
        int(m_std > 0 and m_di > m_mean + OPPOSING_BB_STD_MULT * m_std) +
        int(vh < -OPPOSING_VEL_ATR_MULT * ah) +
        int(vo < -OPPOSING_VEL_ATR_MULT * ao) +
        int(h_std > 0 and hist < h_mean - OPPOSING_BB_STD_MULT * h_std)
    )

    return (
        bull_score, bear_score,
        bull_score >= WARZONE_OPPOSING_SCORE,
        bear_score >= WARZONE_OPPOSING_SCORE,
        bull_struct, bear_struct, bull_shock, bear_shock
    )

def calculate_grids(kin_1m, idx, direction):
    hist, h_mean, h_std = kin_1m['hist'][idx], kin_1m['h_mean'][idx], kin_1m['h_std'][idx]
    obv, o_mean, o_std = kin_1m['obv'][idx], kin_1m['o_mean'][idx], kin_1m['o_std'][idx]
    p_di, p_mean, p_std = kin_1m['plus_di'][idx], kin_1m['p_mean'][idx], kin_1m['p_std'][idx]
    m_di, m_mean, m_std = kin_1m['minus_di'][idx], kin_1m['m_mean'][idx], kin_1m['m_std'][idx]
    adx, a_mean, a_std = kin_1m['adx'][idx], kin_1m['a_mean'][idx], kin_1m['a_std'][idx]

    vh, ah = kin_1m['vel_hist'][idx], kin_1m['atr_hist'][idx]
    vo, ao = kin_1m['vel_obv'][idx], kin_1m['atr_obv'][idx]
    vp, ap = kin_1m['vel_pdi'][idx], kin_1m['atr_pdi'][idx]
    vm, am = kin_1m['vel_mdi'][idx], kin_1m['atr_mdi'][idx]
    va, aa = kin_1m['vel_adx'][idx], kin_1m['atr_adx'][idx]

    atr_grid = {m: 0 for m in GRID_MULTIPLIERS}
    bb_grid = {m: 0 for m in GRID_MULTIPLIERS}

    def check_bb(val, mean, std, mult, is_bull_ind):
        if std <= 0:
            return False
        return (val > mean + (std * mult)) if is_bull_ind else (val < mean - (std * mult))

    for mult in GRID_MULTIPLIERS:
        a_count, b_count = 0, 0
        if direction == "BULL":
            if vh > mult * ah: a_count += 1
            if vo > mult * ao: a_count += 1
            if vp > mult * ap: a_count += 1
            if va > mult * aa: a_count += 1
            if check_bb(hist, h_mean, h_std, mult, True): b_count += 1
            if check_bb(obv, o_mean, o_std, mult, True): b_count += 1
            if check_bb(p_di, p_mean, p_std, mult, True): b_count += 1
            if check_bb(adx, a_mean, a_std, mult, True): b_count += 1
        else:
            if vh < -mult * ah: a_count += 1
            if vo < -mult * ao: a_count += 1
            if vm > mult * am: a_count += 1
            if va > mult * aa: a_count += 1
            if check_bb(hist, h_mean, h_std, mult, False): b_count += 1
            if check_bb(obv, o_mean, o_std, mult, False): b_count += 1
            if check_bb(m_di, m_mean, m_std, mult, True): b_count += 1
            if check_bb(adx, a_mean, a_std, mult, True): b_count += 1
        atr_grid[mult], bb_grid[mult] = a_count, b_count
    return atr_grid, bb_grid

def evaluate_anchor_tripwire(close, dt_1m, kin_1m, base_atr, bb_upper, bb_lower, kc_upper, kc_lower, today_start, is_late_session, end=None):
    n = len(close) if end is None else min(int(end), len(close))
    if today_start >= n:
        return None, {
            'dir': 'NONE', 'anchor_time': '-', 'anchor_price': 0.0,
            'killed_time': '-', 'killed_price': 0.0, 'reason': 'No data for today', 'kills': []
        }

    hard_mult = float(HARD_STOP_ATR_MULT or 0.0)
    soft_mult = float(SOFT_STOP_ATR_MULT)
    opp_thresh = int(SOFT_STOP_OPPOSING_SCORE)
    lock_trig_r = float(PROFIT_LOCK_TRIGGER_R)
    lock_floor_r = float(PROFIT_LOCK_FLOOR_R)
    soft_disable_r = float(SOFT_STOP_DISABLE_R)

    i = today_start
    survived_anchor = None
    last_killed_info = {
        'dir': 'NONE', 'anchor_time': '-', 'anchor_price': 0.0,
        'killed_time': '-', 'killed_price': 0.0, 'reason': 'Failed Kinetic Alignment'
    }
    kills, warzone_kills = [], 0

    while i < n:
        anchor = None
        while i < n:
            (bull_score, bear_score, r_bull, r_bear, bull_struct, bear_struct, bull_shock, bear_shock) = get_kinetics(kin_1m, i)

            # --- LOCK 1: Price Volatility Expansion ---
            bb_kc_bull_fire = (bb_upper[i] > kc_upper[i]) or (close[i] > bb_upper[i] and close[i] > kc_upper[i])
            bb_kc_bear_fire = (bb_lower[i] < kc_lower[i]) or (close[i] < bb_lower[i] and close[i] < kc_lower[i])

            if not is_late_session[i] and bull_struct and bull_shock and bb_kc_bull_fire:
                if r_bear:
                    warzone_kills += 1
                else:
                    anchor = {"dir": "BULL", "idx": i, "time": dt_1m[i], "price": close[i]}
                    break

            elif not is_late_session[i] and bear_struct and bear_shock and bb_kc_bear_fire:
                if r_bull:
                    warzone_kills += 1
                else:
                    anchor = {"dir": "BEAR", "idx": i, "time": dt_1m[i], "price": close[i]}
                    break
            i += 1

        if not anchor:
            break

        survived, is_bull = True, anchor['dir'] == "BULL"
        peak_price, true_peak, j = anchor['price'], anchor['price'], anchor['idx'] + 1

        while j < n:
            (bull_score, bear_score, _, _, _, _, _, _) = get_kinetics(kin_1m, j)
            if is_bull:
                peak_price, true_peak = max(peak_price, close[j]), max(true_peak, close[j])
                adverse, hard_adverse, opposing = peak_price - close[j], true_peak - close[j], bear_score
                soft_reason = f"Kinetic SL ({soft_mult:g} ATR Pullback)"
            else:
                peak_price, true_peak = min(peak_price, close[j]), min(true_peak, close[j])
                adverse, hard_adverse, opposing = close[j] - peak_price, close[j] - true_peak, bull_score
                soft_reason = f"Kinetic SL ({soft_mult:g} ATR Rally)"

            max_r = (true_peak - anchor['price']) / base_atr if is_bull else (anchor['price'] - true_peak) / base_atr
            curr_r = (close[j] - anchor['price']) / base_atr if is_bull else (anchor['price'] - close[j]) / base_atr

            reason = None

            # 1. Unconditional Hard Stop Trailing Floor
            if hard_mult > 0 and hard_adverse >= hard_mult * base_atr:
                reason = f"Hard stop ({hard_mult:g} ATR trailing floor)"

            # 2. Profit Ratchet: Once up PROFIT_LOCK_TRIGGER_R, never drop below PROFIT_LOCK_FLOOR_R
            elif lock_trig_r > 0 and max_r >= lock_trig_r and curr_r <= lock_floor_r:
                reason = f"Profit Lock (+{lock_trig_r:g}R Peak -> +{lock_floor_r:g}R Breakeven Floor)"

            # 3. Soft Kinetic Stop (Active below SOFT_STOP_DISABLE_R)
            elif max_r < soft_disable_r and adverse >= soft_mult * base_atr and opposing >= opp_thresh:
                reason = soft_reason

            if reason:
                last_killed_info = {
                    'dir': anchor['dir'],
                    'anchor_time': pd.to_datetime(anchor['time']).strftime('%H:%M'),
                    'anchor_price': float(anchor['price']),
                    'killed_time': pd.to_datetime(dt_1m[j]).strftime('%H:%M'),
                    'killed_price': float(close[j]),
                    'reason': reason
                }
                kills.append(last_killed_info)
                survived = False
                i = j + 1
                break
            j += 1

        if survived:
            survived_anchor = anchor
            break

    if survived_anchor:
        return survived_anchor, {"status": "Survived", "kills": kills}
    info = dict(last_killed_info)
    if warzone_kills > 0 and info['dir'] == 'NONE':
        info['reason'] = f"Warzone Inversion Chop ({warzone_kills}x)"
    info['kills'] = kills
    return None, info

def prepare_series(master, target_dt):
    if master is None or master.empty:
        return None, "no_data"
    day_mask = (master['Datetime'].dt.date == target_dt).values
    if not day_mask.any():
        return None, "no_data"
    ts = int(np.argmax(day_mask))

    open_ = master['Open'].values.astype(float)
    close = master['Close'].values.astype(float)
    high = master['High'].values.astype(float)
    low = master['Low'].values.astype(float)
    vol = master['Volume'].values.astype(float)
    sid, n = master['SessId'].values, len(close)

    sess_starts = np.zeros(n, dtype=bool)
    sess_starts[0] = True
    sess_starts[1:] = sid[1:] != sid[:-1]

    # Precompute closing auction mask using global LAST_NEW_ANCHOR_MIN
    is_late_session = (master['Min'].values >= LAST_NEW_ANCHOR_MIN)

    base_atr_calc, atr_bars = compute_base_atr(master.iloc[:ts])
    if base_atr_calc is None or atr_bars < ATR_MIN_BARS:
        return None, "short_history"
    base_atr = max(float(base_atr_calc), float(close[ts]) * float(MIN_ATR_PCT), float(ABSOLUTE_MIN_ATR))

    tr = np.maximum(
        high - low,
        np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1)))
    )
    tr = np.where(sess_starts, high - low, tr)

    if ISOLATE_SESSION_BANDS:
        df_p = pd.DataFrame({'close': close, 'tr': tr, 'sid': sid})
        grouped_p = df_p.groupby('sid')
        sma_bb = grouped_p['close'].transform(lambda x: x.rolling(PRICE_BB_PERIOD, min_periods=ROLLING_MIN_PERIODS).mean()).values
        std_bb = grouped_p['close'].transform(lambda x: x.rolling(PRICE_BB_PERIOD, min_periods=ROLLING_MIN_PERIODS).std(ddof=0)).values
        sma_kc = grouped_p['close'].transform(lambda x: x.rolling(PRICE_KC_PERIOD, min_periods=ROLLING_MIN_PERIODS).mean()).values
        atr_kc = grouped_p['tr'].transform(lambda x: x.rolling(PRICE_KC_PERIOD, min_periods=ROLLING_MIN_PERIODS).mean()).values
    else:
        # Gap-Stitched Continuous Price Baseline: removes overnight jump while keeping 19 quiet prior-close bars at 09:15
        delta_p = np.diff(close, prepend=close[0])
        delta_p = np.where(sess_starts, close - open_, delta_p)
        stitched_close = np.cumsum(delta_p) + open_[0]
        stitched_offset = close - stitched_close
        sc_series = pd.Series(stitched_close)
        tr_series = pd.Series(tr)
        sma_bb = sc_series.rolling(PRICE_BB_PERIOD, min_periods=ROLLING_MIN_PERIODS).mean().values + stitched_offset
        std_bb = sc_series.rolling(PRICE_BB_PERIOD, min_periods=ROLLING_MIN_PERIODS).std(ddof=0).values
        sma_kc = sc_series.rolling(PRICE_KC_PERIOD, min_periods=ROLLING_MIN_PERIODS).mean().values + stitched_offset
        atr_kc = tr_series.rolling(PRICE_KC_PERIOD, min_periods=ROLLING_MIN_PERIODS).mean().values

    return {
        'close': close, 'dt': master['Datetime'].values, 'today_start': ts, 'is_late_session': is_late_session,
        'day_open': float(open_[ts]), 'prev_close': float(close[ts - 1]) if ts > 0 else 0.0,
        'base_atr': base_atr, 'atr_bars': int(atr_bars),
        'kin': _evaluate_kinetic_arrays(open_, close, high, low, vol, sess_starts, sid),
        'bb_upper': sma_bb + PRICE_BB_STD_MULT * std_bb,
        'bb_lower': sma_bb - PRICE_BB_STD_MULT * std_bb,
        'kc_upper': sma_kc + PRICE_KC_ATR_MULT * atr_kc,
        'kc_lower': sma_kc - PRICE_KC_ATR_MULT * atr_kc,
    }, None

def row_at(S, end, symbol):
    ts = S['today_start']
    end = min(int(end), len(S['close']))
    if end <= ts:
        return None
    close, base_atr, kin = S['close'], S['base_atr'], S['kin']
    last = end - 1
    ltp = float(close[last])
    ref = S['prev_close'] if (DAY_PCT_BASIS == "PREV_CLOSE" and S['prev_close'] > 0) else S['day_open']

    row = {
        'Symbol': symbol, 'LTP': ltp, 'DayChangePct': ((ltp - ref) / ref) * 100.0 if ref else 0.0,
        'ActiveAnchor': None, 'AnchorPrice': 0.0, 'AnchorDir': "NONE", 'MovePct': 0.0,
        'State': "NONE", 'Blocks': 0, 'KilledTime': '-', 'KilledPrice': 0.0, 'RejectReason': "",
        'Kills': [], 'AtrBars': S['atr_bars'], 'BaseAtr': base_atr, 'PriorStops': 0, 'BankedR': 0.0
    }

    anchor, info = evaluate_anchor_tripwire(
        close, S['dt'], kin, base_atr, S['bb_upper'], S['bb_lower'],
        S['kc_upper'], S['kc_lower'], ts, S['is_late_session'], end
    )
    row['Kills'] = info.get('kills', [])
    eval_dir = anchor['dir'] if anchor else info.get('dir')
    if not eval_dir or eval_dir == "NONE":
        eval_dir = "BULL"

    row['ATR_Grid'], row['BB_Grid'] = calculate_grids(kin, last, eval_dir)
    bb_expanding = (S['bb_upper'][last] > S['kc_upper'][last]) or (S['bb_lower'][last] < S['kc_lower'][last])
    pure_state = "[ BB > KC ]" if bb_expanding else "[ BB < KC ]"

    if anchor:
        row['ActiveAnchor'] = pd.to_datetime(anchor['time']).strftime("%H:%M")
        row['AnchorPrice'] = float(anchor['price'])
        row['AnchorDir'] = anchor['dir']
        row['MovePct'] = ((ltp - anchor['price']) / anchor['price']) * 100.0 if anchor['price'] else 0.0
        row['State'], row['Blocks'] = pure_state, last - anchor['idx']
    else:
        row['AnchorDir'] = info.get('dir', 'NONE')
        row['ActiveAnchor'] = info.get('anchor_time', '-')
        row['AnchorPrice'] = float(info.get('anchor_price', 0.0))
        row['KilledTime'] = info.get('killed_time', '-')
        row['KilledPrice'] = float(info.get('killed_price', 0.0))
        row['RejectReason'] = info.get('reason', 'Failed Kinetic Alignment')
        if row['AnchorPrice'] > 0 and row['KilledPrice'] > 0:
            row['MovePct'] = ((row['KilledPrice'] - row['AnchorPrice']) / row['AnchorPrice']) * 100.0
        row['State'] = "[STOPPED]"
    return row

def compute_row(symbol, master_1m, target_dt):
    S, _ = prepare_series(master_1m, target_dt)
    return None if S is None else row_at(S, len(master_1m), symbol)

# ==============================================================================
# 3. WORKERS, FORMATTERS & MULTI-EPISODE MERGE LOGIC
# ==============================================================================
def _history_worker_daily(args):
    item, start, end, progress = args
    try:
        frames = []
        for c_start, c_end in _date_chunks(start, end + timedelta(days=1), span=CHUNK_SPAN_DAILY_DAYS):
            status, df = _candles(_url_range_daily(item['key'], c_start, c_end))
            if status == 200 and df is not None:
                frames.append(df)
        if not frames:
            return None
        master = prepare_master(frames)
        master = master[master['Datetime'].dt.date < end]
        if master.empty:
            return None
        close, vol = master['Close'].iloc[-1], master['Volume'].iloc[-1]
        return item if (MIN_PRICE <= close <= MAX_PRICE and vol >= MIN_DAILY_VOLUME) else None
    except Exception as e:
        ERRORS.add(f"prefilter:{item.get('symbol')}", e)
        return None
    finally:
        progress.tick()

def process_stock_checkpoints(args):
    item, checkpoints, use_intraday, history_days, mode, progress = args
    sym = item['symbol']
    try:
        final_cutoff = checkpoints[-1]
        target_dt = final_cutoff.date()
        frames = []
        for c_start, c_end in _date_chunks(target_dt - timedelta(days=history_days), target_dt + timedelta(days=1), span=CHUNK_SPAN_1M_DAYS):
            status, df = _candles(_url_range_1m(item['key'], c_start, c_end))
            if status == 200 and df is not None:
                frames.append(df)
        if use_intraday:
            today_df = fetch_today(item['key'])
            if today_df is not None:
                frames.append(today_df)
        if not frames:
            return [(cp, "no_data", sym) for cp in checkpoints]

        master = prepare_master(frames)
        master = master[master['Datetime'] < final_cutoff].reset_index(drop=True)
        if mode == "INDEX_OPTIONS":
            day = master[master['Datetime'].dt.date == target_dt]
            if day.empty:
                return [(cp, "no_data", sym) for cp in checkpoints]
            if day['Close'].iloc[-1] < OPT_MIN_PRICE or day['Volume'].sum() < OPT_MIN_VOLUME:
                return [(cp, "illiquid", sym) for cp in checkpoints]

        S, why = prepare_series(master, target_dt)
        if S is None:
            return [(cp, why, sym) for cp in checkpoints]
        ends = np.searchsorted(S['dt'], np.array(checkpoints, dtype='datetime64[ns]'), side='left')
        out = []
        for cp, end in zip(checkpoints, ends):
            row = row_at(S, int(end), sym)
            if row is None:
                out.append((cp, "no_data", sym))
            else:
                row['CheckpointTime'] = cp
                out.append((cp, "ok", row))
        return out
    except Exception as e:
        ERRORS.add(f"1m:{item.get('symbol')}", e)
        return [(cp, "error", item.get('symbol')) for cp in checkpoints]
    finally:
        progress.tick()

def _clip(s, w):
    return s if len(s) <= w else s[:w - 1] + "~"

def _print_error_summary():
    if STATS.failed:
        print(f"{COLOR_YELLOW}⚠ {STATS.failed} API requests failed.{COLOR_RESET}")
    if ERRORS.items:
        print(f"{COLOR_YELLOW}⚠ {len(ERRORS.items)} instruments raised exceptions. First: [{ERRORS.items[0][0]}] {ERRORS.items[0][1]}{COLOR_RESET}")

def _signal_id(row):
    return (row['AnchorDir'], row['ActiveAnchor'], round(float(row['AnchorPrice']), 6))

def _find_kill(kills, d, t):
    return next((k for k in reversed(kills or []) if k.get('dir') == d and k.get('anchor_time') == t), None)

def _exit_price(row):
    return float(row['KilledPrice']) if row.get('State') == "[STOPPED]" and row.get('KilledPrice', 0) > 0 else float(row.get('LTP', 0.0) or 0.0)

def _kill_r_multiple(kill_dict, base_atr):
    ap, kp = float(kill_dict.get('anchor_price', 0.0) or 0.0), float(kill_dict.get('killed_price', 0.0) or 0.0)
    if ap <= 0 or kp <= 0 or base_atr <= 0:
        return 0.0
    return (kp - ap) / base_atr if kill_dict.get('dir', 'BULL') == 'BULL' else (ap - kp) / base_atr

def _r_multiple(row, include_banked=False):
    ap, base_atr, live_r = row.get('AnchorPrice', 0.0), row.get('BaseAtr', 0.0), 0.0
    if ap > 0 and base_atr > 0:
        price, direction = _exit_price(row), row.get('PremiumDir', row.get('AnchorDir'))
        live_r = (price - ap) / base_atr if direction == 'BULL' else (ap - price) / base_atr
    return live_r + float(row.get('BankedR', 0.0) or 0.0) if include_banked else live_r

def _rank_r_score(row):
    """Ranks setups by the stronger of Live R or Cumulative Session R so an earlier stop never buries a big live winner."""
    live_r = _r_multiple(row, include_banked=False)
    cum_r = _r_multiple(row, include_banked=True)
    return max(live_r, cum_r)

def _merge_signal(sig_rows, last_row, direction):
    first, latest = sig_rows[0], sig_rows[-1]
    row = dict(last_row)
    row['ActiveAnchor'], row['AnchorPrice'], row['AnchorDir'] = first['ActiveAnchor'], first['AnchorPrice'], direction
    kills, base_atr = last_row.get('Kills') or [], float(last_row.get('BaseAtr', 0.0) or 0.0)
    prior_kills = [k for k in kills if k.get('dir') == direction and k.get('anchor_time') != first['ActiveAnchor']]
    row['PriorStops'], row['BankedR'] = len(prior_kills), sum(_kill_r_multiple(k, base_atr) for k in prior_kills)

    if latest['CheckpointTime'] == last_row['CheckpointTime'] and last_row.get('State') in ACTIVE_BULL_STATES and _signal_id(last_row) == _signal_id(first):
        row['State'], row['KilledTime'], row['KilledPrice'], row['RejectReason'] = last_row['State'], '-', 0.0, ""
    else:
        row['State'] = "[STOPPED]"
        k = _find_kill(kills, direction, first['ActiveAnchor'])
        if k:
            row['KilledTime'], row['KilledPrice'], row['RejectReason'] = k['killed_time'], float(k['killed_price']), k['reason']
        else:
            row['KilledTime'], row['KilledPrice'], row['RejectReason'] = latest['CheckpointTime'].strftime('%H:%M'), float(latest['LTP']), "Superseded by newer anchor"
    ap = row['AnchorPrice'] or last_row['LTP']
    row['MovePct'] = ((_exit_price(row) - ap) / ap) * 100.0 if ap else 0.0
    return row

def merge_symbol_rows(rows):
    rows = sorted(rows, key=lambda r: r['CheckpointTime'])
    last_row = rows[-1]
    kills = last_row.get('Kills') or []
    active_merged, grave_episodes, seen_anchor_times = {}, [], set()

    for direction, active in (("BULL", ACTIVE_BULL_STATES), ("BEAR", ACTIVE_BEAR_STATES)):
        episodes = {}
        for r in [r for r in rows if r['AnchorDir'] == direction and r['State'] in active]:
            episodes.setdefault(_signal_id(r), []).append(r)
        for ep_rows in episodes.values():
            merged_ep = _merge_signal(ep_rows, last_row, direction)
            seen_anchor_times.add((direction, merged_ep['ActiveAnchor']))
            if merged_ep['State'] != "[STOPPED]":
                active_merged[direction] = merged_ep
            else:
                grave_episodes.append(merged_ep)
        for k in [k for k in kills if k.get('dir') == direction and (direction, k.get('anchor_time')) not in seen_anchor_times]:
            seen_anchor_times.add((direction, k.get('anchor_time')))
            g_row = dict(last_row)
            g_row['AnchorDir'], g_row['ActiveAnchor'], g_row['AnchorPrice'] = direction, k['anchor_time'], float(k['anchor_price'])
            g_row['KilledTime'], g_row['KilledPrice'], g_row['RejectReason'], g_row['State'] = k['killed_time'], float(k['killed_price']), k['reason'], "[STOPPED]"
            g_row['MovePct'] = ((g_row['KilledPrice'] - g_row['AnchorPrice']) / g_row['AnchorPrice']) * 100.0 if g_row['AnchorPrice'] > 0 else 0.0
            grave_episodes.append(g_row)
    if not active_merged and not grave_episodes:
        grave = dict(last_row)
        grave['CheckpointTime'] = rows[0]['CheckpointTime']
        grave_episodes.append(grave)
    return active_merged.get("BULL"), active_merged.get("BEAR"), grave_episodes

def _state_display(row):
    return f"[STOP {row.get('KilledTime')}]" if row['State'] == "[STOPPED]" and row.get('KilledTime') and row.get('KilledTime') != '-' else row['State']

def _fav_color(pct, direction):
    fav = pct * (1 if direction == "BULL" else -1)
    return COLOR_GREEN_FG if fav > 0 else COLOR_RED_FG if fav < 0 else COLOR_DIM

def _format_val(v, dir_color):
    return f"{dir_color}{v}{COLOR_RESET}" if v > 0 else f"{COLOR_DIM}{v}{COLOR_RESET}"

# ------------------------------------------------------------------------------
# TRACKING EMAIL & INTRADAY HISTORY
# ------------------------------------------------------------------------------
def _send_watch_email(subject, body, tag="track"):
    sender = os.environ.get("SENDER_EMAIL")
    password = os.environ.get("SENDER_PASSWORD")
    recipient = os.environ.get("RECIPIENT_EMAIL")
    smtp_server = os.environ.get("SMTP_SERVER") or DEFAULT_SMTP_SERVER
    smtp_port = os.environ.get("SMTP_PORT") or DEFAULT_SMTP_PORT
    if not sender or not password or not recipient:
        return False
    try:
        port = int(smtp_port)
    except ValueError:
        return False
    msg = EmailMessage()
    msg['Subject'], msg['From'], msg['To'] = subject, sender, recipient
    msg.set_content(body)
    try:
        if port == SMTP_SSL_PORT:
            with smtplib.SMTP_SSL(smtp_server, port, timeout=SMTP_TIMEOUT_SEC) as server:
                server.login(sender, password)
                server.send_message(msg)
        else:
            with smtplib.SMTP(smtp_server, port, timeout=SMTP_TIMEOUT_SEC) as server:
                server.starttls()
                server.login(sender, password)
                server.send_message(msg)
        return True
    except Exception:
        return False

def print_and_alert_tracked_history(tracked_symbol_raw, by_symbol, live_run=True):
    if not tracked_symbol_raw:
        return
    n = _norm_sym(tracked_symbol_raw)
    matched_sym = next((sym for sym in by_symbol.keys() if n in _norm_sym(sym)), None)
    if not matched_sym:
        return
    rows = sorted(by_symbol[matched_sym], key=lambda r: r['CheckpointTime'])
    history, is_pe = [], " PE " in matched_sym or matched_sym.endswith("PE")
    for i in range(1, len(rows) + 1):
        subset = rows[:i]
        b, r_, g_list = merge_symbol_rows(subset)
        selected_row = b if (b and b['State'] != "[STOPPED]") else r_ if (r_ and r_['State'] != "[STOPPED]") else (g_list[-1] if g_list else None)
        if selected_row:
            row_copy = dict(selected_row)
            raw_d = row_copy.get('AnchorDir', 'BULL')
            if raw_d in ("BULL", "BEAR"):
                row_copy['PremiumDir'] = raw_d
                if is_pe:
                    row_copy['AnchorDir'] = "BEAR" if raw_d == "BULL" else "BULL"
            row_copy['_EvalTime'] = subset[-1]['CheckpointTime']
            history.append(row_copy)
    if not history:
        return

    print(f"\n{COLOR_BOLD}🔍 INTRADAY HISTORY FOR TRACKED SYMBOL: {matched_sym}{COLOR_RESET}")
    sym_w = max(MIN_SYMBOL_WIDTH, len(matched_sym))
    grid_hdr = " ".join(f"{m}X" for m in GRID_MULTIPLIERS)
    header_str = f" {COLOR_CYAN}{'Script':<{sym_w}} {'LTP/Exit':>8} | {'Eval':^5} {'Anchor':^7} {'Anch Val':>9} {'R-Mult (Bank)':>15} | ATR: {grid_hdr} | BB: {grid_hdr} | {'State':^13}{COLOR_RESET}"
    print(header_str)
    print("-" * len(ANSI_RE.sub("", header_str)))
    for row in history:
        live_r, banked_r = _r_multiple(row, include_banked=False), float(row.get('BankedR', 0.0) or 0.0)
        rank_r = _rank_r_score(row)
        r_color = COLOR_GREEN_FG if rank_r > 0 else COLOR_RED_FG if rank_r < 0 else COLOR_DIM
        if row.get('PriorStops', 0) > 0 and abs(banked_r) > BANKED_R_DISPLAY_EPSILON:
            r_mult_str, a_time = f"{live_r:>+5.2f}R ({banked_r:>+4.1f}R)", f"{row.get('ActiveAnchor', '-')}*"
        else:
            r_mult_str, a_time = f"{live_r:>+5.2f}R        ", f"{row.get('ActiveAnchor', '-')} "
        direction = row.get('AnchorDir', 'BULL')
        dir_c = COLOR_GREEN_FG if direction == "BULL" else COLOR_RED_FG
        default_g = {m: 0 for m in GRID_MULTIPLIERS}
        ag, bg = row.get('ATR_Grid', default_g), row.get('BB_Grid', default_g)
        a_str = "  ".join(_format_val(ag.get(k, 0), dir_c) for k in GRID_MULTIPLIERS)
        b_str = "  ".join(_format_val(bg.get(k, 0), dir_c) for k in GRID_MULTIPLIERS)
        shown_state = _state_display(row)
        if "[ BB > KC ]" in row['State']:
            dir_tag = "▲" if direction == "BULL" else "▼" if direction == "BEAR" else ""
            state_text = f"{COLOR_GREEN_FG if dir_tag == '▲' else COLOR_RED_FG}{f'[ BB > KC ] {dir_tag}'.strip():^13}{COLOR_RESET}"
        else:
            state_text = f"{COLOR_YELLOW if '[ BB < KC ]' in row['State'] else COLOR_DIM}{shown_state:^13}{COLOR_RESET}"
        ltp_val = f"{row['KilledPrice']:>7.2f}x" if "[STOP" in shown_state and row.get('KilledPrice', 0) > 0 else f"{row['LTP']:>8.2f} "
        print(f" {COLOR_BOLD}{_clip(row['Symbol'], sym_w):<{sym_w}}{COLOR_RESET} {ltp_val} | {row['_EvalTime'].strftime('%H:%M'):^5} {a_time:^7} {row['AnchorPrice']:>9.2f} {r_color}{r_mult_str:>15}{COLOR_RESET} |   {a_str}  |   {b_str}  | {state_text}")

    if len(history) >= 2 and live_run:
        last_row, prev_row = history[-1], history[-2]
        def get_full_state(r):
            return f"{r.get('AnchorDir', 'NONE')} {r.get('State', 'NONE')}"
        last_state, prev_state = get_full_state(last_row), get_full_state(prev_row)
        if last_state != prev_state:
            _send_watch_email(
                f"[System3] {matched_sym} Alert: {prev_state} -> {last_state}",
                f"System3 detected a state transition for {matched_sym} at checkpoint {last_row['_EvalTime'].strftime('%H:%M')}.\n\n"
                f"Previous State: {prev_state}\nCurrent State: {last_state}\n\n"
                f"LTP: {last_row.get('LTP', 0):.2f}\nTotal Session Move: {_r_multiple(last_row, include_banked=True):+.2f} R-Multiples\n",
                "track"
            )

# ==============================================================================
# MAIN RUNNER
# ==============================================================================
def run_screener(mode=TRADING_MODE, days=BACKTRACE_DAYS, target_date_str=None, target_time_str=None, track_symbol=""):
    t_start = time.time()
    explicit = bool(target_date_str or target_time_str)
    cutoff_dt = (
        datetime.strptime(f"{target_date_str or now_ist().strftime('%Y-%m-%d')} {target_time_str or DEFAULT_SNAPSHOT_TIME}", "%Y-%m-%d %H:%M")
        if explicit else now_ist()
    )
    cutoff_dt, notes = resolve_session(cutoff_dt.replace(second=0, microsecond=0))
    if STATS.auth_failed:
        return 1
    target_dt, today_session = cutoff_dt.date(), (cutoff_dt.date() == now_ist().date())
    cutoff_dt = min(cutoff_dt, _close_dt(target_dt))

    print(f"\n{COLOR_CYAN}📡 Initializing Tracker [{mode}] | Time Machine: {cutoff_dt.strftime('%Y-%m-%d %H:%M:%S')} (Live: {not explicit}){COLOR_RESET}")
    for n in notes:
        print(f"   {COLOR_YELLOW}» {n}{COLOR_RESET}")

    universe_raw = get_dynamic_universe(mode, cutoff_dt, today_session, needles=[track_symbol] if track_symbol else [])
    if STATS.auth_failed or not universe_raw:
        return 1
    print(f"   {COLOR_DIM}» Universe: {len(universe_raw)} instruments{COLOR_RESET}")

    if mode == "INDEX_OPTIONS":
        candidates = universe_raw
    else:
        prog = Progress("prefilter", len(universe_raw))
        lookback_cal_days = days * PREFILTER_LOOKBACK_MULT + PREFILTER_LOOKBACK_PAD_DAYS
        with concurrent.futures.ThreadPoolExecutor(max_workers=WORKERS) as ex:
            candidates = [r for r in ex.map(_history_worker_daily, [(it, target_dt - timedelta(days=lookback_cal_days), target_dt, prog) for it in universe_raw]) if r is not None]
        prog.done()
        if STATS.auth_failed or not candidates:
            return 1
        print(f"   {COLOR_DIM}» Prefilter: {len(universe_raw)} -> {len(candidates)} instruments{COLOR_RESET}")

    checkpoints = generate_checkpoints(cutoff_dt) if BACKTRACE_CHECKPOINTS else [cutoff_dt]
    print(f"   {COLOR_DIM}» Backtrace checkpoints ({len(checkpoints)}): {', '.join(c.strftime('%H:%M') for c in checkpoints)}{COLOR_RESET}")

    prog = Progress("checkpoints", len(candidates))
    with concurrent.futures.ThreadPoolExecutor(max_workers=WORKERS) as ex:
        per_symbol = list(ex.map(process_stock_checkpoints, [(item, checkpoints, today_session, MIN1_HISTORY_DAYS, mode, prog) for item in candidates]))
    prog.done()

    if STATS.auth_failed:
        return 1
    outcomes = [o for sym_outcomes in per_symbol for o in sym_outcomes]
    results = [p for _, s, p in outcomes if s == "ok"]
    counts = {k: sum(1 for _, s, _ in outcomes if s == k) for k in ("no_data", "illiquid", "short_history", "error")}
    print(f"   {COLOR_DIM}» Scan summary: {len(candidates)} instruments x {len(checkpoints)} checkpoints = {len(outcomes)} evaluations | ok {len(results)} | no data {counts['no_data']} | illiquid {counts['illiquid']} | short history {counts['short_history']} | errors {counts['error']}{COLOR_RESET}")
    if not results:
        return 1

    by_symbol, bulls_live, bears_live, graveyard_candidates = {}, [], [], []
    for row in results:
        by_symbol.setdefault(row['Symbol'], []).append(row)

    for symbol, rows in by_symbol.items():
        b, r_, grave_list = merge_symbol_rows(rows)
        is_pe = " PE " in symbol or symbol.endswith("PE")
        def _apply_dir(item, raw_dir):
            item['PremiumDir'] = raw_dir
            item['AnchorDir'] = "BEAR" if (raw_dir == "BULL" and is_pe) else ("BULL" if (raw_dir == "BEAR" and is_pe) else raw_dir)
        for item, raw_dir in ((b, "BULL"), (r_, "BEAR")):
            if not item:
                continue
            _apply_dir(item, raw_dir)
            if item['State'] == "[STOPPED]" or "[STOP" in str(item.get('State', '')):
                graveyard_candidates.append(item)
            else:
                (bulls_live if item['AnchorDir'] == "BULL" else bears_live).append(item)
        for g_item in (grave_list or []):
            if g_item.get('AnchorDir', 'NONE') in ("BULL", "BEAR"):
                _apply_dir(g_item, g_item.get('AnchorDir'))
            graveyard_candidates.append(g_item)

    if track_symbol:
        print_and_alert_tracked_history(track_symbol, by_symbol, live_run=not explicit)

    drift_floor = (
        float(ADVERSE_DRIFT_R_FLOOR)
        if ADVERSE_DRIFT_R_FLOOR is not None
        else -abs(float(HARD_STOP_ATR_MULT or 1.2) * float(ADVERSE_DRIFT_STOP_RATIO))
    )

    surviving_bulls = [r for r in bulls_live if _r_multiple(r) > drift_floor or _r_multiple(r, include_banked=True) > BANKED_R_PROTECTION_FLOOR]
    surviving_bears = [r for r in bears_live if _r_multiple(r) > drift_floor or _r_multiple(r, include_banked=True) > BANKED_R_PROTECTION_FLOOR]
    for r in bulls_live:
        if r not in surviving_bulls:
            r['RejectReason'] = f"Adverse Drift (Losing R-Multiple < {drift_floor:g}R)"
            graveyard_candidates.append(r)
    for r in bears_live:
        if r not in surviving_bears:
            r['RejectReason'] = f"Adverse Drift (Losing R-Multiple < {drift_floor:g}R)"
            graveyard_candidates.append(r)

    first_grid_key = GRID_MULTIPLIERS[0] if GRID_MULTIPLIERS else 1
    def state_rank(st):
        return 0 if "[ BB > KC ]" in st else 1 if "[ BB < KC ]" in st else 2

    surviving_bulls.sort(key=lambda r: (
        -_rank_r_score(r), -_r_multiple(r), state_rank(r['State']),
        -r.get('ATR_Grid', {}).get(first_grid_key, 0), -r.get('BB_Grid', {}).get(first_grid_key, 0), -int(r['CheckpointTime'].timestamp())
    ))
    surviving_bears.sort(key=lambda r: (
        -_rank_r_score(r), -_r_multiple(r), state_rank(r['State']),
        -r.get('ATR_Grid', {}).get(first_grid_key, 0), -r.get('BB_Grid', {}).get(first_grid_key, 0), -int(r['CheckpointTime'].timestamp())
    ))
    bulls, bears = surviving_bulls[:TOP_N_BUYERS], surviving_bears[:TOP_N_SELLERS]

    def sort_reason(r):
        return 0 if any(x in str(r.get('RejectReason', '')) for x in ["Hard stop", "Profit Lock", "Kinetic SL", "Superseded"]) else 1 if "Adverse" in str(r.get('RejectReason', '')) else 2 if "Warzone" in str(r.get('RejectReason', '')) else 3
    graveyard_candidates.sort(key=lambda x: (sort_reason(x), -_r_multiple(x) if _r_multiple(x) != 0 else -abs(x.get('DayChangePct', 0.0))))

    dedup_grave, seen_grave = [], set()
    for row in graveyard_candidates:
        gid = (row['Symbol'], row.get('ActiveAnchor', '-'), row.get('KilledTime', '-'))
        if gid not in seen_grave:
            seen_grave.add(gid)
            dedup_grave.append(row)
    grave_total, grave_shown = len(dedup_grave), dedup_grave if TOP_N_GRAVEYARD <= 0 else dedup_grave[:TOP_N_GRAVEYARD]
    sym_w = max(MIN_SYMBOL_WIDTH, min(MAX_SYMBOL_WIDTH, max((len(r['Symbol']) for r in bulls + bears + grave_shown), default=MIN_SYMBOL_WIDTH)))

    print(f"\n{COLOR_BOLD}=== STATEFUL INSTITUTIONAL VOLATILITY TRACKER [{mode}] ==={COLOR_RESET}")
    print(f"Target Snapshot: {cutoff_dt.strftime('%Y-%m-%d %H:%M')} | Scanned: {len(by_symbol)} symbols across {len(checkpoints)} checkpoints | Surviving: {len(bulls)} bull, {len(bears)} bear | Graveyard: {grave_total}\n")

    def print_basket(title, icon, data_list):
        if not data_list:
            return
        print(f"\n{COLOR_BOLD}{icon} {title}{COLOR_RESET}")
        grid_hdr = " ".join(f"{m}X" for m in GRID_MULTIPLIERS)
        header_str = f" {COLOR_CYAN}{'Script':<{sym_w}} {'LTP/Exit':>8} | {'Seen':^5} {'Anchor':^7} {'Anch Val':>9} {'R-Mult (Bank)':>15} | ATR: {grid_hdr} | BB: {grid_hdr} | {'State':^13}{COLOR_RESET}"
        print(header_str)
        print("-" * len(ANSI_RE.sub("", header_str)))
        basket_dir = "BULL" if "BULL" in title else "BEAR"
        default_g = {m: 0 for m in GRID_MULTIPLIERS}
        for row in data_list:
            live_r, banked_r = _r_multiple(row, include_banked=False), float(row.get('BankedR', 0.0) or 0.0)
            rank_r = _rank_r_score(row)
            r_color = COLOR_GREEN_FG if rank_r > 0 else COLOR_RED_FG if rank_r < 0 else COLOR_DIM
            if row.get('PriorStops', 0) > 0 and abs(banked_r) > BANKED_R_DISPLAY_EPSILON:
                r_mult_str, anchor_disp = f"{live_r:>+5.2f}R ({banked_r:>+4.1f}R)", f"{row['ActiveAnchor']}*"
            else:
                r_mult_str, anchor_disp = f"{live_r:>+5.2f}R        ", f"{row['ActiveAnchor']} "
            dir_c = COLOR_GREEN_FG if basket_dir == "BULL" else COLOR_RED_FG
            ag, bg = row.get('ATR_Grid', default_g), row.get('BB_Grid', default_g)
            a_str = "  ".join(_format_val(ag.get(k, 0), dir_c) for k in GRID_MULTIPLIERS)
            b_str = "  ".join(_format_val(bg.get(k, 0), dir_c) for k in GRID_MULTIPLIERS)
            shown_state = _state_display(row)
            if "[ BB > KC ]" in row['State']:
                dir_tag = "▲" if basket_dir == "BULL" else "▼"
                state_text = f"{COLOR_GREEN_FG if dir_tag == '▲' else COLOR_RED_FG}{f'[ BB > KC ] {dir_tag}'.strip():^13}{COLOR_RESET}"
            else:
                state_text = f"{COLOR_YELLOW if '[ BB < KC ]' in row['State'] else COLOR_DIM}{shown_state:^13}{COLOR_RESET}"
            ltp_val = f"{row['KilledPrice']:>7.2f}x" if "[STOP" in shown_state and row.get('KilledPrice', 0) > 0 else f"{row['LTP']:>8.2f} "
            print(f" {COLOR_BOLD}{_clip(row['Symbol'], sym_w):<{sym_w}}{COLOR_RESET} {ltp_val} | {row['CheckpointTime'].strftime('%H:%M'):^5} {anchor_disp:^7} {row['AnchorPrice']:>9.2f} {r_color}{r_mult_str:>15}{COLOR_RESET} |   {a_str}  |   {b_str}  | {state_text}")

    print_basket("TOP BULL SETUPS (Valid Anchors Surviving)", "🔥", bulls)
    print_basket("TOP BEAR SETUPS (Valid Anchors Surviving)", "🩸", bears)
    if not bulls and not bears:
        print(f"{COLOR_YELLOW}No surviving anchors at this snapshot -- every candidate was rejected (see graveyard).{COLOR_RESET}")

    if grave_shown:
        print(f"\n{COLOR_BOLD}🚫 THE GRAVEYARD (Completed Episodes / Filtered / Rejected){COLOR_RESET}")
        header_str = f" {COLOR_CYAN}{'Script':<{sym_w}} {'Signal':^6} {'Anchor':^6} {'Anch Val':>9} {'Killed':^6} {'Kill Val':>9} {'LTP':>8} {'R-Mult':>7} {'Move%':>7} | {'Forensic Rejection / Exit Reason'}{COLOR_RESET}"
        print(header_str)
        print("-" * len(ANSI_RE.sub("", header_str)))
        for row in grave_shown:
            sig = row.get('AnchorDir', 'NONE')
            sig_colored = f"{COLOR_GREEN_FG}BULL{COLOR_RESET}  " if sig == "BULL" else f"{COLOR_RED_FG}BEAR{COLOR_RESET}  " if sig == "BEAR" else f"{COLOR_DIM}NONE{COLOR_RESET}  "
            a_time, a_price = row['ActiveAnchor'] if row['ActiveAnchor'] else '-', f"{row['AnchorPrice']:>9.2f}" if row['AnchorPrice'] > 0 else f"{'-':>9}"
            k_time, k_price = row['KilledTime'] if row['KilledTime'] else '-', f"{row['KilledPrice']:>9.2f}" if row['KilledPrice'] > 0 else f"{'-':>9}"
            if row['AnchorPrice'] > 0 and (row['KilledPrice'] > 0 or row['LTP'] > 0):
                g_r = _r_multiple(row, include_banked=False)
                g_r_str, g_r_color = f"{g_r:>+6.2f}R", COLOR_GREEN_FG if g_r > 0 else COLOR_RED_FG if g_r < 0 else COLOR_DIM
                move_pct_str, move_color = f"{row['MovePct']:>+6.2f}%", _fav_color(row['MovePct'], row.get('PremiumDir', sig))
            else:
                g_r_str, g_r_color, move_pct_str, move_color = f"{'-':>7}", COLOR_DIM, f"{'-':>7}", COLOR_DIM
            reason = row['RejectReason']
            reason_color = COLOR_YELLOW if any(x in reason for x in ["Kinetic SL", "Hard stop", "Adverse", "Profit Lock", "Superseded"]) else COLOR_RED_FG if any(x in reason for x in ["Chop", "Warzone"]) else COLOR_DIM
            print(f" {COLOR_BOLD}{_clip(row['Symbol'], sym_w):<{sym_w}}{COLOR_RESET} {sig_colored} {a_time:^6} {a_price} {k_time:^6} {k_price} {row['LTP']:>8.2f} {g_r_color}{g_r_str}{COLOR_RESET} {move_color}{move_pct_str}{COLOR_RESET} | {reason_color}{reason}{COLOR_RESET}")
        if grave_total > len(grave_shown):
            print(f"{COLOR_DIM}   ... {grave_total - len(grave_shown)} more graveyard row(s) not shown (--graveyard-max 0 prints them all).{COLOR_RESET}")

    _print_error_summary()
    print(f"\n⏱️ Tracker sync completed in {(time.time() - t_start):.2f} seconds ({sum(l.total_calls for l in LIMITERS.values())} API calls).\n")
    return 0

def parse_args():
    p = argparse.ArgumentParser(description="Strict institutional volatility tracker (Upstox) - System 3.0")
    p.add_argument("--mode", choices=["STOCK_FNO", "CASH_EQUITY", "INDEX_OPTIONS"], default=TRADING_MODE)
    p.add_argument("--days", type=int, default=BACKTRACE_DAYS)
    p.add_argument("--date", type=str, default=None)
    p.add_argument("--time", type=str, default=None)
    p.add_argument("--history-days", type=int, default=MIN1_HISTORY_DAYS)
    p.add_argument("--opt-min-price", type=float, default=OPT_MIN_PRICE)
    p.add_argument("--opt-min-volume", type=float, default=OPT_MIN_VOLUME)
    p.add_argument("--expiry", choices=["CURRENT", "NEXT"], default=EXPIRY_SELECTION)
    p.add_argument("--strikes-up", type=int, default=STRIKES_ABOVE_ATM)
    p.add_argument("--strikes-down", type=int, default=STRIKES_BELOW_ATM)
    p.add_argument("--checkpoint-min", type=int, default=CHECKPOINT_INTERVAL_MIN)
    p.add_argument("--single-snapshot", action="store_true")
    p.add_argument("--isolate-session-bands", action="store_true", default=ISOLATE_SESSION_BANDS)

    # --- Indicator Periods & Band Multipliers ---
    p.add_argument("--price-bb-period", type=int, default=PRICE_BB_PERIOD)
    p.add_argument("--price-bb-std", type=float, default=PRICE_BB_STD_MULT)
    p.add_argument("--price-kc-period", type=int, default=PRICE_KC_PERIOD)
    p.add_argument("--price-kc-atr", type=float, default=PRICE_KC_ATR_MULT)
    p.add_argument("--rsi-period", type=int, default=RSI_PERIOD)
    p.add_argument("--macd-fast", type=int, default=MACD_FAST_PERIOD)
    p.add_argument("--macd-slow", type=int, default=MACD_SLOW_PERIOD)
    p.add_argument("--macd-signal", type=int, default=MACD_SIGNAL_PERIOD)
    p.add_argument("--adx-period", type=int, default=ADX_PERIOD)
    p.add_argument("--ind-bb-period", type=int, default=IND_BB_PERIOD)
    p.add_argument("--vel-atr-period", type=int, default=VEL_ATR_PERIOD)

    # --- 3-Lock Tripwires & Quorums ---
    p.add_argument("--shock-mult", type=float, default=KINETIC_SHOCK_ATR_MULT)
    p.add_argument("--ind-bb-std", type=float, default=IND_BB_PIERCE_STD)
    p.add_argument("--struct-quorum", type=int, default=STRUCT_QUORUM)
    p.add_argument("--shock-quorum", type=int, default=SHOCK_QUORUM)
    p.add_argument("--shock-window", type=int, default=SHOCK_WINDOW_BARS)

    # --- Risk, Ratchet & Stop Floors ---
    p.add_argument("--min-atr-pct", type=float, default=MIN_ATR_PCT)
    p.add_argument("--hard-stop-atr", type=float, default=HARD_STOP_ATR_MULT)
    p.add_argument("--soft-stop-atr", type=float, default=SOFT_STOP_ATR_MULT)
    p.add_argument("--soft-stop-score", type=int, default=SOFT_STOP_OPPOSING_SCORE)
    p.add_argument("--soft-disable-r", type=float, default=SOFT_STOP_DISABLE_R)
    p.add_argument("--profit-lock-trig-r", type=float, default=PROFIT_LOCK_TRIGGER_R)
    p.add_argument("--profit-lock-floor-r", type=float, default=PROFIT_LOCK_FLOOR_R)
    p.add_argument("--adverse-drift-r", type=float, default=ADVERSE_DRIFT_R_FLOOR)

    p.add_argument("--day-pct-basis", choices=["PREV_CLOSE", "OPEN"], default=DAY_PCT_BASIS)
    p.add_argument("--graveyard-max", type=int, default=TOP_N_GRAVEYARD)
    p.add_argument("--track", type=str, default="")
    return p.parse_args()

if __name__ == "__main__":
    if not os.environ.get("UPSTOX_ACCESS_TOKEN"):
        print(f"{COLOR_RED_FG}[!] Missing UPSTOX_ACCESS_TOKEN environment variable.{COLOR_RESET}")
        sys.exit(1)
    args = parse_args()

    MIN1_HISTORY_DAYS = args.history_days
    OPT_MIN_PRICE = args.opt_min_price
    OPT_MIN_VOLUME = args.opt_min_volume
    EXPIRY_SELECTION = args.expiry
    STRIKES_ABOVE_ATM = args.strikes_up
    STRIKES_BELOW_ATM = args.strikes_down
    CHECKPOINT_INTERVAL_MIN = args.checkpoint_min
    ISOLATE_SESSION_BANDS = args.isolate_session_bands

    PRICE_BB_PERIOD = args.price_bb_period
    PRICE_BB_STD_MULT = args.price_bb_std
    PRICE_KC_PERIOD = args.price_kc_period
    PRICE_KC_ATR_MULT = args.price_kc_atr
    RSI_PERIOD = args.rsi_period
    MACD_FAST_PERIOD = args.macd_fast
    MACD_SLOW_PERIOD = args.macd_slow
    MACD_SIGNAL_PERIOD = args.macd_signal
    ADX_PERIOD = args.adx_period
    IND_BB_PERIOD = args.ind_bb_period
    VEL_ATR_PERIOD = args.vel_atr_period

    KINETIC_SHOCK_ATR_MULT = args.shock_mult
    IND_BB_PIERCE_STD = args.ind_bb_std
    STRUCT_QUORUM = args.struct_quorum
    SHOCK_QUORUM = args.shock_quorum
    SHOCK_WINDOW_BARS = args.shock_window

    MIN_ATR_PCT = args.min_atr_pct
    HARD_STOP_ATR_MULT = args.hard_stop_atr
    SOFT_STOP_ATR_MULT = args.soft_stop_atr
    SOFT_STOP_OPPOSING_SCORE = args.soft_stop_score
    SOFT_STOP_DISABLE_R = args.soft_disable_r
    PROFIT_LOCK_TRIGGER_R = args.profit_lock_trig_r
    PROFIT_LOCK_FLOOR_R = args.profit_lock_floor_r
    ADVERSE_DRIFT_R_FLOOR = args.adverse_drift_r

    DAY_PCT_BASIS = args.day_pct_basis
    TOP_N_GRAVEYARD = args.graveyard_max
    if args.single_snapshot:
        BACKTRACE_CHECKPOINTS = False

    try:
        code = run_screener(args.mode, args.days, args.date, args.time, args.track)
    except Exception:
        print(f"\n{COLOR_RED_FG}💥 FATAL: unhandled exception -- full traceback follows.{COLOR_RESET}")
        traceback.print_exc()
        code = 1
    sys.exit(code)
