#!/usr/bin/env python3
"""
Strict Institutional Volatility Tracker (Upstox) - ALL-MODE EDITION  (System3.py)
+ Modes: STOCK_FNO, CASH_EQUITY, INDEX_OPTIONS (NIFTY / BANKNIFTY / FINNIFTY / SENSEX option chains).
+ Intraday Backtrace: re-evaluates every 15-min checkpoint from session open to the run time.
  Displays the 'Seen' time of the EARLIEST checkpoint where a symbol genuinely qualified as
  Bull/Bear, but refreshes LTP, Move%, Day%, State and Indicators to the LATEST available
  checkpoint, so a symbol that fired early and is still alive (or was stopped out since) is
  never silently dropped just because the very last checkpoint looks different.
+ Configurable Expiry & Strike Window: EXPIRY_SELECTION (CURRENT/NEXT) and separate STRIKES_ABOVE_ATM/STRIKES_BELOW_ATM.
+ Time-Machine Targeting: --date / --time truncate all data at that snapshot.
+ Directional ATR Tripwires + Trailing Stop-Loss Floor, evaluated on a continuous 1-min indicator engine.
+ NEVER EXITS SILENTLY: every early-exit prints its reason, and real failures return a non-zero exit code.
+ Structured Graveyard with Anchor / Killed / Move% forensics.
+ OrderPlaced tracking: after you place an order off a run's results (a CE/PE strike, an F&O
  stock, or a cash-equity symbol), put that value in the OrderPlaced environment variable for
  subsequent runs. When OrderPlaced is empty, nothing changes -- the scan just runs and prints
  its results as before. When OrderPlaced is set, this run also looks that value up among its
  own results (same case-insensitive substring match as --watch) and emails its current status
  (live/stopped/rejected, LTP, Day%, Move% off anchor) so a scheduled run keeps you posted on a
  position you already took. See check_order_placed() below for the exact behaviour.
"""
import os
import sys
import re
import argparse
import traceback
import urllib.parse
import json
import gzip
import time
import threading
from bisect import bisect_left
from datetime import datetime, timedelta, timezone
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
# 0. ENGINE CONSTANTS & CONFIGURATION
# ==============================================================================
TRADING_MODE = "STOCK_FNO"

# --- INDEX OPTIONS ---
EXPIRY_SELECTION = "CURRENT"
STRIKES_ABOVE_ATM = 5
STRIKES_BELOW_ATM = 5
OPT_MIN_PRICE = 10
OPT_MIN_VOLUME = 5000
INDEX_CONFIG = {
    "NIFTY":     {"spot_key": "NSE_INDEX|Nifty 50",          "aliases": {"NIFTY", "NIFTY 50"}},
    "BANKNIFTY": {"spot_key": "NSE_INDEX|Nifty Bank",        "aliases": {"BANKNIFTY", "NIFTY BANK"}},
    "FINNIFTY":  {"spot_key": "NSE_INDEX|Nifty Fin Service", "aliases": {"FINNIFTY", "NIFTY FIN SERVICE"}},
    "SENSEX":    {"spot_key": "BSE_INDEX|SENSEX",            "aliases": {"SENSEX", "BSE SENSEX", "BSESN"}},
}
ALIAS_TO_INDEX = {a: idx for idx, cfg in INDEX_CONFIG.items() for a in cfg["aliases"]}

HA_ATR_MULTIPLIERS = [1, 2, 3, 5]
ATR_BASIS_PERIOD = 14
ATR_BASIS_TF = "60min"
MIN_ATR_PCT = 0.001

TOP_N_BUYERS = 50
TOP_N_SELLERS = 50
MAX_SYMBOL_WIDTH = 30

# --- INTRADAY BACKTRACE CHECKPOINTS ---
CHECKPOINT_INTERVAL_MIN = 15
BACKTRACE_CHECKPOINTS = True

REQUIRE_BB_KC_PIERCE = False

# --- OPTIONAL WATCH / EMAIL ALERT ---
# If WATCH_VALUE is non-empty (set via --watch), the run looks up that symbol/
# strike in this run's results and, ONLY if its status changed since the last
# run (tracked in WATCH_STATE_FILE), emails an alert using whichever of these
# environment variables (e.g. GitHub Actions repo secrets) are set:
#   SENDER_EMAIL / SENDER_PASSWORD / RECIPIENT_EMAIL   (original names), or
#   EMAIL_SENDER / EMAIL_APP_PWD   / EMAIL_RECEIVER     (alternate names)
# SMTP_SERVER / SMTP_PORT are optional either way -- they default to Gmail's
# smtp.gmail.com:465 if not provided. If WATCH_VALUE is left empty, none of
# this code runs at all -- no email, no state file, nothing.
WATCH_VALUE = ""
WATCH_STATE_FILE = "system3_watch_state.json"

# --- ORDER-PLACED TRACKING (env-driven, separate from --watch) ---
# Name of the environment variable this run reads to decide whether to track
# a placed order. Same idea as WATCH_VALUE/--watch, but sourced purely from
# the environment (so a scheduled run can be pointed at a live position
# without touching the command line) and, by default, emails EVERY run it
# finds a match on rather than only on a status change -- see
# check_order_placed() for the exact rules and ORDER_EMAIL_ONLY_ON_CHANGE
# below to switch it to change-only emails instead.
ORDER_PLACED_ENV = "OrderPlaced"
ORDER_EMAIL_ONLY_ON_CHANGE = False
ORDER_STATE_FILE = "system3_order_state.json"

# --- COLOR PALETTE ---
COLOR_RESET = '\033[0m'
COLOR_BOLD = '\033[1m'
COLOR_DIM = '\033[90m'
COLOR_CYAN = '\033[96;1m'
COLOR_GREEN_FG = '\033[92;1m'
COLOR_RED_FG = '\033[91;1m'
COLOR_YELLOW = '\033[93;1m'
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
        self.samples = []
        self.auth_failed = False
    def fail(self, msg):
        with self.lock:
            self.failed += 1
            if len(self.samples) < 5: self.samples.append(msg)
    def mark_auth_failed(self):
        with self.lock: self.auth_failed = True

class ErrorLog:
    def __init__(self):
        self.lock = threading.Lock()
        self.items = []
    def add(self, where, exc):
        with self.lock:
            self.items.append((where, f"{type(exc).__name__}: {exc}", traceback.format_exc()))

STATS = FetchStats()
ERRORS = ErrorLog()
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
    short = url.split("/v2/")[-1][:90]
    last = "no attempt made"
    for attempt in range(retries):
        limiter.acquire()
        try:
            r = _session().get(url, headers=headers, params=params, timeout=20)
            code = r.status_code
            if code == 200: return 200, r.json()
            if code == 401:
                STATS.mark_auth_failed()
                return 401, None
            last = f"HTTP {code}"
            if code in (400, 404):
                STATS.fail(f"{last} for {short}")
                return code, None
            time.sleep(2.0 * (attempt + 1) if code == 429 else 0.5 * (attempt + 1))
            continue
        except Exception as e:
            last = f"{type(e).__name__}: {e}"
        time.sleep(0.5 * (attempt + 1))
    STATS.fail(f"{last} for {short}")
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
    if status != 200 or not js: return status, None
    try:
        return 200, _to_frame((js.get('data') or {}).get('candles') or [])
    except Exception as e:
        STATS.fail(f"candle parse error ({type(e).__name__}: {e}) for {url.split('/v2/')[-1][:60]}")
        return 200, None

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

def generate_checkpoints(cutoff_dt, interval_min=None):
    """
    Build the intraday checkpoints to backtrace: the FIRST interval boundary
    AFTER session open, then every interval after that, up through cutoff_dt.
    Session open itself (9:15) is NEVER used as a checkpoint -- at 9:15 zero
    minutes of the day have elapsed, so that checkpoint would always be a dead
    read. For a 15-min interval the first checkpoint is therefore 9:30 (the
    9:15->9:30 window), then 9:45, 10:00, ... matching "backtrace 9:15 to
    9:30, 9:30 to 9:45, ...". If cutoff_dt isn't itself on a boundary, one
    final partial checkpoint is added AT cutoff_dt (a 10:35 run adds 10:35
    after ...,10:15,10:30).
    """
    interval = interval_min or CHECKPOINT_INTERVAL_MIN
    session_open = cutoff_dt.replace(hour=SESSION_OPEN_MIN // 60, minute=SESSION_OPEN_MIN % 60,
                                      second=0, microsecond=0)
    if cutoff_dt <= session_open:
        return [cutoff_dt]

    checkpoints = []
    t = session_open + timedelta(minutes=interval)   # first REAL boundary after open, e.g. 9:30 -- not 9:15 itself
    while t <= cutoff_dt:
        checkpoints.append(t)
        t += timedelta(minutes=interval)
    if not checkpoints or checkpoints[-1] < cutoff_dt:
        checkpoints.append(cutoff_dt)
    return checkpoints

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

def _unwrap_master(data):
    if isinstance(data, list): return data
    if isinstance(data, dict):
        if isinstance(data.get("data"), list): return data["data"]
        for v in data.values():
            if isinstance(v, list): return v
    return []

def _download_master(name, _is_fallback=False):
    url = f"https://assets.upstox.com/market-quote/instruments/exchange/{name}.json.gz"
    last_reason, attempts_made = "unknown", 0
    for attempt in range(1, 4):
        attempts_made = attempt
        try:
            resp = requests.get(url, timeout=90)
            if resp.status_code == 200:
                try:
                    rows = _unwrap_master(json.loads(gzip.decompress(resp.content).decode('utf-8')))
                except Exception as e:
                    last_reason = f"downloaded OK but could not gunzip/parse JSON: {e}"
                else:
                    if rows: return rows
                    last_reason = "downloaded and parsed OK but the file contained 0 rows"
            else:
                last_reason = f"HTTP {resp.status_code} ({resp.reason})"
                if resp.status_code == 404:
                    print(f"   {COLOR_YELLOW}[master:{name}] {last_reason} -- {url}{COLOR_RESET}", file=sys.stderr)
                    break
        except requests.RequestException as e:
            last_reason = f"network error: {e}"
        except Exception as e:
            last_reason = f"unexpected error: {type(e).__name__}: {e}"
        print(f"   {COLOR_YELLOW}[master:{name}] attempt {attempt}/3 failed: {last_reason}{COLOR_RESET}", file=sys.stderr)
        time.sleep(1.0)

    print(f"   {COLOR_RED_FG}[master:{name}] FAILED after {attempts_made} attempt(s): {last_reason}{COLOR_RESET}", file=sys.stderr)

    if not _is_fallback and name.upper() in ("NSE", "BSE"):
        print(f"   {COLOR_DIM}[master:{name}] trying combined 'complete' master as fallback...{COLOR_RESET}", file=sys.stderr)
        full = _download_master("complete", _is_fallback=True)
        rows = [r for r in full if str(r.get("exchange", "")).upper() == name.upper()
                or str(r.get("segment", "")).upper().startswith(name.upper())]
        if rows:
            print(f"   {COLOR_DIM}[master:{name}] fallback recovered {len(rows)} rows.{COLOR_RESET}", file=sys.stderr)
            return rows
    return []

def _equity_universe(mode):
    nse = _download_master("NSE")
    if not nse:
        print(f"{COLOR_RED_FG}[!] NSE instrument master unavailable -- cannot build the {mode} universe.{COLOR_RESET}")
        return []
    def ts_of(i): return i.get("tradingsymbol", i.get("trading_symbol"))
    def plain(i): return (i.get("segment") == "NSE_EQ" and ts_of(i) and i.get("instrument_key") and (INCLUDE_NON_EQ_SERIES or (i.get("instrument_type") or "EQ") == "EQ"))
    fno = {i.get("underlying_symbol") for i in nse if i.get("segment") == "NSE_FO" and i.get("underlying_symbol")}
    rows = [i for i in nse if plain(i) and (ts_of(i) in fno if mode == "STOCK_FNO" else ts_of(i) not in fno)]
    return list({i["instrument_key"]: {"symbol": ts_of(i), "key": i["instrument_key"]} for i in rows}.values())
# ------------------------------------------------------------------------------
# INDEX OPTIONS universe
# ------------------------------------------------------------------------------
def _resolve_index_spot_price(key, name, cutoff_dt, is_live):
    target_dt = cutoff_dt.date()
    if is_live:
        q = fetch_quotes([{"key": key, "symbol": name}], batch=1)
        ltp = q.get(key, {}).get('ltp', 0.0)
        if ltp and ltp > 0: return float(ltp), "live quote"
        df = fetch_today(key)
        if df is not None and not df.empty: return float(df['Close'].iloc[-1]), "today's 1-min candles"

    status, df = _candles(_url_range_1m(key, target_dt - timedelta(days=5), target_dt + timedelta(days=1)))
    if status == 200 and df is not None and not df.empty:
        sub = df[df['Datetime'] <= cutoff_dt]
        if not sub.empty: return float(sub['Close'].iloc[-1]), "1-min close at snapshot"

    status, df = _candles(_url_range_daily(key, target_dt - timedelta(days=10), target_dt + timedelta(days=1)))
    if status == 200 and df is not None and not df.empty:
        sub = df[df['Datetime'].dt.date <= target_dt]
        if not sub.empty: return float(sub['Close'].iloc[-1]), "daily close (fallback)"
    if STATS.auth_failed:
        return 0.0, "Upstox rejected the access token (HTTP 401)"
    return 0.0, f"no candles returned (last API failure: {STATS.samples[-1] if STATS.samples else 'none recorded'})"

def _field(row, *names):
    for n in names:
        v = row.get(n)
        if v is None or v == "" or (isinstance(v, float) and v != v): continue
        return v
    return None

def _expiry_to_date(exp):
    try:
        if isinstance(exp, (int, float, np.integer, np.floating)) or (isinstance(exp, str) and exp.strip().isdigit()):
            v = float(exp)
            if v > 1e11: v /= 1000.0
            return datetime.fromtimestamp(v, IST).date()
        s = str(exp).strip().split('T')[0].split(' ')[0]
        for fmt in ("%Y-%m-%d", "%d-%b-%Y", "%d-%m-%Y", "%d/%m/%Y"):
            try: return datetime.strptime(s, fmt).date()
            except ValueError: pass
    except Exception:
        pass
    return None

def _load_local_master(path):
    try:
        if path.lower().endswith('.csv'):
            return pd.read_csv(path).to_dict(orient='records')
        with open(path, 'r') as f:
            return _unwrap_master(json.load(f))
    except Exception as e:
        print(f"   {COLOR_RED_FG}[!] Could not read local options master '{path}': {type(e).__name__}: {e}{COLOR_RESET}")
        return []

def _index_options_universe(cutoff_dt, is_live, options_master_path=""):
    target_dt = cutoff_dt.date()

    spot = {}
    for idx, cfg in INDEX_CONFIG.items():
        try:
            price, src = _resolve_index_spot_price(cfg["spot_key"], idx, cutoff_dt, is_live)
        except Exception as e:
            ERRORS.add(f"spot:{idx}", e)
            price, src = 0.0, f"{type(e).__name__}: {e}"
        if price > 0:
            spot[idx] = price
            print(f"   {COLOR_DIM}» {idx:<10} spot {price:>10.2f}  ({src}){COLOR_RESET}")
        else:
            print(f"   {COLOR_YELLOW}» {idx:<10} spot unavailable -- skipped ({src}){COLOR_RESET}")
    if not spot:
        if not STATS.auth_failed:
            print(f"{COLOR_RED_FG}[!] Could not resolve a spot price for ANY index, so no ATM strikes can be chosen. "
                  f"Check the token/API access to index candles (see failure samples below).{COLOR_RESET}")
        return []

    rows = []
    if options_master_path:
        if os.path.exists(options_master_path):
            print(f"   {COLOR_DIM}» Using local options master: {options_master_path}{COLOR_RESET}")
            rows = _load_local_master(options_master_path)
        else:
            print(f"   {COLOR_YELLOW}» --options-master '{options_master_path}' not found; using live masters instead.{COLOR_RESET}")
    if not rows:
        nse, bse = _download_master("NSE"), _download_master("BSE")
        print(f"   {COLOR_DIM}» Master rows fetched -- NSE: {len(nse)}, BSE: {len(bse)}{COLOR_RESET}")
        rows = nse + bse
    if not rows:
        print(f"{COLOR_RED_FG}[!] Both NSE and BSE instrument masters came back empty -- see the [master:...] lines above "
              f"for the exact HTTP status / network error.{COLOR_RESET}")
        return []

    by_idx = {idx: [] for idx in INDEX_CONFIG}
    for row in rows:
        segment = str(_field(row, "segment") or "").upper()
        if segment and segment not in ("NSE_FO", "BSE_FO"): continue
        ts = str(_field(row, "trading_symbol", "tradingsymbol") or "").upper()

        itype = str(_field(row, "option_type", "instrument_type") or "").upper()
        if itype not in ("CE", "PE"):
            m = re.search(r'\b(CE|PE)\b', ts) or re.search(r'\d(CE|PE)$', ts)
            if not m: continue
            itype = m.group(1)

        cands = [str(_field(row, "underlying_symbol") or ""), str(_field(row, "name") or ""), ts.split()[0] if ts else ""]
        idx = next((ALIAS_TO_INDEX[c.upper().strip()] for c in cands if c.upper().strip() in ALIAS_TO_INDEX), None)
        if idx is None: continue

        exp_date = _expiry_to_date(_field(row, "expiry"))
        if exp_date is None: continue

        strike_raw = _field(row, "strike_price", "strike")
        try:
            strike = float(strike_raw) if strike_raw is not None else 0.0
        except (TypeError, ValueError):
            strike = 0.0
        if strike <= 0:
            m = re.search(r'(\d+(?:\.\d+)?)\s+(?:CE|PE)\b', ts) or re.search(r'(\d+)(?:CE|PE)$', ts)
            strike = float(m.group(1)) if m else 0.0
        if strike <= 0 or not _field(row, "instrument_key"): continue

        by_idx[idx].append((exp_date, strike, itype, row))

    total_opts = sum(len(v) for v in by_idx.values())
    if total_opts == 0:
        sample = sorted(rows[0].keys()) if rows and isinstance(rows[0], dict) else "n/a"
        print(f"{COLOR_RED_FG}[!] Master downloaded ({len(rows)} rows) but 0 CE/PE contracts matched "
              f"NIFTY/BANKNIFTY/FINNIFTY/SENSEX. Field names in this master: {sample}{COLOR_RESET}")
        return []

    universe, seen = [], set()
    for idx, contracts in by_idx.items():
        if idx not in spot: continue
        if not contracts:
            print(f"   {COLOR_YELLOW}» {idx:<10} no option contracts found in master -- skipped{COLOR_RESET}")
            continue
        expiries = sorted({c[0] for c in contracts if c[0] >= target_dt})
        if not expiries:
            latest = max(c[0] for c in contracts)
            print(f"   {COLOR_YELLOW}» {idx:<10} no expiry on/after {target_dt} in this master (latest is {latest}). "
                  f"Expired contracts need --options-master.{COLOR_RESET}")
            continue

        expiry_idx = 1 if str(EXPIRY_SELECTION).upper() == "NEXT" else 0
        if expiry_idx >= len(expiries):
            print(f"   {COLOR_YELLOW}  ⚠ {idx:<10} EXPIRY_SELECTION={EXPIRY_SELECTION!r} wants expiry #{expiry_idx + 1} "
                  f"on/after {target_dt}, but the master only has {len(expiries)}. Using the furthest one available "
                  f"instead of failing outright.{COLOR_RESET}")
        target_exp = expiries[min(expiry_idx, len(expiries) - 1)]
        chain = [c for c in contracts if c[0] == target_exp]
        strikes = np.array(sorted({c[1] for c in chain}))
        atm_i = int(np.abs(strikes - spot[idx]).argmin())

        lo = max(0, atm_i - STRIKES_BELOW_ATM)
        hi = min(len(strikes), atm_i + STRIKES_ABOVE_ATM + 1)
        window = set(strikes[lo:hi].tolist())
        picked = 0
        for exp_date, strike, itype, row in chain:
            key = _field(row, "instrument_key")
            if strike in window and key not in seen:
                seen.add(key)
                universe.append({"key": key, "symbol": str(_field(row, "trading_symbol", "tradingsymbol") or key)})
                picked += 1
        gap = (target_exp - target_dt).days
        print(f"   {COLOR_DIM}» {idx:<10} expiry {target_exp} ({EXPIRY_SELECTION})  "
              f"strikes {strikes[lo]:.0f}-{strikes[hi-1]:.0f} ({STRIKES_BELOW_ATM}↓/{STRIKES_ABOVE_ATM}↑ of ATM)  "
              f"-> {picked} contracts{COLOR_RESET}")
        if not is_live and expiry_idx == 0 and gap > 10:
            print(f"   {COLOR_YELLOW}  ⚠ nearest expiry in today's master is {gap} days after {target_dt}; the true nearest "
                  f"expiry on that date may already have expired. Use --options-master for exact backtests.{COLOR_RESET}")
    return universe

def get_dynamic_universe(mode, cutoff_dt, is_live, options_master_path=""):
    if mode in ("STOCK_FNO", "CASH_EQUITY"):
        return _equity_universe(mode)
    if mode == "INDEX_OPTIONS":
        return _index_options_universe(cutoff_dt, is_live, options_master_path)
    print(f"{COLOR_RED_FG}[!] Unknown mode '{mode}'.{COLOR_RESET}")
    return []

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
    target = base_atr * mult
    for i in range(today_start, len(close)):
        if close[i] - curr_open >= target or curr_open - close[i] >= target:
            _, _, _, _, b_rsi, br_rsi, b_macd, br_macd, b_di, br_di = get_kinetics(kin_1m, i)
            last_rsi = "Buy" if b_rsi else "Sell" if br_rsi else "Neutral"
            last_macd = "Buy" if b_macd else "Sell" if br_macd else "Neutral"
            last_adx = "Buy" if b_di else "Sell" if br_di else "Neutral"
            blocks += 1
            curr_open = close[i]
    return last_rsi, last_macd, last_adx, blocks

def evaluate_anchor_tripwire(close, dt_1m, kin_1m, base_atr, bb_upper, bb_lower, kc_upper, kc_lower, today_start):
    if today_start >= len(close):
        return None, {'dir': 'NONE', 'anchor_time': '-', 'anchor_price': 0.0,
                      'killed_time': '-', 'killed_price': 0.0, 'reason': 'No data for today'}

    i = today_start
    curr_open = close[today_start]
    survived_anchor = None
    last_killed_info = {'dir': 'NONE', 'anchor_time': '-', 'anchor_price': 0.0,
                        'killed_time': '-', 'killed_price': 0.0, 'reason': 'Failed Kinetic Alignment'}
    warzone_kills = 0

    while i < len(close):
        anchor = None

        while i < len(close):
            bull_score, bear_score, r_bull, r_bear, _, _, _, _, _, _ = get_kinetics(kin_1m, i)

            if REQUIRE_BB_KC_PIERCE:
                bb_kc_bull_fire = bb_upper[i] > kc_upper[i]
                bb_kc_bear_fire = bb_lower[i] < kc_lower[i]
            else:
                bb_kc_bull_fire = bb_kc_bear_fire = True

            if close[i] - curr_open >= base_atr:
                if bull_score >= 2 and bb_kc_bull_fire:
                    if r_bear:
                        warzone_kills += 1
                        curr_open = close[i]
                    else:
                        anchor = {"dir": "BULL", "idx": i, "time": dt_1m[i], "price": close[i]}
                        break
                else:
                    curr_open = close[i]

            elif curr_open - close[i] >= base_atr:
                if bear_score >= 2 and bb_kc_bear_fire:
                    if r_bull:
                        warzone_kills += 1
                        curr_open = close[i]
                    else:
                        anchor = {"dir": "BEAR", "idx": i, "time": dt_1m[i], "price": close[i]}
                        break
                else:
                    curr_open = close[i]
            i += 1

        if not anchor:
            break

        survived = True
        peak_price = anchor['price']
        j = anchor['idx'] + 1

        while j < len(close):
            bull_score, bear_score, _, _, _, _, _, _, _, _ = get_kinetics(kin_1m, j)

            if anchor['dir'] == "BULL":
                peak_price = max(peak_price, close[j])
                if peak_price - close[j] >= base_atr:
                    if bear_score >= 2:
                        last_killed_info = {
                            'dir': 'BULL',
                            'anchor_time': pd.to_datetime(anchor['time']).strftime('%H:%M'),
                            'anchor_price': float(anchor['price']),
                            'killed_time': pd.to_datetime(dt_1m[j]).strftime('%H:%M'),
                            'killed_price': float(close[j]),
                            'reason': "Kinetic SL (1 ATR Pullback)"
                        }
                        survived = False
                        curr_open = close[j]
                        i = j + 1
                        break
                    else:
                        peak_price = close[j]

            elif anchor['dir'] == "BEAR":
                peak_price = min(peak_price, close[j])
                if close[j] - peak_price >= base_atr:
                    if bull_score >= 2:
                        last_killed_info = {
                            'dir': 'BEAR',
                            'anchor_time': pd.to_datetime(anchor['time']).strftime('%H:%M'),
                            'anchor_price': float(anchor['price']),
                            'killed_time': pd.to_datetime(dt_1m[j]).strftime('%H:%M'),
                            'killed_price': float(close[j]),
                            'reason': "Kinetic SL (1 ATR Rally)"
                        }
                        survived = False
                        curr_open = close[j]
                        i = j + 1
                        break
                    else:
                        peak_price = close[j]
            j += 1

        if survived:
            survived_anchor = anchor
            break

    if survived_anchor:
        return survived_anchor, {"status": "Survived"}

    if warzone_kills > 0 and last_killed_info['dir'] == 'NONE':
        last_killed_info['reason'] = f"Warzone Inversion Chop ({warzone_kills}x)"

    return None, last_killed_info

def compute_row(symbol, master_1m, target_dt):
    close, high, low, dt = master_1m['Close'].values, master_1m['High'].values, master_1m['Low'].values, master_1m['Datetime'].values

    today_mask = master_1m['Datetime'].dt.date == target_dt
    if not today_mask.any():
        return None

    today_start_idx = int(master_1m.index[today_mask][0])
    day_open_price = master_1m['Open'].iloc[today_start_idx]
    intraday_pct = ((close[-1] - day_open_price) / day_open_price) * 100 if day_open_price else 0.0

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

    row = {
        'Symbol': symbol, 'LTP': float(close[-1]), 'DayChangePct': float(intraday_pct),
        'ActiveAnchor': None, 'AnchorPrice': 0.0, 'AnchorDir': "NONE", 'MovePct': 0.0,
        'State': "NONE", 'Blocks': 0, 'KilledTime': '-', 'KilledPrice': 0.0, 'RejectReason': ""
    }

    anchor, reject_info = evaluate_anchor_tripwire(close, dt, kin_1m, base_atr, bb_upper, bb_lower, kc_upper, kc_lower, today_start_idx)

    if anchor:
        row['ActiveAnchor'] = pd.to_datetime(anchor['time']).strftime("%H:%M")
        row['AnchorPrice'] = float(anchor['price'])
        row['AnchorDir'] = anchor['dir']
        row['MovePct'] = ((float(close[-1]) - anchor['price']) / anchor['price']) * 100 if anchor['price'] else 0.0

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
    else:
        row['AnchorDir'] = reject_info.get('dir', 'NONE')
        row['ActiveAnchor'] = reject_info.get('anchor_time', '-')
        row['AnchorPrice'] = float(reject_info.get('anchor_price', 0.0))
        row['KilledTime'] = reject_info.get('killed_time', '-')
        row['KilledPrice'] = float(reject_info.get('killed_price', 0.0))
        row['RejectReason'] = reject_info.get('reason', 'Failed Kinetic Alignment')
        if row['AnchorPrice'] > 0 and row['KilledPrice'] > 0:
            row['MovePct'] = ((row['KilledPrice'] - row['AnchorPrice']) / row['AnchorPrice']) * 100

    return row
  # ==============================================================================
# 3. WORKERS, FORMATTERS & UI
# ==============================================================================
def format_kinetic_triad(rsi, macd, di):
    def fmt(sig):
        if sig == "Buy": return f"{COLOR_GREEN_FG}Buy{COLOR_RESET}"
        if sig == "Sell": return f"{COLOR_RED_FG}Sel{COLOR_RESET}"
        return f"{COLOR_DIM} - {COLOR_RESET}"
    return f"{fmt(rsi)} {fmt(macd)} {fmt(di)}"

def _history_worker_daily(args):
    item, start, end, progress = args
    try:
        frames = []
        for c_start, c_end in _date_chunks(start, end + timedelta(days=1), span=365):
            status, df = _candles(_url_range_daily(item['key'], c_start, c_end))
            if status == 200 and df is not None: frames.append(df)
        if not frames: return None
        master = prepare_master(frames)
        close, vol = master['Close'].iloc[-1], master['Volume'].iloc[-1]
        return item if (MIN_PRICE <= close <= MAX_PRICE and vol >= MIN_DAILY_VOLUME) else None
    except Exception as e:
        ERRORS.add(f"prefilter:{item.get('symbol')}", e)
        return None
    finally:
        progress.tick()

def process_stock_checkpoints(args):
    item, checkpoints, is_live, history_days, mode, progress = args
    try:
        final_cutoff = checkpoints[-1]
        target_dt = final_cutoff.date()
        start_dt = target_dt - timedelta(days=history_days)
        frames = []
        for c_start, c_end in _date_chunks(start_dt, target_dt + timedelta(days=1), span=7):
            status, df = _candles(_url_range_1m(item['key'], c_start, c_end))
            if status == 200 and df is not None: frames.append(df)

        if is_live:
            today_df = fetch_today(item['key'])
            if today_df is not None: frames.append(today_df)

        if not frames:
            return [(cp, "no_data", item['symbol']) for cp in checkpoints]

        master_full = prepare_master(frames)
        out = []
        for cp in checkpoints:
            master_1m = master_full[master_full['Datetime'] <= cp].reset_index(drop=True)
            if len(master_1m) < 30:
                out.append((cp, "no_data", item['symbol']))
                continue

            if mode == "INDEX_OPTIONS":
                day = master_1m[master_1m['Datetime'].dt.date == target_dt]
                if day.empty:
                    out.append((cp, "no_data", item['symbol']))
                    continue
                if day['Close'].iloc[-1] < OPT_MIN_PRICE or day['Volume'].sum() < OPT_MIN_VOLUME:
                    out.append((cp, "illiquid", item['symbol']))
                    continue

            row = compute_row(item['symbol'], master_1m, target_dt)
            if row is None:
                out.append((cp, "no_data", item['symbol']))
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
        print(f"{COLOR_YELLOW}⚠ {STATS.failed} API request(s) failed after retries. Samples: {'; '.join(STATS.samples)}{COLOR_RESET}")
    if ERRORS.items:
        print(f"{COLOR_YELLOW}⚠ {len(ERRORS.items)} instrument(s) raised exceptions. First one:{COLOR_RESET}")
        where, msg, tb = ERRORS.items[0]
        print(f"{COLOR_DIM}   [{where}] {msg}\n{tb}{COLOR_RESET}")

# ==============================================================================
# 3b. OPTIONAL WATCH / EMAIL ALERT (no-op unless --watch is passed)
#     + ORDER-PLACED TRACKING (no-op unless the OrderPlaced env var is set)
# ==============================================================================
def _load_watch_state(path):
    if not os.path.exists(path):
        return {}
    try:
        with open(path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"   {COLOR_YELLOW}[watch] could not read state file '{path}' ({type(e).__name__}: {e}) -- starting fresh.{COLOR_RESET}")
        return {}

def _save_watch_state(path, state):
    try:
        with open(path, 'w') as f:
            json.dump(state, f, indent=2, default=str)
    except Exception as e:
        print(f"   {COLOR_YELLOW}[watch] could not write state file '{path}' ({type(e).__name__}: {e}).{COLOR_RESET}")

def _send_watch_email(subject, body):
    # Accepts either naming convention, so this works unmodified whichever
    # secrets the workflow happens to define:
    #   SENDER_EMAIL / SENDER_PASSWORD / RECIPIENT_EMAIL   (original names), or
    #   EMAIL_SENDER / EMAIL_APP_PWD   / EMAIL_RECEIVER     (seen in the user's
    #   "Institutional Hybrid AI Sweep" workflow).
    # SMTP_SERVER/SMTP_PORT default to Gmail's SMTP-over-SSL endpoint when not
    # set -- that workflow defines no SMTP_SERVER/SMTP_PORT secrets at all, and
    # an "app password" (EMAIL_APP_PWD) is Gmail's term for that credential.
    sender = os.environ.get("SENDER_EMAIL") or os.environ.get("EMAIL_SENDER")
    password = os.environ.get("SENDER_PASSWORD") or os.environ.get("EMAIL_APP_PWD")
    recipient = os.environ.get("RECIPIENT_EMAIL") or os.environ.get("EMAIL_RECEIVER")
    smtp_server = os.environ.get("SMTP_SERVER") or "smtp.gmail.com"
    smtp_port = os.environ.get("SMTP_PORT") or "465"

    missing = [name for name, val in (
        ("SENDER_EMAIL/EMAIL_SENDER", sender),
        ("SENDER_PASSWORD/EMAIL_APP_PWD", password),
        ("RECIPIENT_EMAIL/EMAIL_RECEIVER", recipient),
    ) if not val]
    if missing:
        print(f"   {COLOR_YELLOW}[watch] cannot send email -- missing env var(s): {', '.join(missing)}.{COLOR_RESET}")
        return False

    try:
        port = int(smtp_port)
    except ValueError:
        print(f"   {COLOR_YELLOW}[watch] cannot send email -- SMTP_PORT='{smtp_port}' is not a valid integer.{COLOR_RESET}")
        return False

    msg = EmailMessage()
    msg['Subject'] = subject
    msg['From'] = sender
    msg['To'] = recipient
    msg.set_content(body)

    try:
        if port == 465:
            with smtplib.SMTP_SSL(smtp_server, port, timeout=20) as server:
                server.login(sender, password)
                server.send_message(msg)
        else:
            with smtplib.SMTP(smtp_server, port, timeout=20) as server:
                server.starttls()
                server.login(sender, password)
                server.send_message(msg)
        print(f"   {COLOR_GREEN_FG}[watch] alert email sent to {recipient}.{COLOR_RESET}")
        return True
    except Exception as e:
        print(f"   {COLOR_YELLOW}[watch] failed to send email ({type(e).__name__}: {e}).{COLOR_RESET}")
        return False

def _find_match(needle_upper, all_bulls, all_bears, rejected):
    """
    Shared lookup for --watch and OrderPlaced alike: case-insensitive substring
    match against 'Symbol', checked in order live-bull -> live-bear -> graveyard,
    so it works the same way whether the placed order was a CE/PE option strike
    (e.g. 'NIFTY25000CE'), an F&O stock, or a plain cash-equity symbol.
    """
    for row in all_bulls:
        if needle_upper in row['Symbol'].upper():
            return row
    for row in all_bears:
        if needle_upper in row['Symbol'].upper():
            return row
    for row in rejected:
        if needle_upper in row['Symbol'].upper():
            return row
    return None

def _describe_match(mode, value, match, cutoff_dt):
    """Returns (symbol, status, detail_text) for a matched row, or a NOT FOUND placeholder."""
    if match is not None:
        symbol = match['Symbol']
        status = match['State'] if match.get('State', 'NONE') != "NONE" else f"REJECTED ({match.get('RejectReason', 'unknown')})"
        seen = match.get('CheckpointTime', cutoff_dt)
        seen_str = seen.strftime('%H:%M') if hasattr(seen, 'strftime') else str(seen)
        detail = (f"Symbol: {symbol}\nStatus: {status}\nLTP: {match.get('LTP', 0):.2f}\n"
                  f"Day Change: {match.get('DayChangePct', 0):+.2f}%\nMove from anchor: {match.get('MovePct', 0):+.2f}%\n"
                  f"Seen: {seen_str}\nMode: {mode}\nSnapshot: {cutoff_dt:%Y-%m-%d %H:%M}")
    else:
        symbol = value
        status = "NOT FOUND"
        detail = f"'{value}' did not match any scanned symbol this run.\nMode: {mode}\nSnapshot: {cutoff_dt:%Y-%m-%d %H:%M}"
    return symbol, status, detail

def check_watch_alerts(mode, watch_value, cutoff_dt, all_bulls, all_bears, rejected):
    """
    Optional single-symbol/strike watch. WATCH_VALUE (set via --watch) is
    matched case-insensitively as a substring against each row's 'Symbol' --
    so a bare strike like '25000' or a full tradingsymbol like
    'NIFTY25000CE' both work. The matched row's status (ACTIVE BUY /
    ACTIVE SELL / COILING / STOPPED / a rejection reason / NOT FOUND) is
    compared against the last run's status for that same watch value,
    persisted in WATCH_STATE_FILE. An email is sent ONLY when the status
    changed (or on the very first run that sees this watch value at all).
    This entire function is only ever called when WATCH_VALUE is non-empty,
    so none of it runs -- no email, no state file read/write -- unless
    --watch was explicitly passed.
    """
    needle = str(watch_value).strip().upper()
    if not needle:
        return

    match = _find_match(needle, all_bulls, all_bears, rejected)
    symbol, status, detail = _describe_match(mode, watch_value, match, cutoff_dt)

    state = _load_watch_state(WATCH_STATE_FILE)
    key = needle
    prev_status = state.get(key, {}).get("status")

    print(f"\n{COLOR_BOLD}👁  WATCH [{watch_value}] -> {status}{COLOR_RESET}" +
          (f" {COLOR_DIM}(previous: {prev_status}){COLOR_RESET}" if prev_status else ""))

    if prev_status == status:
        print(f"   {COLOR_DIM}[watch] status unchanged since last run -- no email sent.{COLOR_RESET}")
    else:
        if prev_status is None:
            subject = f"[System3] Watch started: {watch_value} is {status}"
        else:
            subject = f"[System3] {watch_value} changed: {prev_status} -> {status}"
        body = f"{subject}\n\n{detail}"
        _send_watch_email(subject, body)

    state[key] = {"status": status, "checked_at": cutoff_dt.strftime("%Y-%m-%d %H:%M:%S")}
    _save_watch_state(WATCH_STATE_FILE, state)

def check_order_placed(mode, order_value, cutoff_dt, all_bulls, all_bears, rejected):
    """
    Env-driven order tracker. After placing an order off a previous run's
    results -- a CE/PE option strike, an F&O stock, or a plain cash-equity
    symbol -- put that exact value (or any substring unique enough to match
    only it, e.g. just the strike '25000CE') into the OrderPlaced
    environment variable before the next run.

      * OrderPlaced empty/unset -> this function is never called. The run
        just scans and prints its normal results, exactly as before.
      * OrderPlaced non-empty  -> this run looks that value up among its own
        results (same matching as --watch: live bulls, then live bears,
        then the graveyard) and emails its current status -- LTP, Day%,
        Move% off anchor, and whether it's [ACTIVE BUY]/[ACTIVE SELL]/
        [COILING]/[STOPPED], or its graveyard rejection reason if it never
        became a live setup, or NOT FOUND if nothing matched this run.

    By default (ORDER_EMAIL_ONLY_ON_CHANGE = False) an email goes out on
    EVERY run that has OrderPlaced set, so a position you're holding gets a
    fresh status email each scheduled run. Set ORDER_EMAIL_ONLY_ON_CHANGE =
    True to instead only email when the status changes since the last run
    (tracked separately in ORDER_STATE_FILE, so it never shares state with
    --watch).
    """
    needle = str(order_value).strip().upper()
    if not needle:
        return

    match = _find_match(needle, all_bulls, all_bears, rejected)
    symbol, status, detail = _describe_match(mode, order_value, match, cutoff_dt)

    prev_status = None
    should_send = True
    if ORDER_EMAIL_ONLY_ON_CHANGE:
        state = _load_watch_state(ORDER_STATE_FILE)
        prev_status = state.get(needle, {}).get("status")
        should_send = (prev_status != status)

    print(f"\n{COLOR_BOLD}📦 ORDER PLACED [{order_value}] -> {status}{COLOR_RESET}" +
          (f" {COLOR_DIM}(previous: {prev_status}){COLOR_RESET}" if prev_status else ""))

    if not should_send:
        print(f"   {COLOR_DIM}[order] status unchanged since last run -- no email sent.{COLOR_RESET}")
    else:
        if ORDER_EMAIL_ONLY_ON_CHANGE:
            subject = (f"[System3] Order tracking started: {order_value} is {status}" if prev_status is None
                       else f"[System3] {order_value} changed: {prev_status} -> {status}")
        else:
            subject = f"[System3] Order status: {order_value} -> {status}"
        body = f"{subject}\n\n{detail}"
        _send_watch_email(subject, body)

    if ORDER_EMAIL_ONLY_ON_CHANGE:
        state = _load_watch_state(ORDER_STATE_FILE)
        state[needle] = {"status": status, "checked_at": cutoff_dt.strftime("%Y-%m-%d %H:%M:%S")}
        _save_watch_state(ORDER_STATE_FILE, state)

def run_screener(mode=TRADING_MODE, days=BACKTRACE_DAYS, target_date_str=None, target_time_str="15:30", options_master_path=""):
    t_start = time.time()

    if target_date_str:
        try:
            cutoff_dt = datetime.strptime(f"{target_date_str} {target_time_str}", "%Y-%m-%d %H:%M")
        except ValueError:
            print(f"{COLOR_RED_FG}[!] Invalid --date/--time '{target_date_str} {target_time_str}' (expected YYYY-MM-DD and HH:MM).{COLOR_RESET}")
            return 1
        target_dt, is_live = cutoff_dt.date(), False
    else:
        cutoff_dt = now_ist()
        target_dt, is_live = cutoff_dt.date(), True

    market_close = datetime.combine(target_dt, datetime.min.time()) + timedelta(hours=15, minutes=30)
    cutoff_dt = min(cutoff_dt, market_close)

    print(f"\n{COLOR_CYAN}📡 Initializing Tracker [{mode}] | Time Machine: {cutoff_dt.strftime('%Y-%m-%d %H:%M:%S')} (Live: {is_live}){COLOR_RESET}")

    universe_raw = get_dynamic_universe(mode, cutoff_dt, is_live, options_master_path)
    if STATS.auth_failed:
        print(f"{COLOR_RED_FG}[!] Upstox rejected the access token (HTTP 401). Regenerate UPSTOX_ACCESS_TOKEN -- it expires daily.{COLOR_RESET}")
        return 1
    if not universe_raw:
        print(f"{COLOR_RED_FG}[!] Universe is EMPTY for mode {mode} -- nothing to scan. See the messages above for why.{COLOR_RESET}")
        _print_error_summary()
        return 1
    print(f"   {COLOR_DIM}» Universe: {len(universe_raw)} instruments{COLOR_RESET}")

    if mode == "INDEX_OPTIONS":
        candidates = universe_raw
    else:
        daily_start = target_dt - timedelta(days=days * 2 + 6)
        prog = Progress("prefilter", len(universe_raw))
        with concurrent.futures.ThreadPoolExecutor(max_workers=WORKERS) as ex:
            candidates = [r for r in ex.map(_history_worker_daily, [(it, daily_start, target_dt, prog) for it in universe_raw]) if r is not None]
        prog.done()
        if STATS.auth_failed:
            print(f"{COLOR_RED_FG}[!] Upstox rejected the access token (HTTP 401) during the prefilter.{COLOR_RESET}")
            return 1
        if not candidates:
            print(f"{COLOR_YELLOW}[!] 0 of {len(universe_raw)} instruments passed the price/volume prefilter "
                  f"(price {MIN_PRICE}-{MAX_PRICE}, volume >= {MIN_DAILY_VOLUME}).{COLOR_RESET}")
            _print_error_summary()
            return 1 if (STATS.failed or ERRORS.items) else 0
        print(f"   {COLOR_DIM}» Prefilter: {len(universe_raw)} -> {len(candidates)} instruments{COLOR_RESET}")

    checkpoints = generate_checkpoints(cutoff_dt) if BACKTRACE_CHECKPOINTS else [cutoff_dt]
    print(f"   {COLOR_DIM}» Backtrace checkpoints ({len(checkpoints)}): "
          f"{', '.join(c.strftime('%H:%M') for c in checkpoints)}{COLOR_RESET}")

    prog = Progress("checkpoints", len(candidates))
    with concurrent.futures.ThreadPoolExecutor(max_workers=WORKERS) as ex:
        per_symbol = list(ex.map(process_stock_checkpoints,
                                  [(item, checkpoints, is_live, MIN1_HISTORY_DAYS, mode, prog) for item in candidates]))
    prog.done()

    if STATS.auth_failed:
        print(f"{COLOR_RED_FG}[!] Upstox rejected the access token (HTTP 401) while fetching candles.{COLOR_RESET}")
        return 1

    outcomes = [o for sym_outcomes in per_symbol for o in sym_outcomes]
    results = [p for _, s, p in outcomes if s == "ok"]
    n_nodata = sum(1 for _, s, _ in outcomes if s == "no_data")
    n_illiq = sum(1 for _, s, _ in outcomes if s == "illiquid")
    n_err = sum(1 for _, s, _ in outcomes if s == "error")
    print(f"   {COLOR_DIM}» Scan summary: {len(candidates)} instruments x {len(checkpoints)} checkpoints = "
          f"{len(outcomes)} evaluations | ok {len(results)} | no data {n_nodata} | illiquid {n_illiq} | "
          f"errors {n_err}{COLOR_RESET}")

    if not results:
        print(f"{COLOR_RED_FG}[!] No instrument produced usable data at any checkpoint up to {cutoff_dt:%Y-%m-%d %H:%M}. "
              f"Is that a trading day/time, and does the token have historical-candle access?{COLOR_RESET}")
        _print_error_summary()
        return 1

    by_symbol = {}
    for row in results:
        by_symbol.setdefault(row['Symbol'], []).append(row)

    ACTIVE_BULL_STATES = ("[ACTIVE BUY]", "[COILING]")
    ACTIVE_BEAR_STATES = ("[ACTIVE SELL]", "[COILING]")

    def _merge(first_row, last_row, direction):
        """
        Build the display row for one direction (BULL/BEAR) of one symbol.
        Identity -- Seen time ('CheckpointTime'), Anchor time, Anchor price --
        comes from the FIRST checkpoint where this symbol genuinely qualified
        for `direction` (AnchorDir matches AND State was active). That is what
        "sort by which occurred earlier" ranks on. Every OTHER field (LTP,
        Day%, indicator triads, current State) is refreshed from the LATEST
        checkpoint evaluated, so the row reflects what's true NOW rather than
        what was true back when the anchor first appeared. If the anchor is
        no longer alive as of the latest checkpoint, State is overridden to
        [STOPPED] instead of silently keeping a stale ACTIVE/COILING tag.
        Note: because each checkpoint independently re-runs the anchor search
        over the data available up to that point, 'Killed' time/price (when
        shown) reflect whatever the LATEST checkpoint's search concluded --
        the architecture has no persistent handle on "this exact anchor" across
        checkpoints, only on "was this symbol genuinely bull/bear at time X".
        """
        row = dict(last_row)
        row['CheckpointTime'] = first_row['CheckpointTime']
        row['ActiveAnchor'] = first_row['ActiveAnchor']
        row['AnchorPrice'] = first_row['AnchorPrice']
        row['AnchorDir'] = direction
        anchor_price = first_row['AnchorPrice'] or last_row['LTP']
        row['MovePct'] = ((last_row['LTP'] - anchor_price) / anchor_price) * 100 if anchor_price else 0.0

        active_states = ACTIVE_BULL_STATES if direction == "BULL" else ACTIVE_BEAR_STATES
        still_alive = (last_row['AnchorDir'] == direction and last_row['State'] in active_states)
        if not still_alive:
            row['State'] = "[STOPPED]"
        return row

    bulls, bears, rejected = [], [], []
    for symbol, rows in by_symbol.items():
        rows.sort(key=lambda r: r['CheckpointTime'])    # chronological, earliest first
        last_row = rows[-1]                             # freshest available data for this symbol

        bull_rows = [r for r in rows if r['AnchorDir'] == "BULL" and r['State'] in ACTIVE_BULL_STATES]
        bear_rows = [r for r in rows if r['AnchorDir'] == "BEAR" and r['State'] in ACTIVE_BEAR_STATES]

        if bull_rows:
            bulls.append(_merge(bull_rows[0], last_row, "BULL"))
        if bear_rows:
            bears.append(_merge(bear_rows[0], last_row, "BEAR"))
        if not bull_rows and not bear_rows:
            row = dict(last_row)
            row['CheckpointTime'] = rows[0]['CheckpointTime']   # earliest time this symbol was ever evaluated
            rejected.append(row)

    true_bull_count = len(bulls)
    true_bear_count = len(bears)

    # Rank LIVE setups (ACTIVE/COILING) ahead of [STOPPED] ones first, THEN by earliest
    # Seen time within each group. Sorting by Seen time alone (old behavior) let a pile of
    # long-dead, already-stopped-out anchors -- which had more of the day to both fire AND
    # get killed, so they skew toward earlier Seen times -- outrank and bury genuinely live
    # setups that simply fired later, and the TOP_N truncation below could then drop live
    # setups entirely in favor of historical ones.
    bulls.sort(key=lambda r: (r['State'] == "[STOPPED]", r['CheckpointTime'], r['State'] != "[ACTIVE BUY]", -r['DayChangePct']))
    bears.sort(key=lambda r: (r['State'] == "[STOPPED]", r['CheckpointTime'], r['State'] != "[ACTIVE SELL]", r['DayChangePct']))

    all_bulls, all_bears = list(bulls), list(bears)   # full pre-truncation lists, for --watch / OrderPlaced below

    bulls = bulls[:TOP_N_BUYERS]
    bears = bears[:TOP_N_SELLERS]

    bull_count_str = f"{len(bulls)} (of {true_bull_count})" if true_bull_count > TOP_N_BUYERS else f"{true_bull_count}"
    bear_count_str = f"{len(bears)} (of {true_bear_count})" if true_bear_count > TOP_N_SELLERS else f"{true_bear_count}"

    shown = bulls + bears + rejected
    sym_w = max(12, min(MAX_SYMBOL_WIDTH, max((len(r['Symbol']) for r in shown), default=12)))

    print(f"\n{COLOR_BOLD}=== STATEFUL INSTITUTIONAL VOLATILITY TRACKER [{mode}] ==={COLOR_RESET}")
    print(f"Target Snapshot: {cutoff_dt.strftime('%Y-%m-%d %H:%M')} | Scanned: {len(by_symbol)} symbols "
          f"across {len(checkpoints)} checkpoints | After dedup: {bull_count_str} bull, {bear_count_str} bear, "
          f"{len(rejected)} graveyard\n")

    def print_basket(title, icon, data_list):
        if not data_list: return
        print(f"\n{COLOR_BOLD}{icon} {title}{COLOR_RESET}")
        header_str = (
            f" {COLOR_CYAN}{'Script':<{sym_w}} {'LTP':>8} {'Day%':>7} | "
            f"{'Seen':^6} {'Anchor':^6} {'Anch Val':>9} {'Move%':>7} | "
            f"{'1X (R-M-D)':^11}  {'2X (R-M-D)':^11}  {'3X (R-M-D)':^11}  {'5X (R-M-D)':^11} | "
            f"{'State':^14}{COLOR_RESET}"
        )
        print(header_str)
        print("-" * len(ANSI_RE.sub("", header_str)))

        for row in data_list:
            day_pct_str = f"{row['DayChangePct']:>+6.2f}%"
            day_color = COLOR_GREEN_FG if row['DayChangePct'] > 0 else COLOR_RED_FG if row['DayChangePct'] < 0 else COLOR_DIM
            move_pct_str = f"{row['MovePct']:>+6.2f}%"
            move_color = COLOR_GREEN_FG if row['MovePct'] > 0 else COLOR_RED_FG if row['MovePct'] < 0 else COLOR_DIM

            k1 = format_kinetic_triad(row.get('BB_RSI_1X'), row.get('BB_MACD_1X'), row.get('ADX_1X'))
            k2 = format_kinetic_triad(row.get('BB_RSI_2X'), row.get('BB_MACD_2X'), row.get('ADX_2X'))
            k3 = format_kinetic_triad(row.get('BB_RSI_3X'), row.get('BB_MACD_3X'), row.get('ADX_3X'))
            k5 = format_kinetic_triad(row.get('BB_RSI_5X'), row.get('BB_MACD_5X'), row.get('ADX_5X'))

            state = row['State']
            if "BUY]" in state: state_text = f"{COLOR_GREEN_FG}{state:^14}{COLOR_RESET}"
            elif "SELL]" in state: state_text = f"{COLOR_RED_FG}{state:^14}{COLOR_RESET}"
            else: state_text = f"{COLOR_YELLOW}{state:^14}{COLOR_RESET}"

            print(
                f" {COLOR_BOLD}{_clip(row['Symbol'], sym_w):<{sym_w}}{COLOR_RESET} "
                f"{row['LTP']:>8.2f} "
                f"{day_color}{day_pct_str}{COLOR_RESET} | "
                f"{row['CheckpointTime'].strftime('%H:%M'):^6} "
                f"{row['ActiveAnchor']:^6} "
                f"{row['AnchorPrice']:>9.2f} "
                f"{move_color}{move_pct_str}{COLOR_RESET} | "
                f"{k1}  {k2}  {k3}  {k5} | "
                f"{state_text}"
            )

    print_basket("TOP BULL SETUPS (Valid Anchors Surviving)", "🔥", bulls)
    print_basket("TOP BEAR SETUPS (Valid Anchors Surviving)", "🩸", bears)
    if not bulls and not bears:
        print(f"{COLOR_YELLOW}No surviving anchors at this snapshot -- every candidate was rejected (see graveyard).{COLOR_RESET}")

    if rejected:
        print(f"\n{COLOR_BOLD}🚫 THE GRAVEYARD (Filtered / Rejected){COLOR_RESET}")
        header_str = (
            f" {COLOR_CYAN}{'Script':<{sym_w}} {'Seen':^6} {'Signal':^6} {'Anchor':^6} {'Anch Val':>9} "
            f"{'Killed':^6} {'Kill Val':>9} {'LTP':>8} {'Day%':>7} {'Move%':>7} | "
            f"{'Forensic Rejection Reason'}{COLOR_RESET}"
        )
        print(header_str)
        print("-" * len(ANSI_RE.sub("", header_str)))

        def sort_reason(r):
            reason = r['RejectReason']
            if "Kinetic SL" in reason: return 0
            if "Warzone" in reason: return 1
            if "Alignment" in reason: return 2
            return 3

        rejected.sort(key=lambda x: (x['CheckpointTime'], sort_reason(x), -abs(x['DayChangePct'])))

        for row in rejected:
            sig = row['AnchorDir']
            if sig == "BULL": sig_colored = f"{COLOR_GREEN_FG}BULL{COLOR_RESET}  "
            elif sig == "BEAR": sig_colored = f"{COLOR_RED_FG}BEAR{COLOR_RESET}  "
            else: sig_colored = f"{COLOR_DIM}NONE{COLOR_RESET}  "

            a_time = row['ActiveAnchor'] if row['ActiveAnchor'] else '-'
            a_price = f"{row['AnchorPrice']:>9.2f}" if row['AnchorPrice'] > 0 else f"{'-':>9}"
            k_time = row['KilledTime'] if row['KilledTime'] else '-'
            k_price = f"{row['KilledPrice']:>9.2f}" if row['KilledPrice'] > 0 else f"{'-':>9}"

            day_pct_str = f"{row['DayChangePct']:>+6.2f}%"
            day_color = COLOR_GREEN_FG if row['DayChangePct'] > 0 else COLOR_RED_FG if row['DayChangePct'] < 0 else COLOR_DIM

            if row['AnchorPrice'] > 0 and row['KilledPrice'] > 0:
                move_pct_str = f"{row['MovePct']:>+6.2f}%"
                move_color = COLOR_GREEN_FG if row['MovePct'] > 0 else COLOR_RED_FG if row['MovePct'] < 0 else COLOR_DIM
            else:
                move_pct_str, move_color = f"{'-':>7}", COLOR_DIM

            reason = row['RejectReason']
            if "Kinetic SL" in reason or "Pullback" in reason or "Rally" in reason: reason_color = COLOR_YELLOW
            elif "Chop" in reason or "Warzone" in reason: reason_color = COLOR_RED_FG
            else: reason_color = COLOR_DIM

            print(
                f" {COLOR_BOLD}{_clip(row['Symbol'], sym_w):<{sym_w}}{COLOR_RESET} "
                f"{row['CheckpointTime'].strftime('%H:%M'):^6} "
                f"{sig_colored} {a_time:^6} {a_price} {k_time:^6} {k_price} "
                f"{row['LTP']:>8.2f} {day_color}{day_pct_str}{COLOR_RESET} "
                f"{move_color}{move_pct_str}{COLOR_RESET} | "
                f"{reason_color}{reason}{COLOR_RESET}"
            )

    if WATCH_VALUE:
        check_watch_alerts(mode, WATCH_VALUE, cutoff_dt, all_bulls, all_bears, rejected)

    # OrderPlaced: purely env-driven, independent of --watch. Empty/unset -> no-op,
    # the run above already printed its results and nothing else happens. Non-empty
    # -> look that value up in this run's results and email its current status.
    order_value = os.environ.get(ORDER_PLACED_ENV, "").strip()
    if order_value:
        check_order_placed(mode, order_value, cutoff_dt, all_bulls, all_bears, rejected)

    _print_error_summary()
    total_calls = sum(l.total_calls for l in LIMITERS.values())
    print(f"\n⏱️ Tracker sync completed in {(time.time() - t_start):.2f} seconds ({total_calls} API calls).\n")
    return 0


def parse_args():
    p = argparse.ArgumentParser(description="Strict institutional volatility tracker (Upstox)")
    p.add_argument("--mode", choices=["STOCK_FNO", "CASH_EQUITY", "INDEX_OPTIONS"], default=TRADING_MODE)
    p.add_argument("--days", type=int, default=BACKTRACE_DAYS, help="trading sessions of history (daily prefilter only)")
    p.add_argument("--date", type=str, default=None, help="Target date (YYYY-MM-DD)")
    p.add_argument("--time", type=str, default="15:30", help="Target time (HH:MM). Defaults to 15:30.")
    p.add_argument("--history-days", type=int, default=MIN1_HISTORY_DAYS,
                   help=f"Calendar days of 1-minute history for indicator math (default: {MIN1_HISTORY_DAYS}).")
    p.add_argument("--disable-bb-kc-gate", action="store_true",
                   help="Drop the 'Bollinger Band already pierced Keltner Channel' requirement (already off by default).")
    p.add_argument("--options-master", type=str, default="",
                   help="INDEX_OPTIONS only: local JSON/CSV instrument master, for backtesting contracts that have already expired.")
    p.add_argument("--opt-min-price", type=float, default=OPT_MIN_PRICE, help="INDEX_OPTIONS: minimum premium at the snapshot.")
    p.add_argument("--opt-min-volume", type=float, default=OPT_MIN_VOLUME, help="INDEX_OPTIONS: minimum traded volume by the snapshot.")
    p.add_argument("--expiry", choices=["CURRENT", "NEXT"], default=EXPIRY_SELECTION,
                   help=f"INDEX_OPTIONS: which expiry to trade -- CURRENT (nearest expiry on/after the snapshot date) "
                        f"or NEXT (the one after that). Default: {EXPIRY_SELECTION}.")
    p.add_argument("--strikes-up", type=int, default=STRIKES_ABOVE_ATM,
                   help=f"INDEX_OPTIONS: number of strikes ABOVE ATM to include, CE+PE each (default {STRIKES_ABOVE_ATM}).")
    p.add_argument("--strikes-down", type=int, default=STRIKES_BELOW_ATM,
                   help=f"INDEX_OPTIONS: number of strikes BELOW ATM to include, CE+PE each (default {STRIKES_BELOW_ATM}).")
    p.add_argument("--checkpoint-min", type=int, default=CHECKPOINT_INTERVAL_MIN,
                   help=f"Size in minutes of each intraday backtrace window, from session open up to the run "
                        f"time (default {CHECKPOINT_INTERVAL_MIN}). E.g. a 10:30 run with 15 checks 9:30, "
                        f"9:45, 10:00, 10:15, 10:30; a 10:35 run adds a final 10:30->10:35 partial checkpoint.")
    p.add_argument("--single-snapshot", action="store_true",
                   help="Disable the backtrace -- evaluate only the current run time, like a single snapshot.")
    p.add_argument("--watch", type=str, default="",
                   help="Optional. A symbol, tradingsymbol, or strike-bearing string to monitor (e.g. 'RELIANCE', "
                        "'NIFTY25000CE', or just '25000' -- matched as a case-insensitive substring of the symbol). "
                        "When set, this run checks that symbol's status and emails an alert via "
                        "SENDER_EMAIL/EMAIL_SENDER, SENDER_PASSWORD/EMAIL_APP_PWD, RECIPIENT_EMAIL/EMAIL_RECEIVER "
                        "(SMTP_SERVER/SMTP_PORT optional, default smtp.gmail.com:465) ONLY if the status changed "
                        "since the last run (tracked in system3_watch_state.json). If this flag is omitted "
                        "(default: no watch), no email code runs at all.")
    p.add_argument("--order-email-only-on-change", action="store_true",
                   help="Switch OrderPlaced tracking (see the OrderPlaced env var) from 'email every run' to "
                        "'email only when the status changes since the last run', using its own state file "
                        f"({ORDER_STATE_FILE}) so it never shares state with --watch.")
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
    WATCH_VALUE = args.watch
    if args.order_email_only_on_change:
        ORDER_EMAIL_ONLY_ON_CHANGE = True
    if args.single_snapshot:
        BACKTRACE_CHECKPOINTS = False
    if args.disable_bb_kc_gate:
        REQUIRE_BB_KC_PIERCE = False
    try:
        code = run_screener(args.mode, args.days, args.date, args.time, args.options_master)
    except Exception:
        print(f"\n{COLOR_RED_FG}💥 FATAL: unhandled exception -- full traceback follows.{COLOR_RESET}")
        traceback.print_exc()
        code = 1
    sys.exit(code)
