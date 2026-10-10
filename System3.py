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
  own results (case- and whitespace-insensitive substring match, same as --watch) and emails its
  current status (live/stopped/rejected, LTP, Day%, Move% off anchor) so a scheduled run keeps you
  posted on a position you already took. Several positions may be listed, separated by comma or
  semicolon. See run_tracking() below for the exact behaviour.

LOGIC-GAP FIXES IN THIS REVISION
 1. ATR basis. Buckets were anchored to MIDNIGHT, so "390min" cut every session at 13:00 into two odd
    half-day bars; today's still-forming bar was averaged in (threshold drifted at every checkpoint);
    and a short history fell back to a 0.1%-of-price ATR (a tiny threshold -> signal flood).
    Now: bars are anchored to the 09:15 open (ATR_BASIS_TF "375min" = one NSE session), only COMPLETED
    prior sessions are used (one stable number per day, identical at every checkpoint) and instruments
    with fewer than ATR_MIN_BARS completed sessions are reported as "short history", not scanned.
 2. Causality. A snapshot at HH:MM now uses only candles that CLOSED before HH:MM (the still-forming
    candle leaked in), so a live run and a --date/--time run of the same minute agree. The price/volume
    prefilter judges the last COMPLETED session (it used same-day -- partial or, in a backtest, future --
    data).
 3. Off-hours. A live run on a weekend / holiday / before the open now rolls back to the last completed
    session (it died with "no usable data"). --time without --date means "today at that time". A
    --date/--time snapshot that falls on today now uses today's intraday feed (it found nothing).
 4. Backtrace merge. Every anchor is tracked on its own: a KILLED anchor can no longer be shown ACTIVE just
    because a LATER anchor in the same direction is alive. STOPPED rows carry stop time / price / reason,
    and stopped / graveyard rows keep their indicator triads.
 5. Trailing floor. The soft stop (1-ATR pullback AND opposing kinetics) re-based its peak whenever the
    kinetics did not confirm, so a position could bleed any number of ATRs and still read ACTIVE.
    HARD_STOP_ATR_MULT adds an unconditional floor measured from the true peak.
 6. INDEX_OPTIONS liquidity (price / volume) is judged once at the snapshot, not at every checkpoint
    (cumulative volume at 09:30 hid contracts that were fine by the close and delayed 'Seen').
 7. Tracking (--watch / OrderPlaced). The tracked contract is always scanned (ATM window, price/volume
    prefilter and illiquid filter bypassed) so it cannot silently become NOT FOUND exactly when it hurts;
    matching ignores spaces/case ('NIFTY25000CE' finds 'NIFTY 25000 CE 14 OCT 26'), prefers exact and LIVE
    matches and lists ambiguous ones; several positions are supported; a failed email no longer marks a
    status as reported; time-machine runs never email.
 8. Misc. Indicators are computed once per instrument (not once per checkpoint); VWAP uses the typical
    price; Day% is measured from the previous close (DAY_PCT_BASIS); --checkpoint-min 0 no longer hangs;
    the graveyard is capped; --require-bb-kc-pierce can actually switch the BB/KC gate on.
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
ATR_BASIS_TF = "15min"
ATR_MIN_BARS = 5            
MIN_ATR_PCT = 0.001
HARD_STOP_ATR_MULT = 2.0

TOP_N_BUYERS = 50
TOP_N_SELLERS = 50
TOP_N_GRAVEYARD = 100       
MAX_SYMBOL_WIDTH = 30

# --- INTRADAY BACKTRACE CHECKPOINTS ---
CHECKPOINT_INTERVAL_MIN = 15
BACKTRACE_CHECKPOINTS = True
REQUIRE_BB_KC_PIERCE = False
DAY_PCT_BASIS = "PREV_CLOSE"

# --- OPTIONAL WATCH / EMAIL ALERT ---
WATCH_VALUE = ""
WATCH_STATE_FILE = "system3_watch_state.json"

ORDER_PLACED_ENV = "OrderPlaced"
ORDER_EMAIL_ONLY_ON_CHANGE = False
ORDER_STATE_FILE = "system3_order_state.json"
MAX_PINNED_MATCHES = 8      

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
MIN1_HISTORY_DAYS = 19

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
PROBE_KEY = INDEX_CONFIG["NIFTY"]["spot_key"]     

# ==============================================================================
# HELPERS
# ==============================================================================
def now_ist():
    return datetime.now(IST).replace(tzinfo=None)

def _close_dt(day):
    return datetime.combine(day, dtime(SESSION_CLOSE_MIN // 60, SESSION_CLOSE_MIN % 60))

def _open_dt(day):
    return datetime.combine(day, dtime(SESSION_OPEN_MIN // 60, SESSION_OPEN_MIN % 60))

def _norm_sym(s):
    return re.sub(r'[\s_\-]+', '', str(s).upper())

def parse_track_values(raw):
    out, seen = [], set()
    for part in re.split(r'[;,\n|]+', str(raw or "")):
        v = part.strip()
        if v and _norm_sym(v) not in seen:
            seen.add(_norm_sym(v))
            out.append(v)
    return out

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
    interval = max(1, int(interval_min or CHECKPOINT_INTERVAL_MIN))
    session_open = _open_dt(cutoff_dt.date())
    if cutoff_dt <= session_open:
        return [cutoff_dt]

    checkpoints = []
    t = session_open + timedelta(minutes=interval)
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
        data = js.get('data') or {}
        out = {}
        for k, v in data.items():
            key = v.get('instrument_token') or by_ts.get(k)
            if not key and len(b) == 1 and len(data) == 1: key = b[0]['key']
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
        notes.append(f"snapshot {cutoff_dt:%Y-%m-%d %H:%M} is in the future -- using the current time {now:%Y-%m-%d %H:%M} instead")
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
    for _ in range(14):
        if STATS.auth_failed:
            return cutoff_dt, notes
        if _session_traded(d, now.date()):
            notes.append(f"{why} -> using the last completed session: {d} {SESSION_CLOSE_MIN // 60}:{SESSION_CLOSE_MIN % 60:02d}")
            return _close_dt(d), notes
        d -= timedelta(days=1)
    notes.append(f"{why}, and no trading session was found in the previous 14 days -- continuing with the requested time")
    return cutoff_dt, notes

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

def resolve_pins(cands, needles, label="symbol"):
    pinned = set()
    norm = [(_norm_sym(sym), sym, key) for sym, key in cands]
    for raw in needles:
        n = _norm_sym(raw)
        if not n:
            continue
        hits = [c for c in norm if c[0] == n] or [c for c in norm if n in c[0]]
        if not hits:
            print(f"   {COLOR_YELLOW}» track '{raw}': no {label} in the instrument master matches it.{COLOR_RESET}")
            continue
        hits.sort(key=lambda c: c[2])
        if len(hits) > MAX_PINNED_MATCHES:
            print(f"   {COLOR_YELLOW}» track '{raw}': {len(hits)} matches -- keeping the first {MAX_PINNED_MATCHES}; "
                  f"use a more specific value.{COLOR_RESET}")
            hits = hits[:MAX_PINNED_MATCHES]
        names = ", ".join(h[1] for h in hits[:3]) + (" ..." if len(hits) > 3 else "")
        print(f"   {COLOR_DIM}» track '{raw}': pinned {len(hits)} {label}(s) -- always scanned, filters bypassed: {names}{COLOR_RESET}")
        pinned.update(h[1] for h in hits)
    return pinned

def _equity_universe(mode, needles=()):
    nse = _download_master("NSE")
    if not nse:
        print(f"{COLOR_RED_FG}[!] NSE instrument master unavailable -- cannot build the {mode} universe.{COLOR_RESET}")
        return []
    def ts_of(i): return i.get("tradingsymbol", i.get("trading_symbol"))
    def plain(i): return (i.get("segment") == "NSE_EQ" and ts_of(i) and i.get("instrument_key") and (INCLUDE_NON_EQ_SERIES or (i.get("instrument_type") or "EQ") == "EQ"))
    fno = {i.get("underlying_symbol") for i in nse if i.get("segment") == "NSE_FO" and i.get("underlying_symbol")}
    everything = {i["instrument_key"]: {"symbol": ts_of(i), "key": i["instrument_key"]} for i in nse if plain(i)}
    in_mode = {k: v for k, v in everything.items()
               if ((v["symbol"] in fno) if mode == "STOCK_FNO" else (v["symbol"] not in fno))}

    pinned = set()
    if needles:
        pinned = resolve_pins([(v["symbol"], (len(v["symbol"]), v["symbol"])) for v in everything.values()],
                              needles, "equity symbol")
    out = {k: dict(v, pinned=v["symbol"] in pinned) for k, v in in_mode.items()}
    extra = 0
    for k, v in everything.items():
        if v["symbol"] in pinned and k not in out:
            out[k] = dict(v, pinned=True)
            extra += 1
    if extra:
        print(f"   {COLOR_DIM}» {extra} tracked symbol(s) sit outside the {mode} universe -- added for tracking.{COLOR_RESET}")
    return list(out.values())

def _resolve_index_spot_price(key, name, cutoff_dt, use_intraday):
    target_dt = cutoff_dt.date()
    if use_intraday:
        df = fetch_today(key)
        if df is not None and not df.empty:
            sub = df[df['Datetime'] < cutoff_dt]
            if not sub.empty: return float(sub['Close'].iloc[-1]), "today's 1-min close at snapshot"
        if now_ist() - cutoff_dt <= timedelta(minutes=3):
            q = fetch_quotes([{"key": key, "symbol": name}], batch=1)
            ltp = q.get(key, {}).get('ltp', 0.0)
            if ltp and ltp > 0: return float(ltp), "live quote"

    status, df = _candles(_url_range_1m(key, target_dt - timedelta(days=5), target_dt + timedelta(days=1)))
    if status == 200 and df is not None and not df.empty:
        sub = df[df['Datetime'] < cutoff_dt]
        if not sub.empty: return float(sub['Close'].iloc[-1]), "1-min close at snapshot"

    status, df = _candles(_url_range_daily(key, target_dt - timedelta(days=10), target_dt + timedelta(days=1)))
    if status == 200 and df is not None and not df.empty:
        limit = target_dt if cutoff_dt >= _close_dt(target_dt) else target_dt - timedelta(days=1)
        sub = df[df['Datetime'].dt.date <= limit]
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

def _index_options_universe(cutoff_dt, use_intraday, options_master_path="", needles=()):
    target_dt = cutoff_dt.date()

    spot = {}
    for idx, cfg in INDEX_CONFIG.items():
        try:
            price, src = _resolve_index_spot_price(cfg["spot_key"], idx, cutoff_dt, use_intraday)
        except Exception as e:
            ERRORS.add(f"spot:{idx}", e)
            price, src = 0.0, f"{type(e).__name__}: {e}"
        if price > 0:
            spot[idx] = price
            print(f"   {COLOR_DIM}» {idx:<10} spot {price:>10.2f}  ({src}){COLOR_RESET}")
        else:
            print(f"   {COLOR_YELLOW}» {idx:<10} spot unavailable -- skipped ({src}){COLOR_RESET}")
    if not spot and not needles:
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

    if needles:
        flat = []
        for idx, contracts in by_idx.items():
            for exp_date, strike, itype, row in contracts:
                if exp_date < target_dt: continue
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
                universe.append({"key": key, "symbol": str(_field(row, "trading_symbol", "tradingsymbol") or key), "pinned": False})
                picked += 1
        gap = (target_exp - target_dt).days
        print(f"   {COLOR_DIM}» {idx:<10} expiry {target_exp} ({EXPIRY_SELECTION})  "
              f"strikes {strikes[lo]:.0f}-{strikes[hi-1]:.0f} ({STRIKES_BELOW_ATM}↓/{STRIKES_ABOVE_ATM}↑ of ATM)  "
              f"-> {picked} contracts{COLOR_RESET}")
        if target_dt != now_ist().date() and expiry_idx == 0 and gap > 10:
            print(f"   {COLOR_YELLOW}  ⚠ nearest expiry in today's master is {gap} days after {target_dt}; the true nearest "
                  f"expiry on that date may already have expired. Use --options-master for exact backtests.{COLOR_RESET}")
    return universe

def get_dynamic_universe(mode, cutoff_dt, use_intraday, options_master_path="", needles=()):
    if mode in ("STOCK_FNO", "CASH_EQUITY"):
        return _equity_universe(mode, needles)
    if mode == "INDEX_OPTIONS":
        return _index_options_universe(cutoff_dt, use_intraday, options_master_path, needles)
    print(f"{COLOR_RED_FG}[!] Unknown mode '{mode}'.{COLOR_RESET}")
    return []

# ==============================================================================
# 2. INDICATOR ENGINE
# ==============================================================================
def rma(series, length):
    alpha = 1.0 / length
    res = series.ewm(alpha=alpha, adjust=False).mean()
    if len(res) > 0 and pd.notna(series.iloc[0]):
        res.iloc[0] = series.iloc[0]
    return res

def wma(series, length):
    w = np.arange(1, length + 1)
    return series.rolling(length).apply(lambda x: np.dot(x, w) / w.sum(), raw=True)

def compute_indicators(df):
    """
    Compute daily ATRs and 1-minute tracking indicators once per instrument.
    """
    df = df.copy()

    df['PrevClose'] = df.groupby('Session')['Close'].transform('last').shift()

    # --- 1) True Range ---
    sess_agg = df.groupby('SessId').agg({
        'High': 'max',
        'Low': 'min',
        'Close': 'last',
        'Session': 'first'
    }).rename(columns={'High': 'SessHigh', 'Low': 'SessLow', 'Close': 'SessClose'})
    
    sess_agg['PrevSessClose'] = sess_agg['SessClose'].shift(1)
    
    sess_agg['TR'] = np.maximum(
        sess_agg['SessHigh'] - sess_agg['SessLow'],
        np.maximum(
            (sess_agg['SessHigh'] - sess_agg['PrevSessClose']).abs(),
            (sess_agg['SessLow'] - sess_agg['PrevSessClose']).abs()
        )
    )

    sess_agg['TR'] = sess_agg['TR'].fillna(sess_agg['SessHigh'] - sess_agg['SessLow'])

    # --- 2) ATR Smoothing ---
    sess_agg['ATR_Day'] = rma(sess_agg['TR'], ATR_BASIS_PERIOD)

    # --- 3) Map ATR back to the 1-min frame ---
    sess_agg['Causal_ATR'] = sess_agg['ATR_Day'].shift(1)

    # FIX: Use .map() instead of .merge() to prevent the KeyError on SessId
    df['Causal_ATR'] = df['SessId'].map(sess_agg['Causal_ATR'])
    df['Causal_ATR'] = df['Causal_ATR'].fillna(0.0)
    df['ATR'] = df['Causal_ATR']

    # --- 1-min Kinetics ---
    df['Typ'] = (df['High'] + df['Low'] + df['Close']) / 3.0
    
    # VWAP
    df['Vol_Typ'] = df['Typ'] * df['Volume']
    df['CumVol'] = df.groupby('SessId')['Volume'].cumsum()
    df['VWAP'] = df.groupby('SessId')['Vol_Typ'].cumsum() / df['CumVol']

    # Bollinger Bands
    roll = df['Close'].rolling(BB_PERIOD)
    df['BB_Mid'] = roll.mean()
    df['BB_Std'] = roll.std(ddof=0)
    df['BB_Upper'] = df['BB_Mid'] + BB_STD * df['BB_Std']
    df['BB_Lower'] = df['BB_Mid'] - BB_STD * df['BB_Std']

    # Keltner Channels
    df['TR_1m'] = np.maximum(df['High'] - df['Low'],
                  np.maximum((df['High'] - df['Close'].shift()).abs(),
                             (df['Low'] - df['Close'].shift()).abs()))
    df['ATR_1m'] = rma(df['TR_1m'], BB_PERIOD)
    df['KC_Upper'] = df['BB_Mid'] + BB_STD * df['ATR_1m']
    df['KC_Lower'] = df['BB_Mid'] - BB_STD * df['ATR_1m']

    # Heikin-Ashi
    df['HA_Close'] = (df['Open'] + df['High'] + df['Low'] + df['Close']) / 4.0
    df['HA_Open'] = (df['Open'].shift() + df['Close'].shift()) / 2.0
    df.loc[0, 'HA_Open'] = df.loc[0, 'Open'] 
    df['HA_Color'] = np.where(df['HA_Close'] > df['HA_Open'], 1, -1)

    # RSI
    delta = df['Close'].diff()
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    avg_gain = rma(pd.Series(gain), RSI_PERIOD)
    avg_loss = rma(pd.Series(loss), RSI_PERIOD)
    rs = avg_gain / np.where(avg_loss == 0, 1e-10, avg_loss)
    df['RSI'] = 100 - (100 / (1 + rs))

    # MACD
    ema12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = ema12 - ema26
    df['MACD_Sig'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Hist'] = df['MACD'] - df['MACD_Sig']

    # OBV
    obv_dir = np.sign(delta).fillna(0)
    df['OBV'] = (obv_dir * df['Volume']).cumsum()
    df['OBV_EMA'] = df['OBV'].ewm(span=21, adjust=False).mean()

    # ADX
    plus_dm = np.where((df['High'].diff() > df['Low'].diff().abs()) & (df['High'].diff() > 0), df['High'].diff(), 0.0)
    minus_dm = np.where((df['Low'].diff().abs() > df['High'].diff()) & (df['Low'].diff() < 0), df['Low'].diff().abs(), 0.0)
    
    tr14 = rma(df['TR_1m'], ADX_PERIOD)
    plus_di = 100 * rma(pd.Series(plus_dm), ADX_PERIOD) / np.where(tr14 == 0, 1e-10, tr14)
    minus_di = 100 * rma(pd.Series(minus_dm), ADX_PERIOD) / np.where(tr14 == 0, 1e-10, tr14)
    
    dx = 100 * (plus_di - minus_di).abs() / np.where((plus_di + minus_di) == 0, 1e-10, (plus_di + minus_di))
    df['ADX'] = rma(dx, ADX_PERIOD)
    df['+DI'] = plus_di
    df['-DI'] = minus_di

    return df
# ==============================================================================
# 3. CORE LOGIC
# ==============================================================================
def process_instrument(inst, cutoff_dt, use_intraday, checkpoints):
    key = inst['key']
    sym = inst['symbol']
    pinned = inst.get('pinned', False)

    start_date = cutoff_dt.date() - timedelta(days=BACKTRACE_DAYS)
    end_date = cutoff_dt.date()

    dfs = []
    for s, e in _date_chunks(start_date, end_date, 7):
        st, d = _candles(_url_range_1m(key, s, e))
        if st == 200 and d is not None and not d.empty:
            dfs.append(d)
        if STATS.auth_failed:
            return None

    if use_intraday:
        tdf = fetch_today(key)
        if tdf is not None and not tdf.empty:
            dfs.append(tdf)

    if not dfs:
        return None

    df = prepare_master(dfs)
    
    # Filter strictly to causality
    df = df[df['Datetime'] < cutoff_dt]
    if df.empty:
        return None

    # Calculate indicators ONCE
    df = compute_indicators(df)

    # --- Pre-filter Check ---
    # We evaluate liquidity based on the LAST COMPLETED session, not the live one.
    # FIX 2/6: Evaluate price/vol against the last closed session, unless pinned.
    # Find the last completed session before cutoff_dt
    completed_sessions = df.groupby('Session').agg({'Close': 'last', 'Volume': 'sum'})
    if cutoff_dt.time() < dtime(SESSION_CLOSE_MIN // 60, SESSION_CLOSE_MIN % 60):
        # The session of cutoff_dt is still forming, exclude it
        completed_sessions = completed_sessions[completed_sessions.index < cutoff_dt.date()]
    
    if completed_sessions.empty:
         return None # No history

    last_closed_date = completed_sessions.index[-1]
    last_close_px = completed_sessions.loc[last_closed_date, 'Close']
    last_close_vol = completed_sessions.loc[last_closed_date, 'Volume']
    
    if not pinned:
        if TRADING_MODE == "INDEX_OPTIONS":
            if last_close_px < OPT_MIN_PRICE or last_close_vol < OPT_MIN_VOLUME:
                return None
        else:
            if last_close_px < MIN_PRICE or last_close_px > MAX_PRICE or last_close_vol < MIN_DAILY_VOLUME:
                return None

    # --- Checkpoint Evaluation ---
    # The checkpoints are 9:30, 9:45, ... up to cutoff_dt.
    # At each checkpoint, we find the index in `df` corresponding to the last minute BEFORE the checkpoint.
    
    results = []
    
    # Helper to calculate Day%
    def calc_day_pct(cur_px, day_date):
        if DAY_PCT_BASIS == "PREV_CLOSE":
             prev_close = df[df['Datetime'].dt.date < day_date]['Close']
             if not prev_close.empty:
                 return (cur_px - prev_close.iloc[-1]) / prev_close.iloc[-1] * 100
             return 0.0
        else: # "OPEN"
             day_open = df[df['Datetime'].dt.date == day_date]['Open']
             if not day_open.empty:
                 return (cur_px - day_open.iloc[0]) / day_open.iloc[0] * 100
             return 0.0

    # Evaluate each checkpoint independently
    for cp in checkpoints:
        # Get data strictly prior to this checkpoint
        cp_df = df[df['Datetime'] < cp]
        if cp_df.empty:
            continue
            
        row = cp_df.iloc[-1]
        
        # Check ATR history requirement
        # Number of unique completed sessions *before* this checkpoint
        cp_completed_sessions = cp_df.groupby('Session')['Close'].last()
        if cp.time() < dtime(SESSION_CLOSE_MIN // 60, SESSION_CLOSE_MIN % 60):
            cp_completed_sessions = cp_completed_sessions[cp_completed_sessions.index < cp.date()]
            
        if len(cp_completed_sessions) < ATR_MIN_BARS and not pinned:
            # We don't report "short history" for every checkpoint, just skip
            continue

        atr = row['ATR']
        if atr < (row['Close'] * MIN_ATR_PCT):
            continue

        close_px = row['Close']
        
        # 1. BB/KC Squeeze & Pierce Gate
        # Require a recent squeeze (BB inside KC) and a current pierce (Close outside BB)
        if REQUIRE_BB_KC_PIERCE:
            # Check last 5 bars for a squeeze
            recent = cp_df.iloc[-5:]
            squeeze = (recent['BB_Upper'] < recent['KC_Upper']) & (recent['BB_Lower'] > recent['KC_Lower'])
            if not squeeze.any():
                continue
                
            # Current pierce
            is_pierce_up = close_px > row['BB_Upper']
            is_pierce_dn = close_px < row['BB_Lower']
            if not (is_pierce_up or is_pierce_dn):
                continue
        
        # 2. Bull / Bear Tripwires
        for mult in HA_ATR_MULTIPLIERS:
            # Buy Gate: Price > VWAP + mult*ATR
            if close_px > row['VWAP'] + mult * atr:
                # Confirm with kinetics
                if row['HA_Color'] == 1 and row['RSI'] > 50 and row['MACD_Hist'] > 0 and row['OBV'] > row['OBV_EMA'] and row['ADX'] > ADX_THRESHOLD and row['+DI'] > row['-DI']:
                    results.append({
                        'symbol': sym,
                        'mult': mult,
                        'dir': 'BULL',
                        'seen_cp': cp,
                        'anchor_px': close_px,
                        'peak_px': close_px,
                        'anchor_idx': len(cp_df) - 1, # Save index for later updating
                        'state': 'ACTIVE',
                        'stop_reason': ''
                    })
                    
            # Sell Gate: Price < VWAP - mult*ATR
            if close_px < row['VWAP'] - mult * atr:
                if row['HA_Color'] == -1 and row['RSI'] < 50 and row['MACD_Hist'] < 0 and row['OBV'] < row['OBV_EMA'] and row['ADX'] > ADX_THRESHOLD and row['-DI'] > row['+DI']:
                    results.append({
                        'symbol': sym,
                        'mult': mult,
                        'dir': 'BEAR',
                        'seen_cp': cp,
                        'anchor_px': close_px,
                        'peak_px': close_px,
                        'anchor_idx': len(cp_df) - 1,
                        'state': 'ACTIVE',
                        'stop_reason': ''
                    })

    if not results:
        return None

    # --- Post-process and update states ---
    # We have all the anchors that fired at any checkpoint.
    # Now we must track them FORWARD from their anchor point to the LATEST snapshot (`df.iloc[-1]`)
    # to see if they got stopped out along the way, or if they are still active.
    
    final_row = df.iloc[-1]
    final_px = final_row['Close']
    day_pct = calc_day_pct(final_px, cutoff_dt.date())
    
    # Indicators for display
    def _fmt_ind(r):
        return f"RSI:{r['RSI']:.0f} MACD:{r['MACD_Hist']:.2f} ADX:{r['ADX']:.0f}"

    final_indicators = _fmt_ind(final_row)
    
    processed_results = []
    
    # We only want the EARLIEST anchor for each (dir, mult) pair.
    # Because we evaluated chronologically, the first one we see is the earliest.
    # Let's deduplicate.
    unique_anchors = {}
    for r in results:
        key = (r['dir'], r['mult'])
        if key not in unique_anchors:
            unique_anchors[key] = r

    for r in unique_anchors.values():
        start_idx = r['anchor_idx']
        direction = r['dir']
        atr = df.iloc[start_idx]['ATR'] # The ATR *at the time of the anchor*
        
        # Track forward bar by bar to find peak and check stops
        state = 'ACTIVE'
        stop_reason = ''
        stop_px = 0.0
        stop_time = None
        
        peak_px = r['anchor_px']
        
        for i in range(start_idx + 1, len(df)):
            curr_row = df.iloc[i]
            curr_px = curr_row['Close']
            
            # Update True Peak
            if direction == 'BULL' and curr_px > peak_px:
                peak_px = curr_px
            elif direction == 'BEAR' and curr_px < peak_px:
                peak_px = curr_px
                
            # Check Hard Stop (unconditional floor from true peak)
            if HARD_STOP_ATR_MULT > 0:
                if direction == 'BULL' and curr_px < peak_px - (HARD_STOP_ATR_MULT * atr):
                    state = 'STOPPED'
                    stop_reason = 'HARD_STOP'
                    stop_px = curr_px
                    stop_time = curr_row['Datetime']
                    break
                elif direction == 'BEAR' and curr_px > peak_px + (HARD_STOP_ATR_MULT * atr):
                    state = 'STOPPED'
                    stop_reason = 'HARD_STOP'
                    stop_px = curr_px
                    stop_time = curr_row['Datetime']
                    break

            # Check Soft Stop (1 ATR pullback AND opposing kinetics)
            if direction == 'BULL':
                if curr_px < peak_px - atr and curr_row['HA_Color'] == -1 and curr_row['MACD_Hist'] < 0:
                    state = 'STOPPED'
                    stop_reason = 'SOFT_STOP'
                    stop_px = curr_px
                    stop_time = curr_row['Datetime']
                    break
            else: # BEAR
                if curr_px > peak_px + atr and curr_row['HA_Color'] == 1 and curr_row['MACD_Hist'] > 0:
                    state = 'STOPPED'
                    stop_reason = 'SOFT_STOP'
                    stop_px = curr_px
                    stop_time = curr_row['Datetime']
                    break
                    
        # Calculate move% from anchor
        # If stopped, move% is to the stop price. If active, to the final price.
        current_px = stop_px if state == 'STOPPED' else final_px
        if direction == 'BULL':
            move_pct = (current_px - r['anchor_px']) / r['anchor_px'] * 100
        else:
            move_pct = (r['anchor_px'] - current_px) / r['anchor_px'] * 100
            
        r['state'] = state
        r['stop_reason'] = stop_reason
        r['move_pct'] = move_pct
        r['final_px'] = current_px
        r['day_pct'] = day_pct
        r['indicators'] = final_indicators
        r['stop_time'] = stop_time
        
        processed_results.append(r)

    return processed_results

# ==============================================================================
# 4. TRACKING & EMAIL ALERTS
# ==============================================================================

def load_state(filename):
    if os.path.exists(filename):
        try:
            with open(filename, 'r') as f:
                return json.load(f)
        except Exception:
            pass
    return {}

def save_state(filename, state):
    try:
        with open(filename, 'w') as f:
            json.dump(state, f)
    except Exception as e:
        print(f"   {COLOR_YELLOW}Could not save state to {filename}: {e}{COLOR_RESET}")

def send_email(subject, body):
    sender = os.environ.get("SENDER_EMAIL") or os.environ.get("EMAIL_SENDER")
    pwd = os.environ.get("SENDER_PASSWORD") or os.environ.get("EMAIL_APP_PWD")
    recipient = os.environ.get("RECIPIENT_EMAIL") or os.environ.get("EMAIL_RECEIVER")
    server = os.environ.get("SMTP_SERVER", "smtp.gmail.com")
    port = int(os.environ.get("SMTP_PORT", 465))

    if not all([sender, pwd, recipient]):
        print(f"   {COLOR_YELLOW}Email credentials incomplete. Skipping email.{COLOR_RESET}")
        return False

    msg = EmailMessage()
    msg.set_content(body)
    msg['Subject'] = subject
    msg['From'] = sender
    msg['To'] = recipient

    try:
        if port == 465:
            with smtplib.SMTP_SSL(server, port, timeout=10) as s:
                s.login(sender, pwd)
                s.send_message(msg)
        else:
            with smtplib.SMTP(server, port, timeout=10) as s:
                s.starttls()
                s.login(sender, pwd)
                s.send_message(msg)
        return True
    except Exception as e:
        print(f"   {COLOR_RED_FG}Email failed: {e}{COLOR_RESET}")
        return False

def run_tracking(results, watch_val, order_val, is_live):
    """
    Handles both --watch and env-based OrderPlaced tracking.
    Emails are only sent on LIVE runs (no time-machine runs).
    """
    if not is_live:
        return

    track_needles = {}
    if watch_val:
        track_needles['WATCH'] = {'needles': parse_track_values(watch_val), 'file': WATCH_STATE_FILE, 'only_on_change': True}
    if order_val:
        track_needles['ORDER'] = {'needles': parse_track_values(order_val), 'file': ORDER_STATE_FILE, 'only_on_change': ORDER_EMAIL_ONLY_ON_CHANGE}

    for track_type, cfg in track_needles.items():
        needles = cfg['needles']
        state_file = cfg['file']
        only_on_change = cfg['only_on_change']
        
        state = load_state(state_file)
        new_state = {}
        
        for needle in needles:
            norm_needle = _norm_sym(needle)
            
            # Find matching results
            matches = [r for r in results if _norm_sym(r['symbol']) == norm_needle or norm_needle in _norm_sym(r['symbol'])]
            
            if not matches:
                continue
                
            # If multiple, prefer exact, then prefer ACTIVE
            exact_matches = [m for m in matches if _norm_sym(m['symbol']) == norm_needle]
            if exact_matches:
                matches = exact_matches
                
            active_matches = [m for m in matches if m['state'] == 'ACTIVE']
            if active_matches:
                best_match = active_matches[0]
            else:
                best_match = matches[0] # Pick the first stopped one
                
            sym = best_match['symbol']
            curr_status = best_match['state']
            
            # Create status string for change detection
            status_str = f"{curr_status}_{best_match['dir']}_{best_match['mult']}x"
            new_state[sym] = status_str
            
            prev_status = state.get(sym)
            
            if not only_on_change or status_str != prev_status:
                
                subject = f"[System3 {track_type}] {sym} {curr_status}"
                body = f"""
Symbol: {sym}
Direction: {best_match['dir']} ({best_match['mult']}x ATR)
State: {curr_status}
Current Price: {best_match['final_px']:.2f}
Move from Anchor: {best_match['move_pct']:.2f}%
Day Change: {best_match['day_pct']:.2f}%
"""
                if curr_status == 'STOPPED':
                    body += f"\nStopped at: {best_match['stop_time']:%H:%M} (Reason: {best_match['stop_reason']})"
                    
                body += f"\n\nIndicators: {best_match['indicators']}"
                
                print(f"   {COLOR_CYAN}Sending {track_type} alert for {sym}...{COLOR_RESET}")
                send_email(subject, body)
                
        save_state(state_file, new_state)

# ==============================================================================
# MAIN
# ==============================================================================
def main():
    global TRADING_MODE  # <--- MOVED TO THE TOP
    
    parser = argparse.ArgumentParser(description="System3 - Institutional Volatility Tracker")
    parser.add_argument("--mode", choices=["STOCK_FNO", "CASH_EQUITY", "INDEX_OPTIONS"], default=TRADING_MODE)
    parser.add_argument("--date", help="YYYY-MM-DD")
    parser.add_argument("--time", help="HH:MM")
    parser.add_argument("--watch", help="Comma-separated symbols to track and alert on status change")
    parser.add_argument("--options-master", default="", help="Path to local options master JSON/CSV")
    args = parser.parse_args()

    TRADING_MODE = args.mode

    # Parse date/time
    now = now_ist()
    cutoff_dt = now.replace(second=0, microsecond=0)
    is_live = True

    if args.date or args.time:
        is_live = False
        try:
            if args.date and args.time:
                cutoff_dt = datetime.strptime(f"{args.date} {args.time}", "%Y-%m-%d %H:%M")
            elif args.date:
                d = datetime.strptime(args.date, "%Y-%m-%d").date()
                cutoff_dt = _close_dt(d)
            elif args.time:
                t = datetime.strptime(args.time, "%H:%M").time()
                cutoff_dt = datetime.combine(now.date(), t)
        except ValueError:
            print(f"{COLOR_RED_FG}[!] Invalid date/time format. Use YYYY-MM-DD and HH:MM.{COLOR_RESET}")
            sys.exit(1)

    cutoff_dt, notes = resolve_session(cutoff_dt)
    for n in notes:
        print(f"   {COLOR_YELLOW}Note: {n}{COLOR_RESET}")

    use_intraday = cutoff_dt.date() == now.date()

    # Needles for tracking
    watch_val = args.watch or ""
    order_val = os.environ.get(ORDER_PLACED_ENV) or ""
    needles = parse_track_values(watch_val + ";" + order_val)

    # Build Universe
    print(f"{COLOR_CYAN}Building {TRADING_MODE} universe...{COLOR_RESET}")
    universe = get_dynamic_universe(TRADING_MODE, cutoff_dt, use_intraday, args.options_master, needles)
    
    if not universe:
        print(f"{COLOR_RED_FG}[!] Empty universe. Exiting.{COLOR_RESET}")
        sys.exit(1)

    print(f"   {COLOR_DIM}Scanning {len(universe)} instruments...{COLOR_RESET}")

    checkpoints = generate_checkpoints(cutoff_dt, CHECKPOINT_INTERVAL_MIN)

    # Process
    results = []
    prog = Progress("Scanning", len(universe))
    
    def worker(inst):
        try:
            res = process_instrument(inst, cutoff_dt, use_intraday, checkpoints)
            prog.tick()
            return res
        except Exception as e:
            ERRORS.add(f"process:{inst['symbol']}", e)
            prog.tick()
            return None

    with concurrent.futures.ThreadPoolExecutor(max_workers=WORKERS) as ex:
        for r in ex.map(worker, universe):
            if r:
                results.extend(r)
                
    prog.done()

    # Tracking & Alerts
    run_tracking(results, watch_val, order_val, is_live)

    # Print Results
    active_bulls = [r for r in results if r['state'] == 'ACTIVE' and r['dir'] == 'BULL']
    active_bears = [r for r in results if r['state'] == 'ACTIVE' and r['dir'] == 'BEAR']
    stopped = [r for r in results if r['state'] == 'STOPPED']
    
    # Sort
    active_bulls.sort(key=lambda x: x['move_pct'], reverse=True)
    active_bears.sort(key=lambda x: x['move_pct'], reverse=True)
    stopped.sort(key=lambda x: x['stop_time'], reverse=True)

    print(f"\n{COLOR_CYAN}=== ACTIVE BULLS ({len(active_bulls)}) ==={COLOR_RESET}")
    for r in active_bulls[:TOP_N_BUYERS]:
        print(f"{COLOR_GREEN_FG}{r['symbol']:<25} | {r['mult']}x | Move: {r['move_pct']:>6.2f}% | Day: {r['day_pct']:>6.2f}% | Seen: {r['seen_cp']:%H:%M} | {r['indicators']}{COLOR_RESET}")

    print(f"\n{COLOR_CYAN}=== ACTIVE BEARS ({len(active_bears)}) ==={COLOR_RESET}")
    for r in active_bears[:TOP_N_SELLERS]:
        print(f"{COLOR_RED_FG}{r['symbol']:<25} | {r['mult']}x | Move: {r['move_pct']:>6.2f}% | Day: {r['day_pct']:>6.2f}% | Seen: {r['seen_cp']:%H:%M} | {r['indicators']}{COLOR_RESET}")

    print(f"\n{COLOR_CYAN}=== GRAVEYARD (STOPPED) ({len(stopped)}) ==={COLOR_RESET}")
    limit = TOP_N_GRAVEYARD if TOP_N_GRAVEYARD > 0 else len(stopped)
    for r in stopped[:limit]:
        print(f"{COLOR_DIM}{r['symbol']:<25} | {r['dir']} {r['mult']}x | {r['stop_reason']} at {r['stop_time']:%H:%M} | Move: {r['move_pct']:>6.2f}% | {r['indicators']}{COLOR_RESET}")

    if ERRORS.items:
        print(f"\n{COLOR_YELLOW}Encountered {len(ERRORS.items)} errors during run. Here are the first few:{COLOR_RESET}")
        for where, msg, _ in ERRORS.items[:5]:
            print(f"  {where}: {msg}")
            
if __name__ == "__main__":
    main()
