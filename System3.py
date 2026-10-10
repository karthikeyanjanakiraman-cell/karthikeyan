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

LOGIC-GAP FIXES IN THIS REVISION (each one is marked "FIX" in the code)
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
# FIX 1: "375min" = ONE NSE session (09:15-15:30) and buckets are anchored to the open (it was "390min"
# anchored to midnight, which cut every session in two at 13:00).
ATR_BASIS_TF = "375min"
ATR_MIN_BARS = 5            # FIX 1: fewer completed sessions than this => "short history" (unless pinned)
MIN_ATR_PCT = 0.001
# FIX 5: unconditional trailing floor, in ATRs below the TRUE peak since the anchor (0 = disabled).
HARD_STOP_ATR_MULT = 2.0

TOP_N_BUYERS = 50
TOP_N_SELLERS = 50
TOP_N_GRAVEYARD = 100       # FIX 8: 0 = print the whole graveyard
MAX_SYMBOL_WIDTH = 30

# --- INTRADAY BACKTRACE CHECKPOINTS ---
CHECKPOINT_INTERVAL_MIN = 15
BACKTRACE_CHECKPOINTS = True

REQUIRE_BB_KC_PIERCE = False

# FIX 8: "PREV_CLOSE" = broker-style day change; "OPEN" = move since today's first print.
DAY_PCT_BASIS = "PREV_CLOSE"

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
# run_tracking() for the exact rules and ORDER_EMAIL_ONLY_ON_CHANGE below to
# switch it to change-only emails instead. The value may hold several
# positions separated by comma / semicolon / newline.
ORDER_PLACED_ENV = "OrderPlaced"
ORDER_EMAIL_ONLY_ON_CHANGE = False
ORDER_STATE_FILE = "system3_order_state.json"
MAX_PINNED_MATCHES = 8      # a loose tracking value pins at most this many instruments

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
# FIX 1: 19 calendar days still costs 3 seven-day chunk calls but holds ~13 sessions (15 held ~10, which
# cannot feed a 14-bar ATR).
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
PROBE_KEY = INDEX_CONFIG["NIFTY"]["spot_key"]     # used to ask "did the exchange trade on that date?"

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
    """Case-, whitespace-, '_' and '-' insensitive form of a symbol, used for all tracking matches."""
    return re.sub(r'[\s_\-]+', '', str(s).upper())

def parse_track_values(raw):
    """'A, B;C' -> ['A', 'B', 'C'] (de-duplicated on the normalised form, order kept)."""
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
    FIX 8: the interval is forced to >= 1 minute (0 or a negative value used to loop forever).
    """
    interval = max(1, int(interval_min or CHECKPOINT_INTERVAL_MIN))
    session_open = _open_dt(cutoff_dt.date())
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

# ------------------------------------------------------------------------------
# FIX 3: which session can actually be scanned?
# ------------------------------------------------------------------------------
def _session_traded(d, today):
    """True if the exchange traded on date d. Probed with the NIFTY 50 index 1-min candles (one call)."""
    if d.weekday() >= 5:
        return False
    if d == today:
        df = fetch_today(PROBE_KEY)
        return df is not None and not df.empty
    status, df = _candles(_url_range_1m(PROBE_KEY, d, d + timedelta(days=1)))
    return bool(status == 200 and df is not None and (df['Datetime'].dt.date == d).any())

def resolve_session(cutoff_dt):
    """
    Turn the requested snapshot into one that can actually be scanned. Returns (cutoff_dt, notes).
      * a snapshot in the future is clamped to 'now';
      * a snapshot at/before the 09:15 open, on a weekend, or on a day the exchange did not trade
        (holiday) rolls back to the 15:30 CLOSE of the last completed session. These runs used to die with
        "no usable data" although the previous session was perfectly scannable.
    """
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
        return cutoff_dt, notes                  # the caller reports the rejected token
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

# ------------------------------------------------------------------------------
# FIX 7: tracked instruments are pinned into the universe
# ------------------------------------------------------------------------------
def resolve_pins(cands, needles, label="symbol"):
    """
    cands = [(symbol, sort_key), ...]. A tracking value pins every candidate whose normalised symbol EQUALS
    it (exact hits win) or, failing that, CONTAINS it. Pinned instruments are always scanned: the
    price/volume prefilter, the ATM strike window and the illiquid filter do not apply to them.
    """
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

# ------------------------------------------------------------------------------
# INDEX OPTIONS universe
# ------------------------------------------------------------------------------
def _resolve_index_spot_price(key, name, cutoff_dt, use_intraday):
    """
    FIX 3: for a snapshot that falls on TODAY the intraday feed is used (the historical endpoint does not
    serve the current session, so a --date <today> run used to silently pick up YESTERDAY's close).
    FIX 2: only candles that closed before the snapshot count.
    """
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

    # FIX 7: tracked contracts first -- every expiry on/after the snapshot, no ATM window, no liquidity filter.
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

# @@STAGE2@@
