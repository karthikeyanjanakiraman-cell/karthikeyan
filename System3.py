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

# ==============================================================================
# 2. CONTINUOUS KINETIC TRIPWIRE ENGINE
# ==============================================================================
def compute_base_atr(m):
    """
    FIX 1. Mean true range of the last ATR_BASIS_PERIOD COMPLETED bars.
    `m` must hold only sessions BEFORE the target day, so the number is the same at every checkpoint of the day
    (it used to include today's still-forming bar and drift with each checkpoint). Bars are anchored to the
    09:15 open: ATR_BASIS_TF "375min" is one NSE session (the old "390min" bucket was anchored to midnight and
    cut every session at 13:00). Returns (atr, n_bars); (None, 0) when there is no prior session at all.
    """
    if m is None or len(m) == 0:
        return None, 0
    hi, lo, cl = m['High'].values, m['Low'].values, m['Close'].values
    sid = m['SessId'].values
    mins = np.clip(m['Min'].values, SESSION_OPEN_MIN, SESSION_CLOSE_MIN - 1)
    tf_minutes = max(1, int(str(ATR_BASIS_TF).replace("min", "")))
    bucket = sid * 10000 + (mins - SESSION_OPEN_MIN) // tf_minutes
    starts = np.concatenate(([0], np.flatnonzero(np.diff(bucket)) + 1))
    ends = np.concatenate((starts[1:], [len(bucket)])) - 1
    b_high, b_low, b_close = np.maximum.reduceat(hi, starts), np.minimum.reduceat(lo, starts), cl[ends]
    prev_close = np.concatenate(([np.nan], b_close[:-1]))
    tr = np.fmax(np.fmax(b_high - b_low, np.abs(b_high - prev_close)), np.abs(b_low - prev_close))
    atr = float(tr[-ATR_BASIS_PERIOD:].mean())
    return max(atr, float(b_close[-1]) * MIN_ATR_PCT, 0.01), len(b_close)

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

    a_mean = pd.Series(adx).rolling(BB_PERIOD, min_periods=1).mean().values
    a_std = pd.Series(adx).rolling(BB_PERIOD, min_periods=1).std(ddof=0).values

    p_mean = pd.Series(plus_di).rolling(BB_PERIOD, min_periods=1).mean().values
    p_std = pd.Series(plus_di).rolling(BB_PERIOD, min_periods=1).std(ddof=0).values
    m_mean = pd.Series(minus_di).rolling(BB_PERIOD, min_periods=1).mean().values
    m_std = pd.Series(minus_di).rolling(BB_PERIOD, min_periods=1).std(ddof=0).values

    return {
        'rsi': rsi, 'r_mean': rsi_mean, 'r_std': rsi_std,
        'hist': hist, 'h_mean': h_mean, 'h_std': h_std,
        'plus_di': plus_di, 'p_mean': p_mean, 'p_std': p_std,
        'minus_di': minus_di, 'm_mean': m_mean, 'm_std': m_std,
        'adx': adx, 'a_mean': a_mean, 'a_std': a_std
    }

def get_kinetics(kin_1m, idx):
    rsi = kin_1m['rsi'][idx]; r_mean = kin_1m['r_mean'][idx]; r_std = kin_1m['r_std'][idx]
    hist = kin_1m['hist'][idx]; h_mean = kin_1m['h_mean'][idx]; h_std = kin_1m['h_std'][idx]
    p_di = kin_1m['plus_di'][idx]; p_mean = kin_1m['p_mean'][idx]; p_std = kin_1m['p_std'][idx]
    m_di = kin_1m['minus_di'][idx]; m_mean = kin_1m['m_mean'][idx]; m_std = kin_1m['m_std'][idx]
    adx = kin_1m['adx'][idx]; a_mean = kin_1m['a_mean'][idx]; a_std = kin_1m['a_std'][idx]

    bull_rsi = r_std > 0 and rsi > r_mean + BB_STD * r_std
    bear_rsi = r_std > 0 and rsi < r_mean - BB_STD * r_std
    bull_macd = h_std > 0 and hist > h_mean + BB_STD * h_std
    bear_macd = h_std > 0 and hist < h_mean - BB_STD * h_std

    adx_breakout = a_std > 0 and adx > a_mean + BB_STD * a_std
    bull_di = p_std > 0 and p_di > p_mean + BB_STD * p_std and p_di > m_di and adx_breakout
    bear_di = m_std > 0 and m_di > m_mean + BB_STD * m_std and m_di > p_di and adx_breakout

    raw_bull_power = p_std > 0 and p_di > p_mean + BB_STD * p_std
    raw_bear_power = m_std > 0 and m_di > m_mean + BB_STD * m_std

    bull_score = (1 if bull_rsi else 0) + (1 if bull_macd else 0) + (1 if bull_di else 0)
    bear_score = (1 if bear_rsi else 0) + (1 if bear_macd else 0) + (1 if bear_di else 0)

    return bull_score, bear_score, raw_bull_power, raw_bear_power, bull_rsi, bear_rsi, bull_macd, bear_macd, bull_di, bear_di

def get_tripwire_state(close, kin_1m, base_atr, mult, today_start, end=None):
    n = len(close) if end is None else min(int(end), len(close))
    curr_open = close[today_start]
    last_rsi, last_macd, last_adx = "Neutral", "Neutral", "Neutral"
    blocks = 0
    target = base_atr * mult
    for i in range(today_start, n):
        if close[i] - curr_open >= target or curr_open - close[i] >= target:
            _, _, _, _, b_rsi, br_rsi, b_macd, br_macd, b_di, br_di = get_kinetics(kin_1m, i)
            last_rsi = "Buy" if b_rsi else "Sell" if br_rsi else "Neutral"
            last_macd = "Buy" if b_macd else "Sell" if br_macd else "Neutral"
            last_adx = "Buy" if b_di else "Sell" if br_di else "Neutral"
            blocks += 1
            curr_open = close[i]
    return last_rsi, last_macd, last_adx, blocks

def evaluate_anchor_tripwire(close, dt_1m, kin_1m, base_atr, bb_upper, bb_lower, kc_upper, kc_lower, vwap, today_start, end=None):
    """
    Replays the day from the open up to candle index `end` (exclusive) and returns
    (surviving_anchor | None, info).

    FIX 4: `info['kills']` lists EVERY anchor killed so far (direction, anchor time/price, kill time/price,
           reason), so the merge step can say which signal was stopped, when and why.
    FIX 5: besides the soft stop (1-ATR pullback from the peak AND opposing kinetics, where the reference peak is
           re-based whenever the kinetics do not confirm) there is now an unconditional trailing floor
           HARD_STOP_ATR_MULT x ATR below the TRUE peak since the anchor. Without it a position could bleed any
           number of ATRs and still read ACTIVE. With HARD_STOP_ATR_MULT = 0 the behaviour is identical to the
           original state machine.
    """
    n = len(close) if end is None else min(int(end), len(close))
    if today_start >= n:
        return None, {'dir': 'NONE', 'anchor_time': '-', 'anchor_price': 0.0,
                      'killed_time': '-', 'killed_price': 0.0, 'reason': 'No data for today', 'kills': []}

    hard_mult = float(HARD_STOP_ATR_MULT or 0.0)
    i = today_start
    curr_open = close[today_start]
    survived_anchor = None
    last_killed_info = {'dir': 'NONE', 'anchor_time': '-', 'anchor_price': 0.0,
                        'killed_time': '-', 'killed_price': 0.0, 'reason': 'Failed Kinetic Alignment'}
    kills = []
    warzone_kills = 0

    while i < n:
        anchor = None

        while i < n:
            bull_score, bear_score, r_bull, r_bear, _, _, _, _, _, _ = get_kinetics(kin_1m, i)

            if REQUIRE_BB_KC_PIERCE:
                bb_kc_bull_fire = bb_upper[i] > kc_upper[i]
                bb_kc_bear_fire = bb_lower[i] < kc_lower[i]
            else:
                bb_kc_bull_fire = bb_kc_bear_fire = True

            if close[i] - curr_open >= base_atr:
                if bull_score >= 2 and bb_kc_bull_fire and close[i] > vwap[i]:
                    if r_bear:
                        warzone_kills += 1
                        curr_open = close[i]
                    else:
                        anchor = {"dir": "BULL", "idx": i, "time": dt_1m[i], "price": close[i]}
                        break
                else:
                    curr_open = close[i]

            elif curr_open - close[i] >= base_atr:
                if bear_score >= 2 and bb_kc_bear_fire and close[i] < vwap[i]:
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
        is_bull = anchor['dir'] == "BULL"
        peak_price = anchor['price']     # soft-stop reference: re-based when a pullback is NOT kinetically confirmed
        true_peak = anchor['price']      # FIX 5: never re-based
        j = anchor['idx'] + 1

        while j < n:
            bull_score, bear_score, _, _, _, _, _, _, _, _ = get_kinetics(kin_1m, j)
            if is_bull:
                peak_price = max(peak_price, close[j]); true_peak = max(true_peak, close[j])
                adverse, hard_adverse, opposing = peak_price - close[j], true_peak - close[j], bear_score
                soft_reason = "Kinetic SL (1 ATR Pullback)"
            else:
                peak_price = min(peak_price, close[j]); true_peak = min(true_peak, close[j])
                adverse, hard_adverse, opposing = close[j] - peak_price, close[j] - true_peak, bull_score
                soft_reason = "Kinetic SL (1 ATR Rally)"

            reason = None
            if hard_mult > 0 and hard_adverse >= hard_mult * base_atr:
                reason = f"Hard stop ({hard_mult:g} ATR trailing floor)"
            elif adverse >= base_atr:
                if opposing >= 2:
                    reason = soft_reason
                else:
                    peak_price = close[j]

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
                curr_open = close[j]
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
    elif info['dir'] == 'NONE':
        info['reason'] = "Failed Kinetic or VWAP Alignment"
    info['kills'] = kills
    return None, info

def prepare_series(master, target_dt, pinned=False):
    """
    FIX 8. Everything that does not depend on the checkpoint, computed ONCE per instrument. Every indicator
    here is causal (EWMs, rolling windows and the running VWAP only look backwards), so slicing these arrays at a
    checkpoint is exactly what recomputing on the truncated data gave -- for ~25x less work, and with ONE ATR
    (FIX 1) instead of one per checkpoint. `master` must already be cut at the snapshot (no future candles).
    Returns (series | None, reason) with reason 'no_data' or 'short_history'.
    """
    if master is None or master.empty:
        return None, "no_data"
    day_mask = (master['Datetime'].dt.date == target_dt).values
    if not day_mask.any():
        return None, "no_data"
    ts = int(np.argmax(day_mask))

    close = master['Close'].values.astype(float)
    high = master['High'].values.astype(float)
    low = master['Low'].values.astype(float)
    vol = master['Volume'].values.astype(float)
    n = len(close)

    base_atr, atr_bars = compute_base_atr(master.iloc[:ts])
    if base_atr is None:
        if not pinned:
            return None, "short_history"
        base_atr = max(float(close[ts]) * MIN_ATR_PCT, 0.01)
    elif atr_bars < ATR_MIN_BARS and not pinned:
        return None, "short_history"

    # VWAP from today's first candle. FIX 8: typical price (H+L+C)/3 -- it used the close alone.
    typ = (high + low + close) / 3.0
    cum_vol, cum_pv = np.zeros(n), np.zeros(n)
    cum_vol[ts:] = np.cumsum(vol[ts:])
    cum_pv[ts:] = np.cumsum((typ * vol)[ts:])
    with np.errstate(divide='ignore', invalid='ignore'):
        vwap = np.where(cum_vol > 0, cum_pv / cum_vol, close)

    sma20 = pd.Series(close).rolling(20, min_periods=1).mean().values
    std20 = pd.Series(close).rolling(20, min_periods=1).std(ddof=0).values
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    tr[1:] = np.maximum(high[1:] - low[1:], np.maximum(np.abs(high[1:] - close[:-1]), np.abs(low[1:] - close[:-1])))
    atr20 = pd.Series(tr).rolling(20, min_periods=1).mean().values

    return {
        'close': close, 'dt': master['Datetime'].values, 'today_start': ts,
        'day_open': float(master['Open'].iloc[ts]), 'prev_close': float(close[ts - 1]) if ts > 0 else 0.0,
        'base_atr': float(base_atr), 'atr_bars': int(atr_bars),
        'kin': _evaluate_kinetic_arrays(close, high, low),
        'vwap': vwap,
        'bb_upper': sma20 + 2.0 * std20, 'bb_lower': sma20 - 2.0 * std20,
        'kc_upper': sma20 + 1.5 * atr20, 'kc_lower': sma20 - 1.5 * atr20,
    }, None

def row_at(S, end, symbol):
    """The row for ONE checkpoint: the replay of the day from the open up to candle index `end` (exclusive)."""
    ts = S['today_start']
    end = min(int(end), len(S['close']))
    if end <= ts:
        return None
    close, base_atr, kin = S['close'], S['base_atr'], S['kin']
    last = end - 1
    ltp = float(close[last])
    # FIX 8: Day% from the previous close (what a broker shows); falls back to today's open
    ref = S['prev_close'] if (DAY_PCT_BASIS == "PREV_CLOSE" and S['prev_close'] > 0) else S['day_open']
    day_pct = ((ltp - ref) / ref) * 100 if ref else 0.0

    row = {
        'Symbol': symbol, 'LTP': ltp, 'DayChangePct': float(day_pct),
        'ActiveAnchor': None, 'AnchorPrice': 0.0, 'AnchorDir': "NONE", 'MovePct': 0.0,
        'State': "NONE", 'Blocks': 0, 'KilledTime': '-', 'KilledPrice': 0.0, 'RejectReason': "",
        'Kills': [], 'AtrBars': S['atr_bars'], 'BaseAtr': base_atr,
    }

    anchor, info = evaluate_anchor_tripwire(close, S['dt'], kin, base_atr, S['bb_upper'], S['bb_lower'],
                                            S['kc_upper'], S['kc_lower'], S['vwap'], ts, end)
    row['Kills'] = info.get('kills', [])

    # FIX 4: the R-M-D triads are computed for EVERY row, so stopped / graveyard rows keep their indicators
    for mult in HA_ATR_MULTIPLIERS:
        rsi_st, macd_st, adx_st, blk_count = get_tripwire_state(close, kin, base_atr, mult, ts, end)
        gtag = f"{mult}X"
        row[f'BB_RSI_{gtag}'] = rsi_st
        row[f'BB_MACD_{gtag}'] = macd_st
        row[f'ADX_{gtag}'] = adx_st
        if mult == 1:
            row['Blocks'] = blk_count

    if anchor:
        row['ActiveAnchor'] = pd.to_datetime(anchor['time']).strftime("%H:%M")
        row['AnchorPrice'] = float(anchor['price'])
        row['AnchorDir'] = anchor['dir']
        row['MovePct'] = ((ltp - anchor['price']) / anchor['price']) * 100 if anchor['price'] else 0.0
        if anchor['dir'] == "BULL":
            row['State'] = "[ACTIVE BUY]" if S['bb_upper'][last] > S['kc_upper'][last] else "[COILING]"
        else:
            row['State'] = "[ACTIVE SELL]" if S['bb_lower'][last] < S['kc_lower'][last] else "[COILING]"
    else:
        row['AnchorDir'] = info.get('dir', 'NONE')
        row['ActiveAnchor'] = info.get('anchor_time', '-')
        row['AnchorPrice'] = float(info.get('anchor_price', 0.0))
        row['KilledTime'] = info.get('killed_time', '-')
        row['KilledPrice'] = float(info.get('killed_price', 0.0))
        row['RejectReason'] = info.get('reason', 'Failed Kinetic Alignment')
        if row['AnchorPrice'] > 0 and row['KilledPrice'] > 0:
            row['MovePct'] = ((row['KilledPrice'] - row['AnchorPrice']) / row['AnchorPrice']) * 100
    return row

def compute_row(symbol, master_1m, target_dt):
    """Single-snapshot convenience wrapper (kept for callers of the old API): the row at the END of master_1m."""
    S, _ = prepare_series(master_1m, target_dt, pinned=True)
    return None if S is None else row_at(S, len(master_1m), symbol)

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
        if item.get('pinned'):
            return item                  # FIX 7: a tracked instrument is never filtered out
        frames = []
        for c_start, c_end in _date_chunks(start, end + timedelta(days=1), span=365):
            status, df = _candles(_url_range_daily(item['key'], c_start, c_end))
            if status == 200 and df is not None: frames.append(df)
        if not frames: return None
        master = prepare_master(frames)
        # FIX 2: judge the last COMPLETED session before the snapshot date. Same-day data is wrong twice over:
        # a partial day's volume fails the threshold, and in a backtest the full day leaks the future.
        master = master[master['Datetime'].dt.date < end]
        if master.empty: return None
        close, vol = master['Close'].iloc[-1], master['Volume'].iloc[-1]
        return item if (MIN_PRICE <= close <= MAX_PRICE and vol >= MIN_DAILY_VOLUME) else None
    except Exception as e:
        ERRORS.add(f"prefilter:{item.get('symbol')}", e)
        return None
    finally:
        progress.tick()

def process_stock_checkpoints(args):
    """
    One instrument -> one (checkpoint, status, payload) tuple per checkpoint.
    status: ok | no_data | illiquid | short_history | error   (payload = row for ok, the symbol otherwise)
    """
    item, checkpoints, use_intraday, history_days, mode, progress = args
    sym = item['symbol']
    pinned = bool(item.get('pinned'))
    try:
        final_cutoff = checkpoints[-1]
        target_dt = final_cutoff.date()
        start_dt = target_dt - timedelta(days=history_days)
        frames = []
        for c_start, c_end in _date_chunks(start_dt, target_dt + timedelta(days=1), span=7):
            status, df = _candles(_url_range_1m(item['key'], c_start, c_end))
            if status == 200 and df is not None: frames.append(df)

        if use_intraday:         # FIX 3: any snapshot that falls on TODAY needs the intraday feed, live or not
            today_df = fetch_today(item['key'])
            if today_df is not None: frames.append(today_df)

        if not frames:
            return [(cp, "no_data", sym) for cp in checkpoints]

        # FIX 2: only candles that CLOSED before the snapshot (a candle stamped HH:MM is still forming at HH:MM)
        master = prepare_master(frames)
        master = master[master['Datetime'] < final_cutoff].reset_index(drop=True)

        if mode == "INDEX_OPTIONS" and not pinned:
            # FIX 6: liquidity is judged ONCE, at the snapshot, not on the cumulative volume of each checkpoint
            day = master[master['Datetime'].dt.date == target_dt]
            if day.empty:
                return [(cp, "no_data", sym) for cp in checkpoints]
            if day['Close'].iloc[-1] < OPT_MIN_PRICE or day['Volume'].sum() < OPT_MIN_VOLUME:
                return [(cp, "illiquid", sym) for cp in checkpoints]

        S, why = prepare_series(master, target_dt, pinned)
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
        print(f"{COLOR_YELLOW}⚠ {STATS.failed} API request(s) failed after retries. Samples: {'; '.join(STATS.samples)}{COLOR_RESET}")
    if ERRORS.items:
        print(f"{COLOR_YELLOW}⚠ {len(ERRORS.items)} instrument(s) raised exceptions. First one:{COLOR_RESET}")
        where, msg, tb = ERRORS.items[0]
        print(f"{COLOR_DIM}   [{where}] {msg}\n{tb}{COLOR_RESET}")

# ------------------------------------------------------------------------------
# FIX 4: backtrace merge -- every anchor is its own signal
# ------------------------------------------------------------------------------
ACTIVE_BULL_STATES = ("[ACTIVE BUY]", "[COILING]")
ACTIVE_BEAR_STATES = ("[ACTIVE SELL]", "[COILING]")
LIVE_STATES = ("[ACTIVE BUY]", "[ACTIVE SELL]", "[COILING]")

def _signal_id(row):
    return (row['AnchorDir'], row['ActiveAnchor'], round(float(row['AnchorPrice']), 6))

def _find_kill(kills, direction, anchor_time):
    for k in reversed(kills or []):
        if k.get('dir') == direction and k.get('anchor_time') == anchor_time:
            return k
    return None

def _merge_signal(sig_rows, last_row, direction):
    """
    sig_rows: the checkpoint rows in which ONE specific anchor was alive. 'Seen' is the first of them.
    LTP / Day% / indicators come from the latest checkpoint. The signal is alive only if that same anchor is
    still alive at the latest checkpoint; otherwise it is STOPPED and the kill (time / price / reason) is
    attached. The old merge looked only at the latest checkpoint's DIRECTION, so a killed anchor was shown
    ACTIVE whenever a later anchor in the same direction had formed (with the earlier anchor's time and price).
    """
    first, latest = sig_rows[0], sig_rows[-1]
    row = dict(last_row)
    row['CheckpointTime'] = first['CheckpointTime']
    row['ActiveAnchor'] = first['ActiveAnchor']
    row['AnchorPrice'] = first['AnchorPrice']
    row['AnchorDir'] = direction
    ap = first['AnchorPrice'] or last_row['LTP']
    row['MovePct'] = ((last_row['LTP'] - ap) / ap) * 100 if ap else 0.0
    kills = last_row.get('Kills') or []
    row['PriorStops'] = sum(1 for k in kills if k.get('dir') == direction and k.get('anchor_time') != first['ActiveAnchor'])

    if latest['CheckpointTime'] == last_row['CheckpointTime']:
        row['State'] = last_row['State']
        row['KilledTime'], row['KilledPrice'], row['RejectReason'] = '-', 0.0, ""
    else:
        row['State'] = "[STOPPED]"
        k = _find_kill(kills, direction, first['ActiveAnchor'])
        if k:
            row['KilledTime'], row['KilledPrice'], row['RejectReason'] = k['killed_time'], float(k['killed_price']), k['reason']
        else:
            row['KilledTime'], row['KilledPrice'], row['RejectReason'] = '-', 0.0, "Anchor no longer valid at the latest checkpoint"
    return row

def merge_symbol_rows(rows):
    """rows: one row per checkpoint for ONE instrument. Returns (bull_row | None, bear_row | None, graveyard_row | None)."""
    rows = sorted(rows, key=lambda r: r['CheckpointTime'])
    last_row = rows[-1]
    merged = {}
    for direction, active in (("BULL", ACTIVE_BULL_STATES), ("BEAR", ACTIVE_BEAR_STATES)):
        d_rows = [r for r in rows if r['AnchorDir'] == direction and r['State'] in active]
        if not d_rows:
            continue
        sid = _signal_id(d_rows[-1])                     # the most recent signal of this direction
        merged[direction] = _merge_signal([r for r in d_rows if _signal_id(r) == sid], last_row, direction)
    grave = None
    if not merged:
        grave = dict(last_row)
        grave['CheckpointTime'] = rows[0]['CheckpointTime']
    return merged.get("BULL"), merged.get("BEAR"), grave

def _state_display(row):
    st = row['State']
    if st == "[STOPPED]":
        k = row.get('KilledTime')
        return f"[STOPPED {k}]" if k and k != '-' else st
    return st

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
        print(f"   {COLOR_YELLOW}[track] could not read state file '{path}' ({type(e).__name__}: {e}) -- starting fresh.{COLOR_RESET}")
        return {}

def _save_watch_state(path, state):
    try:
        with open(path, 'w') as f:
            json.dump(state, f, indent=2, default=str)
    except Exception as e:
        print(f"   {COLOR_YELLOW}[track] could not write state file '{path}' ({type(e).__name__}: {e}).{COLOR_RESET}")

def _send_watch_email(subject, body, tag="watch"):
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
        print(f"   {COLOR_YELLOW}[{tag}] cannot send email -- missing env var(s): {', '.join(missing)}.{COLOR_RESET}")
        return False

    try:
        port = int(smtp_port)
    except ValueError:
        print(f"   {COLOR_YELLOW}[{tag}] cannot send email -- SMTP_PORT='{smtp_port}' is not a valid integer.{COLOR_RESET}")
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
        print(f"   {COLOR_GREEN_FG}[{tag}] email sent to {recipient}.{COLOR_RESET}")
        return True
    except Exception as e:
        print(f"   {COLOR_YELLOW}[{tag}] failed to send email ({type(e).__name__}: {e}).{COLOR_RESET}")
        return False

def _read_order_env():
    for name in dict.fromkeys((ORDER_PLACED_ENV, ORDER_PLACED_ENV.upper(), "ORDER_PLACED")):
        v = (os.environ.get(name) or "").strip()
        if v:
            return v
    return ""

def _status_of(row):
    """(key, text). `key` carries no prices, so a drifting LTP never looks like a status change."""
    st, d = row.get('State', 'NONE'), row.get('AnchorDir', 'NONE')
    if st in LIVE_STATES:
        return f"{d} LIVE", f"{d} LIVE {st}"
    if st == "[STOPPED]":
        k = row.get('KilledTime', '-')
        return f"{d} STOPPED", (f"{d} STOPPED at {k}" if k and k != '-' else f"{d} STOPPED")
    return "REJECTED", f"REJECTED ({row.get('RejectReason') or 'unknown'})"

def resolve_track(value, all_bulls, all_bears, rejected, unevaluated):
    """
    FIX 7. Look a tracked value up in this run's results. Case- and whitespace-insensitive substring match
    ('NIFTY25000CE' finds 'NIFTY 25000 CE 14 OCT 26'). When several symbols match, an EXACT symbol wins, then a
    LIVE one, then a STOPPED one, then the shortest name -- the old lookup returned the first hit in the order
    bull / bear / graveyard, which could be a stopped bull signal while the bear signal was live, or simply the
    wrong strike. A symbol can sit in both baskets: every entry of the winning symbol is reported.
    """
    n = _norm_sym(value)
    hits = {}
    for rows in (all_bulls, all_bears, rejected):
        for r in rows:
            if n and n in _norm_sym(r['Symbol']):
                hits.setdefault(r['Symbol'], []).append(r)

    if not hits:
        for sym, why in unevaluated.items():
            if n and n in _norm_sym(sym):
                return {"key": "NO DATA", "text": f"NO DATA ({why})", "symbol": sym, "entries": [], "others": []}
        return {"key": "NOT FOUND", "text": "NOT FOUND (no instrument in this run's universe matches)",
                "symbol": value, "entries": [], "others": []}

    def rank(sym):
        rows = hits[sym]
        return (_norm_sym(sym) != n,
                not any(r.get('State') in LIVE_STATES for r in rows),
                not any(r.get('State') == "[STOPPED]" for r in rows),
                len(sym), sym)
    order = sorted(hits, key=rank)
    best = order[0]
    entries = sorted(hits[best], key=lambda r: (r.get('State') == 'NONE', r.get('AnchorDir') != 'BULL'))
    statuses = [_status_of(r) for r in entries]
    return {"key": " + ".join(k for k, _ in statuses), "text": " | ".join(t for _, t in statuses),
            "symbol": best, "entries": entries, "others": order[1:]}

def _entry_brief(row):
    return (f"{row['Symbol']}: {_status_of(row)[1]} | LTP {row.get('LTP', 0):.2f} | "
            f"Day {row.get('DayChangePct', 0):+.2f}% | Move {row.get('MovePct', 0):+.2f}%")

def _entry_lines(row):
    key, text = _status_of(row)
    lines = [f"Symbol : {row['Symbol']}", f"Status : {text}",
             f"LTP    : {row.get('LTP', 0):.2f}    Day: {row.get('DayChangePct', 0):+.2f}%    "
             f"Move from anchor: {row.get('MovePct', 0):+.2f}%"]
    if row.get('AnchorPrice', 0) > 0:
        seen = row.get('CheckpointTime')
        tail = f"    first seen {seen:%H:%M}" if (key != "REJECTED" and hasattr(seen, 'strftime')) else ""
        lines.append(f"Anchor : {row.get('ActiveAnchor', '-')} @ {row['AnchorPrice']:.2f}{tail}")
    if row.get('KilledTime', '-') not in ('-', '', None):
        lines.append(f"Stopped: {row['KilledTime']} @ {row.get('KilledPrice', 0):.2f} -- {row.get('RejectReason', '')}")
    elif key == "REJECTED":
        lines.append(f"Reason : {row.get('RejectReason') or 'unknown'}")
    if row.get('PriorStops'):
        lines.append(f"Earlier {row['AnchorDir']} signals stopped out today: {row['PriorStops']}")
    return lines

TRACK_CFG = {
    "watch": {"icon": "👁 ", "label": "WATCH", "tag": "watch", "word": "Watch"},
    "order": {"icon": "📦", "label": "ORDER PLACED", "tag": "order", "word": "Order status"},
}

def run_tracking(kind, values, mode, cutoff_dt, all_bulls, all_bears, rejected, unevaluated, can_email, notes=()):
    """
    kind 'watch': emails only when a value's status CHANGED since the last run (state file).
    kind 'order': emails on EVERY run (one digest for all positions), or only on change when
                  ORDER_EMAIL_ONLY_ON_CHANGE / --order-email-only-on-change is set.
    FIX 7: a value is marked as reported only after the email was actually delivered (it used to be recorded
           even when SMTP failed, so the alert was lost for good); time-machine runs never email.
    """
    if not values:
        return
    cfg = TRACK_CFG[kind]
    only_on_change = True if kind == "watch" else bool(ORDER_EMAIL_ONLY_ON_CHANGE)
    state_file = WATCH_STATE_FILE if kind == "watch" else ORDER_STATE_FILE
    state = _load_watch_state(state_file) if (only_on_change and can_email) else {}

    results = []
    for value in values:
        res = resolve_track(value, all_bulls, all_bears, rejected, unevaluated)
        nk = _norm_sym(value)
        prev = state.get(nk, {}).get("status")
        results.append({"value": value, "res": res, "prev": prev, "changed": prev != res['key'], "nk": nk})

        print(f"\n{COLOR_BOLD}{cfg['icon']} {cfg['label']} [{value}] -> {res['text']}{COLOR_RESET}" +
              (f" {COLOR_DIM}(previous: {prev}){COLOR_RESET}" if prev else ""))
        for r in res['entries']:
            print(f"   {COLOR_DIM}{_entry_brief(r)}{COLOR_RESET}")
        if res['others']:
            shown = ", ".join(res['others'][:5]) + (" ..." if len(res['others']) > 5 else "")
            print(f"   {COLOR_YELLOW}⚠ '{value}' also matches {len(res['others'])} other symbol(s): {shown}. "
                  f"Tracking '{res['symbol']}' -- use a more specific value if that is wrong.{COLOR_RESET}")

    if not can_email:
        print(f"   {COLOR_DIM}[{cfg['tag']}] time-machine snapshot -- no email (it would not describe the live position).{COLOR_RESET}")
        return
    due = [r for r in results if r['changed'] or not only_on_change]
    if not due:
        print(f"   {COLOR_DIM}[{cfg['tag']}] status unchanged since last run -- no email sent.{COLOR_RESET}")
        return

    if len(due) == 1:
        r = due[0]
        if not only_on_change:
            subject = f"[System3] {cfg['word']}: {r['value']} -> {r['res']['key']}"
        elif r['prev'] is None:
            subject = f"[System3] {cfg['word']} started: {r['value']} is {r['res']['key']}"
        else:
            subject = f"[System3] {r['value']} changed: {r['prev']} -> {r['res']['key']}"
    else:
        subject = f"[System3] {cfg['word']}: " + "; ".join(f"{r['value']} -> {r['res']['key']}" for r in due)
        if len(subject) > 150:
            subject = subject[:147] + "..."

    head = [f"Snapshot : {cutoff_dt:%Y-%m-%d %H:%M}  (live run)", f"Mode     : {mode}"] + [f"Note     : {n}" for n in notes]
    parts = ["\n".join(head)]
    for r in due:
        block = [f"== {r['value']}  ->  {r['res']['text']}"]
        if only_on_change and r['prev'] is not None:
            block.append(f"   previous status: {r['prev']}")
        for e in r['res']['entries']:
            block += ["   " + ln for ln in _entry_lines(e)] + [""]
        if r['res']['others']:
            block.append("   Also matches: " + ", ".join(r['res']['others'][:8]))
        parts.append("\n".join(block).rstrip())
    body = subject + "\n\n" + "\n\n".join(parts) + "\n"

    sent = _send_watch_email(subject, body, cfg['tag'])
    if only_on_change:
        if sent:
            stamp = cutoff_dt.strftime("%Y-%m-%d %H:%M:%S")
            for r in due:
                state[r['nk']] = {"status": r['res']['key'], "checked_at": stamp}
            _save_watch_state(state_file, state)
        else:
            print(f"   {COLOR_YELLOW}[{cfg['tag']}] email not delivered -- the status is NOT marked as reported and will be retried next run.{COLOR_RESET}")

def _fav_color(pct, direction):
    """Green = the move is in favour of the signal (a falling price is GOOD for a BEAR setup)."""
    sign = 1 if direction == "BULL" else -1 if direction == "BEAR" else 0
    fav = pct * sign
    return COLOR_GREEN_FG if fav > 0 else COLOR_RED_FG if fav < 0 else COLOR_DIM

def run_screener(mode=TRADING_MODE, days=BACKTRACE_DAYS, target_date_str=None, target_time_str=None, options_master_path=""):
    t_start = time.time()

    # FIX 3: --time without --date means "today at that time" (it used to be silently ignored)
    explicit = bool(target_date_str or target_time_str)
    if explicit:
        d_str = target_date_str or now_ist().strftime("%Y-%m-%d")
        t_str = target_time_str or "15:30"
        try:
            cutoff_dt = datetime.strptime(f"{d_str} {t_str}", "%Y-%m-%d %H:%M")
        except ValueError:
            print(f"{COLOR_RED_FG}[!] Invalid --date/--time '{d_str} {t_str}' (expected YYYY-MM-DD and HH:MM).{COLOR_RESET}")
            return 1
    else:
        cutoff_dt = now_ist()
    cutoff_dt = cutoff_dt.replace(second=0, microsecond=0)
    live_run = not explicit

    # FIX 3: weekend / holiday / pre-open / future snapshots roll back to the last completed session
    cutoff_dt, notes = resolve_session(cutoff_dt)
    if STATS.auth_failed:
        print(f"{COLOR_RED_FG}[!] Upstox rejected the access token (HTTP 401). Regenerate UPSTOX_ACCESS_TOKEN -- it expires daily.{COLOR_RESET}")
        return 1
    target_dt = cutoff_dt.date()
    cutoff_dt = min(cutoff_dt, _close_dt(target_dt))
    today_session = (target_dt == now_ist().date())

    print(f"\n{COLOR_CYAN}📡 Initializing Tracker [{mode}] | Time Machine: {cutoff_dt.strftime('%Y-%m-%d %H:%M:%S')} (Live: {live_run}){COLOR_RESET}")
    for n in notes:
        print(f"   {COLOR_YELLOW}» {n}{COLOR_RESET}")

    watch_values = parse_track_values(WATCH_VALUE)
    order_values = parse_track_values(_read_order_env())
    needles = watch_values + order_values

    universe_raw = get_dynamic_universe(mode, cutoff_dt, today_session, options_master_path, needles)
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
                                  [(item, checkpoints, today_session, MIN1_HISTORY_DAYS, mode, prog) for item in candidates]))
    prog.done()

    if STATS.auth_failed:
        print(f"{COLOR_RED_FG}[!] Upstox rejected the access token (HTTP 401) while fetching candles.{COLOR_RESET}")
        return 1

    outcomes = [o for sym_outcomes in per_symbol for o in sym_outcomes]
    results = [p for _, s, p in outcomes if s == "ok"]
    counts = {k: sum(1 for _, s, _ in outcomes if s == k) for k in ("no_data", "illiquid", "short_history", "error")}
    print(f"   {COLOR_DIM}» Scan summary: {len(candidates)} instruments x {len(checkpoints)} checkpoints = "
          f"{len(outcomes)} evaluations | ok {len(results)} | no data {counts['no_data']} | illiquid {counts['illiquid']} | "
          f"short history {counts['short_history']} | errors {counts['error']}{COLOR_RESET}")
    if counts["short_history"]:
        print(f"   {COLOR_YELLOW}» {counts['short_history']} evaluations skipped: fewer than {ATR_MIN_BARS} completed sessions of "
              f"history for the ATR (new contracts, or --history-days too small).{COLOR_RESET}")

    if not results:
        print(f"{COLOR_RED_FG}[!] No instrument produced usable data at any checkpoint up to {cutoff_dt:%Y-%m-%d %H:%M}. "
              f"Is that a trading day/time, and does the token have historical-candle access?{COLOR_RESET}")
        _print_error_summary()
        return 1

    # why an instrument that was looked at has no row (used by the tracking lookup: NO DATA vs NOT FOUND)
    why_text = {"no_data": "no candles for this session", "illiquid": "below the price/volume floor",
                "short_history": "too little history for an ATR", "error": "processing error"}
    ok_symbols = {r['Symbol'] for r in results}
    unevaluated = {}
    for _, s, p in outcomes:
        if s != "ok" and p not in ok_symbols:
            unevaluated.setdefault(p, set()).add(why_text.get(s, s))
    unevaluated = {sym: ", ".join(sorted(v)) for sym, v in unevaluated.items()}

    by_symbol = {}
    for row in results:
        by_symbol.setdefault(row['Symbol'], []).append(row)

    bulls, bears, rejected = [], [], []
    for symbol, rows in by_symbol.items():
        b, r_, g = merge_symbol_rows(rows)
        if b: bulls.append(b)
        if r_: bears.append(r_)
        if g: rejected.append(g)

    true_bull_count = len(bulls)
    true_bear_count = len(bears)
    bull_live = sum(1 for r in bulls if r['State'] != "[STOPPED]")
    bear_live = sum(1 for r in bears if r['State'] != "[STOPPED]")

    bulls.sort(key=lambda r: (r['State'] == "[STOPPED]", r['CheckpointTime'], r['State'] != "[ACTIVE BUY]", -r['DayChangePct']))
    bears.sort(key=lambda r: (r['State'] == "[STOPPED]", r['CheckpointTime'], r['State'] != "[ACTIVE SELL]", r['DayChangePct']))

    all_bulls, all_bears = list(bulls), list(bears)      # the tracking lookup searches EVERYTHING, not just the top N

    bulls = bulls[:TOP_N_BUYERS]
    bears = bears[:TOP_N_SELLERS]

    def sort_reason(r):
        reason = r['RejectReason']
        if "Kinetic SL" in reason or "Hard stop" in reason: return 0
        if "Warzone" in reason: return 1
        if "Alignment" in reason: return 2
        return 3

    rejected.sort(key=lambda x: (sort_reason(x), -abs(x['DayChangePct'])))
    grave_total = len(rejected)
    grave_shown = rejected if TOP_N_GRAVEYARD <= 0 else rejected[:TOP_N_GRAVEYARD]

    bull_count_str = f"{len(bulls)} (of {true_bull_count})" if true_bull_count > TOP_N_BUYERS else f"{true_bull_count}"
    bear_count_str = f"{len(bears)} (of {true_bear_count})" if true_bear_count > TOP_N_SELLERS else f"{true_bear_count}"

    shown = bulls + bears + grave_shown
    sym_w = max(12, min(MAX_SYMBOL_WIDTH, max((len(r['Symbol']) for r in shown), default=12)))

    print(f"\n{COLOR_BOLD}=== STATEFUL INSTITUTIONAL VOLATILITY TRACKER [{mode}] ==={COLOR_RESET}")
    print(f"Target Snapshot: {cutoff_dt.strftime('%Y-%m-%d %H:%M')} | Scanned: {len(by_symbol)} symbols "
          f"across {len(checkpoints)} checkpoints | After dedup: {bull_count_str} bull ({bull_live} live), "
          f"{bear_count_str} bear ({bear_live} live), {grave_total} graveyard\n")

    def print_basket(title, icon, data_list):
        if not data_list: return
        print(f"\n{COLOR_BOLD}{icon} {title}{COLOR_RESET}")
        header_str = (
            f" {COLOR_CYAN}{'Script':<{sym_w}} {'LTP':>8} {'Day%':>7} | "
            f"{'Seen':^6} {'Anchor':^6} {'Anch Val':>9} {'Move%':>7} | "
            f"{'1X (R-M-D)':^11}  {'2X (R-M-D)':^11}  {'3X (R-M-D)':^11}  {'5X (R-M-D)':^11} | "
            f"{'State':^16}{COLOR_RESET}"
        )
        print(header_str)
        print("-" * len(ANSI_RE.sub("", header_str)))

        for row in data_list:
            day_pct_str = f"{row['DayChangePct']:>+6.2f}%"
            day_color = COLOR_GREEN_FG if row['DayChangePct'] > 0 else COLOR_RED_FG if row['DayChangePct'] < 0 else COLOR_DIM
            move_pct_str = f"{row['MovePct']:>+6.2f}%"
            move_color = _fav_color(row['MovePct'], row['AnchorDir'])

            k1 = format_kinetic_triad(row.get('BB_RSI_1X'), row.get('BB_MACD_1X'), row.get('ADX_1X'))
            k2 = format_kinetic_triad(row.get('BB_RSI_2X'), row.get('BB_MACD_2X'), row.get('ADX_2X'))
            k3 = format_kinetic_triad(row.get('BB_RSI_3X'), row.get('BB_MACD_3X'), row.get('ADX_3X'))
            k5 = format_kinetic_triad(row.get('BB_RSI_5X'), row.get('BB_MACD_5X'), row.get('ADX_5X'))

            state = row['State']
            shown_state = _state_display(row)
            if "BUY]" in state: state_text = f"{COLOR_GREEN_FG}{shown_state:^16}{COLOR_RESET}"
            elif "SELL]" in state: state_text = f"{COLOR_RED_FG}{shown_state:^16}{COLOR_RESET}"
            else: state_text = f"{COLOR_YELLOW}{shown_state:^16}{COLOR_RESET}"

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

    if grave_shown:
        print(f"\n{COLOR_BOLD}🚫 THE GRAVEYARD (Filtered / Rejected){COLOR_RESET}")
        # FIX 4: no 'Seen' column here -- a rejected symbol never had a surviving checkpoint to be 'seen' at
        header_str = (
            f" {COLOR_CYAN}{'Script':<{sym_w}} {'Signal':^6} {'Anchor':^6} {'Anch Val':>9} "
            f"{'Killed':^6} {'Kill Val':>9} {'LTP':>8} {'Day%':>7} {'Move%':>7} | "
            f"{'Forensic Rejection Reason'}{COLOR_RESET}"
        )
        print(header_str)
        print("-" * len(ANSI_RE.sub("", header_str)))

        for row in grave_shown:
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
                move_color = _fav_color(row['MovePct'], sig)
            else:
                move_pct_str, move_color = f"{'-':>7}", COLOR_DIM

            reason = row['RejectReason']
            if "Kinetic SL" in reason or "Hard stop" in reason or "Pullback" in reason or "Rally" in reason: reason_color = COLOR_YELLOW
            elif "Chop" in reason or "Warzone" in reason: reason_color = COLOR_RED_FG
            else: reason_color = COLOR_DIM

            print(
                f" {COLOR_BOLD}{_clip(row['Symbol'], sym_w):<{sym_w}}{COLOR_RESET} "
                f"{sig_colored} {a_time:^6} {a_price} {k_time:^6} {k_price} "
                f"{row['LTP']:>8.2f} {day_color}{day_pct_str}{COLOR_RESET} "
                f"{move_color}{move_pct_str}{COLOR_RESET} | "
                f"{reason_color}{reason}{COLOR_RESET}"
            )
        if grave_total > len(grave_shown):
            print(f"{COLOR_DIM}   ... {grave_total - len(grave_shown)} more graveyard row(s) not shown "
                  f"(--graveyard-max 0 prints them all; --watch / OrderPlaced still search every one).{COLOR_RESET}")

    # --watch / OrderPlaced: a time-machine snapshot is printed but never emailed
    can_email = live_run
    run_tracking("watch", watch_values, mode, cutoff_dt, all_bulls, all_bears, rejected, unevaluated, can_email, notes)
    run_tracking("order", order_values, mode, cutoff_dt, all_bulls, all_bears, rejected, unevaluated, can_email, notes)

    _print_error_summary()
    total_calls = sum(l.total_calls for l in LIMITERS.values())
    print(f"\n⏱️ Tracker sync completed in {(time.time() - t_start):.2f} seconds ({total_calls} API calls).\n")
    return 0


def parse_args():
    p = argparse.ArgumentParser(description="Strict institutional volatility tracker (Upstox)")
    p.add_argument("--mode", choices=["STOCK_FNO", "CASH_EQUITY", "INDEX_OPTIONS"], default=TRADING_MODE)
    p.add_argument("--days", type=int, default=BACKTRACE_DAYS, help="calendar-day lookback of the daily prefilter fetch (x2 + 6)")
    p.add_argument("--date", type=str, default=None, help="Target date (YYYY-MM-DD). A weekend/holiday rolls back to the last session.")
    p.add_argument("--time", type=str, default=None,
                   help="Target time (HH:MM). With --date the default is 15:30; WITHOUT --date it means 'today at that time'.")
    p.add_argument("--history-days", type=int, default=MIN1_HISTORY_DAYS,
                   help=f"Calendar days of 1-minute history for indicator math and the ATR (default: {MIN1_HISTORY_DAYS}; "
                        f"the ATR needs at least {ATR_MIN_BARS} completed sessions).")
    gate = p.add_mutually_exclusive_group()
    gate.add_argument("--disable-bb-kc-gate", action="store_true",
                      help="Drop the 'Bollinger Band already pierced Keltner Channel' requirement (already off by default).")
    gate.add_argument("--require-bb-kc-pierce", action="store_true",
                      help="Turn the 'Bollinger Band already pierced Keltner Channel' anchor requirement ON.")
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
    p.add_argument("--hard-stop-atr", type=float, default=HARD_STOP_ATR_MULT,
                   help=f"Unconditional trailing floor, in ATRs below the true peak since the anchor (default {HARD_STOP_ATR_MULT:g}; 0 disables it).")
    p.add_argument("--day-pct-basis", choices=["PREV_CLOSE", "OPEN"], default=DAY_PCT_BASIS,
                   help=f"Day%% is measured from the previous close (broker-style) or from today's open. Default {DAY_PCT_BASIS}.")
    p.add_argument("--graveyard-max", type=int, default=TOP_N_GRAVEYARD,
                   help=f"Max graveyard rows printed (default {TOP_N_GRAVEYARD}; 0 = all). Tracking searches every row regardless.")
    p.add_argument("--watch", type=str, default="",
                   help="Optional. A symbol, tradingsymbol, or strike-bearing string to monitor (e.g. 'RELIANCE', "
                        "'NIFTY25000CE', or just '25000' -- matched as a case- and whitespace-insensitive substring of the "
                        "symbol; several values may be separated by commas). The matching instrument is always scanned, "
                        "even if the price/volume prefilter or the ATM window would drop it. "
                        "When set, this run checks that symbol's status and emails an alert via "
                        "SENDER_EMAIL/EMAIL_SENDER, SENDER_PASSWORD/EMAIL_APP_PWD, RECIPIENT_EMAIL/EMAIL_RECEIVER "
                        "(SMTP_SERVER/SMTP_PORT optional, default smtp.gmail.com:465) ONLY if the status changed "
                        "since the last run (tracked in system3_watch_state.json). If this flag is omitted "
                        "(default: no watch), no email code runs at all.")
    p.add_argument("--order-email-only-on-change", action="store_true",
                   help="Switch OrderPlaced tracking (see the OrderPlaced env var) from 'email every run' to "
                        "'email only when the status changes since the last run', using its own state file "
                        f"({ORDER_STATE_FILE}) so it never shares state with --watch.")
    args = p.parse_args()
    if args.checkpoint_min < 1: p.error("--checkpoint-min must be >= 1 (0 or a negative value used to hang the run)")
    if args.strikes_up < 0 or args.strikes_down < 0: p.error("--strikes-up / --strikes-down must be >= 0")
    if args.history_days < 1: p.error("--history-days must be >= 1")
    if args.hard_stop_atr < 0: p.error("--hard-stop-atr must be >= 0")
    if args.graveyard_max < 0: p.error("--graveyard-max must be >= 0")
    return args

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
    HARD_STOP_ATR_MULT = args.hard_stop_atr
    DAY_PCT_BASIS = args.day_pct_basis
    TOP_N_GRAVEYARD = args.graveyard_max
    WATCH_VALUE = args.watch
    if args.order_email_only_on_change:
        ORDER_EMAIL_ONLY_ON_CHANGE = True
    if args.single_snapshot:
        BACKTRACE_CHECKPOINTS = False
    if args.require_bb_kc_pierce:
        REQUIRE_BB_KC_PIERCE = True
    elif args.disable_bb_kc_gate:
        REQUIRE_BB_KC_PIERCE = False
    try:
        code = run_screener(args.mode, args.days, args.date, args.time, args.options_master)
    except Exception:
        print(f"\n{COLOR_RED_FG}💥 FATAL: unhandled exception -- full traceback follows.{COLOR_RESET}")
        traceback.print_exc()
        code = 1
    sys.exit(code)
