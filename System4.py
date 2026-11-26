#!/usr/bin/env python3
"""
Strict Institutional Volatility Tracker (Upstox) - System 3.0 (Consolidated Audited Build)
+ Implied Volatility (IV) & IV Crush Engine: Real-time bar-by-bar IV derivation via Black-Scholes inversion for options (session-isolated/gap-stitched IV bands, causal median IV without lookahead).
+ Dual-Confluence Option Anchor Gate: Option buyers require concurrent Price breakout ([BB > KC]) and non-collapsing IV (vel_iv >= 0).
+ Dynamic Kinetic SL (IV Crush): Kills live option setups if IV drops >15% from post-anchor peak or suffers severe negative velocity shock.
+ Multi-Timeframe IV Grids: Seamlessly integrates IV velocity and IV Bollinger Band pierces into ATR and BB grids (0 to 5) for options.
+ Configurable Timestamp Gap Check: Filters stale Pure 1st-Time setups where (Seen - Anchor) > MAX_TIMESTAMP_GAP_MINUTES without misclassifying active Stage 2 setups into the Graveyard.
+ Risk-to-Reward (R:R) Column: Formatted like 0.1:1, 0.28:1 before LTP/Exit, sorted ascending for Bulls and Bears.
+ Dedicated Still-Holding Good Email: Sends a separate email for active setups that triggered earlier and still hold good at snapshot time (deduplicated from fresh Pure 1st-Time popups by default).
+ Sectoral Indices in Pure 1st-Time Email: Sectoral index popups are seamlessly included alongside stock popups in the Pure 1st-Time email (and sent even if only indices pop).
+ Target Ladder: once T1 is achieved the target steps up (T2 = 1.5x, T3 = 2.0x ...) synced to engine BaseAtr so R:R never collapses to 99.9:1; row shows a T2/T3 tag.
+ Profit-Lock SL Trail: once the lock is armed (peak >= PROFIT_LOCK_TRIGGER_R on engine BaseAtr) the SL column shows the lock floor, so R:R reflects real remaining risk.
+ Attempts Line On EVERY Row: Bull or Bear, stock or index, STOCK_FNO / CASH_EQUITY / INDEX_OPTIONS / SWING_FNO / SWING_INDEX, every email + console: earlier stopped-out attempts, current attempt R, Day%.
+ Attempt-Tier Row Order: 1) all attempts +R, 2) net +R, 3) net -R; newest anchor first inside each tier (ATTEMPT_SORT_ENABLED; same order in every email).
+ Pure 1st-Time + Re-anchor: a first anchor needs its Anchor column inside PURE_ANCHOR_LOOKBACK_MIN (strictly honored, independent of NO_OF_ITERATIONS); a re-anchored row qualifies ONLY if earlier attempts meet PRIOR_ATTEMPT_MIN_POSITIVE_R (0.10R) — losing RE-ANCHOR* rows never leak into Pure 1st-Time.
+ Trailing Stop Loss: SL trails the best price since the anchor (never loosens); R:R prints 'SL hit' when LTP is already through it without silently desyncing from the tripwire engine.
+ ATR basis fix: gap-free intraday/swing ATR (ATR_IGNORE_GAPS = True) prevents overnight gaps from inflating 15m/30m/60m ATRs, folds end-of-day stub bars into the last full bar, falls back to ATR_FALLBACK_TF for young contracts, and caps ATR at a % of price (--atr-tf, --atr-period).
+ RISK_ATR_TF (mode-profiled: 30min INDEX_OPTIONS / 60min STOCK_FNO / 360min CASH_EQUITY / 375min SWING): SL / Target / R:R distances use mode-appropriate ATR while anchors, kills, Profit-Lock, and R-multiples stay on ATR_BASIS_TF (--risk-atr-tf).
+ CE/PE Dual-Band Confluence Gate (MANDATORY for every option anchor): ONE Bollinger arena per strike on actual CE/PE option premiums (catching IV-spike & IV-crush traps); the twin leg of the same strike/expiry is NORMALISED into it
  (ZSCORE: PE(norm) = SMA(CE) + (PE - SMA(PE)) * sigma(CE)/sigma(PE), valid on ANY strike; OFFSET = literal rupee alignment, ATM only).
  CE BUY = CE > Upper & PE(norm) < Lower | PE BUY = PE(norm) > Upper & CE < Lower. Both legs above the Upper band = IV-spike TRAP, both below the Lower = IV-crush TRAP, both inside = CHOP, any other mix = not confirmed -> no anchor.
+ Option-Delta Swing Engine (SWING_FNO / SWING_INDEX):
- Exact NSE 375-min divisor architecture: 15min candles (25 bars/day) | 75min ATR basis (5 bars/day) | 375min Risk ATR (1 full session, zero stub bars).
- Current-expiry liquidity preserved (EXPIRY_SELECTION = "CURRENT", auto-roll only on expiry day).
- Overnight gap-ups and gap-downs completely ignored across all indicators, BB/KC, Delta-BB, IV-BB, and Dual-Band via gap-stitched continuous multi-day chains (ISOLATE_SESSION_BANDS = False).
- Charm & Vanna neutralized on current-expiry multi-day |Delta| signals using causal fixed-horizon/constant-IV signal |Delta| while gating & displaying true live |Delta| (0.45-0.80).
- Multi-day stateful anchor carryover (SWING_LOOKBACK_SESSIONS = 10) with Delta-based trailing exits (SWING_DELTA_TRAIL_ATR_MULT) and Fresh Swing (1-2 sessions) vs Still-Holding Swing separation.
"""
import os, sys, re, html, argparse, traceback, urllib.parse, gzip, json, time, threading, math
from bisect import bisect_left
from datetime import datetime, timedelta, timezone, time as dtime
import concurrent.futures, smtplib
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

# --- 0.1 EMAIL & GITHUB SECRET CONFIGURATION ---
SEND_DASHBOARD_EMAIL = True
EMAIL_SEND_ONLY_WHEN_PURE_1ST_TIME_GT_0 = True
EMAIL_ONLY_PURE_1ST_TIME_CONTENT = True
EMAIL_ATTACH_INTERACTIVE_HTML = True
EMAIL_ATTACHMENT_FILENAME = "System3_Pure_1stTime_Popups.html"
SAVE_LOCAL_HTML_FILE = False
HTML_DASHBOARD_PATH = "system3_dashboard.html"

# --- 0.1B STILL HOLDING GOOD EMAIL CONFIGURATION ---
SEND_STILL_HOLDING_EMAIL = True                  # Master toggle: send separate email for setups that still hold good
EMAIL_SEND_ONLY_WHEN_STILL_HOLDING_GT_0 = True   # Only send if at least 1 setup still holds good
STILL_HOLDING_EXCLUDE_PURE_1ST_TIME = True       # Avoid duplicating brand-new Pure 1st-Time popups into the Still-Holding email
STILL_HOLDING_MIN_R = 0.0                        # Minimum live R-multiple for Still-Holding email (0.0 = non-underwater only; e.g. -0.5 to allow slightly red)
EMAIL_ATTACH_STILL_HOLDING_HTML = True           # Attach interactive HTML for still-holding setups
EMAIL_STILL_HOLDING_ATTACHMENT_FILENAME = "System3_Still_Holding_Setups.html"
ENV_SEND_STILL_HOLDING_EMAIL = "SEND_STILL_HOLDING_EMAIL"
ENV_EMAIL_SEND_ONLY_WHEN_STILL_HOLDING_GT_0 = "EMAIL_SEND_ONLY_WHEN_STILL_HOLDING_GT_0"
ENV_STILL_HOLDING_EXCLUDE_PURE_1ST_TIME = "STILL_HOLDING_EXCLUDE_PURE_1ST_TIME"
ENV_STILL_HOLDING_MIN_R = "STILL_HOLDING_MIN_R"

EMAIL_MAX_BUYERS = 0
EMAIL_MAX_SELLERS = 0
EMAIL_MAX_GRAVEYARD = 0
EMAIL_CLIP_SYMBOLS = False

ENV_UPSTOX_TOKEN = "UPSTOX_ACCESS_TOKEN"
ENV_FALLBACK_TOKEN = "ACCESS_TOKEN"
ENV_SENDER_EMAIL = "SENDER_EMAIL"
ENV_SENDER_PASSWORD = "SENDER_PASSWORD"
ENV_RECIPIENT_EMAIL = "RECIPIENT_EMAIL"
ENV_SMTP_SERVER = "SMTP_SERVER"
ENV_SMTP_PORT = "SMTP_PORT"
ENV_TRACK_SYMBOL = "TRACK_SYMBOL"
ENV_NO_OF_ITERATIONS = "NO_OF_ITERATIONS"
ENV_LOOKBACK_ITERATIONS = "LOOKBACK_ITERATIONS"
ENV_TIME_OF_INTERVAL_TO_LOOKBACK = "TIME_OF_INTERVAL_TO_LOOKBACK"
ENV_LOOKBACK_INTERVAL_MIN = "LOOKBACK_INTERVAL_MIN"
ENV_EMAIL_SEND_ONLY_WHEN_PURE_1ST_TIME = "EMAIL_SEND_ONLY_WHEN_PURE_1ST_TIME_GT_0"
ENV_EMAIL_ONLY_PURE_1ST_TIME_CONTENT = "EMAIL_ONLY_PURE_1ST_TIME_CONTENT"

DEFAULT_SMTP_SERVER = "smtp.gmail.com"
DEFAULT_SMTP_PORT = "465"
SMTP_SSL_PORT = 465
SMTP_TIMEOUT_SEC = 30

# --- 0.2 EXECUTION & UNIVERSE FILTERING ---
TRADING_MODE = "STOCK_FNO"
MODE_PROFILES = {
    "INDEX_OPTIONS": {
        "candle_tf_min": 3,
        "atr_basis_tf": "15min",
        "risk_atr_tf": "30min",
        "history_days": 7,
        "strikes_up": 2,
        "strikes_down": 2,
        "opt_min_price": 10.0,
        "opt_min_volume": 5000,
        "dual_band_std_mult": 1.0,
        "dual_band_twin_max_stale_min": 6,
    },
    "STOCK_FNO": {
        "candle_tf_min": 3,
        "atr_basis_tf": "15min",
        "risk_atr_tf": "60min",
        "history_days": 7,
        "strikes_up": 0,
        "strikes_down": 0,
        "opt_min_price": 5.0,
        "opt_min_volume": 50000,
        "dual_band_std_mult": 1.0,
        "dual_band_twin_max_stale_min": 15,
    },
    "SWING_FNO": {
        "candle_tf_min": 15,
        "atr_basis_tf": "75min",
        "risk_atr_tf": "375min",
        "history_days": 21,
        "strikes_up": 0,
        "strikes_down": 0,
        "opt_min_price": 5.0,
        "opt_min_volume": 50000,
        "dual_band_std_mult": 1.0,
        "dual_band_twin_max_stale_min": 15,
        "isolate_session_bands": False,
        "swing_mode": True,
    },
    "SWING_INDEX": {
        "candle_tf_min": 75,
        "atr_basis_tf": "75min",
        "risk_atr_tf": "375min",
        "history_days": 30,
        "strikes_up": 2,
        "strikes_down": 2,
        "opt_min_price": 10.0,
        "opt_min_volume": 5000,
        "dual_band_std_mult": 1.0,
        "dual_band_twin_max_stale_min": 75,
        "isolate_session_bands": False,
        "swing_mode": True,
    },
    "CASH_EQUITY": {
        "candle_tf_min": 15,
        "atr_basis_tf": "60min",
        "risk_atr_tf": "360min",
        "history_days": 19,
    },
}
_CLI_OVERRIDES = set()
WORKERS = 16
INCLUDE_NON_EQ_SERIES = False
MIN_PRICE = 100.0
MAX_PRICE = 5000.0
MIN_DAILY_VOLUME = 100000
BACKTRACE_DAYS = 30
PREFILTER_LOOKBACK_MULT = 2
PREFILTER_LOOKBACK_PAD_DAYS = 6
MIN1_HISTORY_DAYS = 19
MAX_PINNED_MATCHES = 8

# --- 0.2B OPTION-DELTA SWING ENGINE CONFIGURATION ---
SWING_MODE_ACTIVE = False               # Automatically set True for SWING_FNO / SWING_INDEX
SWING_LOOKBACK_SESSIONS = 10            # Multi-day stateful anchor carryover window (trading sessions)
SWING_FRESH_SESSIONS = 2                # Anchors within last N trading sessions count as Fresh Swing (Pure 1st-Time)
SWING_DELTA_MIN_ABS = 0.45              # Minimum live |Delta| for swing entry (avoids heavy OTM Charm decay)
SWING_DELTA_MAX_ABS = 0.80              # Maximum live |Delta| for swing entry (anti-saturation ceiling)
SWING_DELTA_EXIT_FLOOR = 0.40           # Structural |Delta| invalidation floor (kills swing if live |Delta| < floor)
SWING_DELTA_TRAIL_ATR_MULT = 2.0        # Multi-day |Delta| trailing stop distance in Delta-ATR units
SWING_DELTA_PROFIT_LOCK_ABS = 0.82      # Tighten |Delta| trail once Peak |Delta| reaches deep ITM saturation
SWING_DELTA_PROFIT_LOCK_TRAIL_MULT = 1.0
SWING_CHARM_REF_DTE_DAYS = 15.0         # Constant-horizon reference DTE (days) to neutralize Charm on current expiry
SWING_LAST_NEW_ANCHOR_MIN = 15 * 60 + 30  # Allow closing 15:15-15:30 confirmation bars in Swing mode
ENV_SWING_LOOKBACK_SESSIONS = "SWING_LOOKBACK_SESSIONS"
ENV_SWING_FRESH_SESSIONS = "SWING_FRESH_SESSIONS"
ENV_SWING_DELTA_MIN_ABS = "SWING_DELTA_MIN_ABS"
ENV_SWING_DELTA_MAX_ABS = "SWING_DELTA_MAX_ABS"
ENV_SWING_DELTA_TRAIL_ATR_MULT = "SWING_DELTA_TRAIL_ATR_MULT"

# --- 0.3 SESSION & TIME BOUNDARIES ---
IST_OFFSET_HOURS = 5
IST_OFFSET_MINUTES = 30
IST = timezone(timedelta(hours=IST_OFFSET_HOURS, minutes=IST_OFFSET_MINUTES))
SESSION_OPEN_MIN = 9 * 60 + 15
SESSION_CLOSE_MIN = 15 * 60 + 30
LAST_NEW_ANCHOR_MIN = 15 * 60 + 15
CANDLE_TF_MIN = 3                       # Engine candle timeframe in minutes (overridden per mode by MODE_PROFILES; CLI --candle-tf)
ENV_CANDLE_TF_MIN = "CANDLE_TF_MIN"
ANCHOR_OPEN_WARMUP_BARS = 5
DEFAULT_SNAPSHOT_TIME = "15:30"
CHECKPOINT_INTERVAL_MIN = 5
BACKTRACE_CHECKPOINTS = True
DAY_PCT_BASIS = "PREV_CLOSE"
MAX_SESSION_LOOKBACK_DAYS = 14
SESSION_BUCKET_MULT = 10000

# --- 0.3B IN-MEMORY MULTI-ITERATION FIRST-TIME POPUP LOOKBACK ---
ENABLE_ITERATION_LOOKBACK = True
NO_OF_ITERATIONS = 10
TIME_OF_INTERVAL_TO_LOOKBACK = 5
LOOKBACK_ITERATIONS = NO_OF_ITERATIONS
LOOKBACK_INTERVAL_MIN = TIME_OF_INTERVAL_TO_LOOKBACK

STRICT_FIRST_TIME_POPUP_FILTER = True
POPUP_REQUIRE_NON_ZERO_GRID = True
POPUP_MIN_R_MULTIPLE = 0.0
POPUP_MAX_R_MULTIPLE = 2.0              # Anti-chase cap for Pure 1st-Time popups: skip setups that already ran > 2.0R (0 = off)
POPUP_MAX_PULLBACK_FROM_PEAK_R = 0.5    # Anti-drop guard: skip Pure 1st-Time popups already down > 0.5R from post-anchor peak (0 = off)
PURE_1ST_TIME_CURRENT_ITER_ONLY = False

# --- 0.3B-2 PURE 1ST-TIME POPUPS: FIRST ANCHORS AND RE-ANCHORS ---
PURE_ANCHOR_LOOKBACK_MIN = 15       # Strict window for FIRST anchors: Anchor column within the last N minutes of the run (fresh early entries). 0 = use full iteration lookback window
PURE_REANCHOR_ANY_TIME = False      # Re-anchors obey the same freshness window so stale old setups do not pop up after the move
PURE_ANCHOR_IGNORE_GAP = False      # Enforce MAX_TIMESTAMP_GAP_MINUTES on Pure 1st-Time popups so stale post-move anchors are excluded
PURE_ANCHOR_REQUIRE_GRID = False    # False: never reject an all-zero ATR/BB grid (momentum fades long before a setup stops holding). True = apply POPUP_REQUIRE_NON_ZERO_GRID to both
PURE_ANCHOR_MIN_R = None            # Lowest CURRENT R (since the current anchor) for both. None = POPUP_MIN_R_MULTIPLE (0.0 = not underwater). E.g. -0.5 also keeps slightly red rows
PURE_ALLOW_REANCHOR_WITH_POSITIVE_PRIOR = True   # Master toggle for re-anchors (False = old behaviour: every '*' row is kept out of the pure list)
PURE_REANCHOR_PRIOR_MODE = "ANY"    # Re-anchor history test: ANY = at least one earlier attempt is +R | ALL = every earlier attempt is +R | NET = earlier attempts net +R | NONE = every re-anchor
PRIOR_ATTEMPT_MIN_POSITIVE_R = 0.10 # "+R" for an earlier attempt = it exited at least this many R in profit. A +0.03R scratch is NOT a win. 0 = any R > 0
PURE_INCLUDE_PLAIN_FIRST_TIME = True             # True: first anchors of the day (no earlier attempt) are listed too | False: pure list = re-anchors only
ENV_PURE_ALLOW_REANCHOR = "PURE_ALLOW_REANCHOR_WITH_POSITIVE_PRIOR"
ENV_PURE_REANCHOR_PRIOR_MODE = "PURE_REANCHOR_PRIOR_MODE"
ENV_PURE_REANCHOR_ANY_TIME = "PURE_REANCHOR_ANY_TIME"
ENV_PRIOR_ATTEMPT_MIN_POSITIVE_R = "PRIOR_ATTEMPT_MIN_POSITIVE_R"
ENV_PURE_ANCHOR_LOOKBACK_MIN = "PURE_ANCHOR_LOOKBACK_MIN"
ENV_PURE_ANCHOR_REQUIRE_GRID = "PURE_ANCHOR_REQUIRE_GRID"
ENV_PURE_ANCHOR_MIN_R = "PURE_ANCHOR_MIN_R"
ENV_PURE_ANCHOR_IGNORE_GAP = "PURE_ANCHOR_IGNORE_GAP"
ENV_PURE_INCLUDE_PLAIN_FIRST_TIME = "PURE_INCLUDE_PLAIN_FIRST_TIME"

# --- 0.3C TIMESTAMP GAP CHECK CONFIGURATION ---
ENABLE_TIMESTAMP_GAP_CHECK = True       # Master toggle: filter Pure 1st-Time setups where (Seen - Anchor) exceeds threshold
MAX_TIMESTAMP_GAP_MINUTES = 12          # Max allowed gap in minutes between Seen Time and Anchor Time (0 = disabled)
ENV_ENABLE_TIMESTAMP_GAP_CHECK = "ENABLE_TIMESTAMP_GAP_CHECK"
ENV_MAX_TIMESTAMP_GAP_MINUTES = "MAX_TIMESTAMP_GAP_MINUTES"

# --- 0.4 LOCK 1: PRICE VOLATILITY BANDS ---
PRICE_BB_PERIOD = 20
PRICE_BB_STD_MULT = 1.5
PRICE_KC_PERIOD = 20
PRICE_KC_ATR_MULT = 1.5
ROLLING_MIN_PERIODS = 1
ISOLATE_SESSION_BANDS = True

# --- 0.5 KINETIC INDICATOR PERIODS ---
RSI_PERIOD = 14
MACD_FAST_PERIOD = 12
MACD_SLOW_PERIOD = 26
MACD_SIGNAL_PERIOD = 9
ADX_PERIOD = 14
IND_BB_PERIOD = 20
VEL_ATR_PERIOD = 14
INDEX_SYNTHETIC_VOL_FALLBACK = True
EPSILON = 1e-8

# --- 0.6 LOCK 2 & LOCK 3: STRUCTURE & KINETIC SHOCK TRIPWIRES ---
IND_BB_PIERCE_STD = 0.75
STRUCT_QUORUM = 2
KINETIC_SHOCK_ATR_MULT = 1.15
SHOCK_QUORUM = 2
SHOCK_WINDOW_BARS = 3
SHOCK_FLOW_MIN_THRESHOLD = 0.0
GRID_MULTIPLIERS = [1, 2, 3, 5]

# --- 0.7 BASE ATR, TRAILING STOPS, PROFIT RATCHET & TARGET/SL ---
ATR_BASIS_PERIOD = 14
ATR_BASIS_TF = "15min"                  # Any timeframe: "15min", "60min", "75min", "375min" (~ one full NSE day), "6h" ...
ATR_IGNORE_GAPS = True                  # Exclude overnight gaps from intraday/swing True Range so 15m/30m/60m ATRs aren't blown up by open gaps
ATR_MIN_BARS = 5                        # Fewer completed bars than this at ATR_BASIS_TF -> use the fallback TF (or skip as short_history)
ATR_FALLBACK_TF = "15min"               # Used ONLY for contracts too young for ATR_BASIS_TF (e.g. new weekly options at 375min). "off" = skip them instead
MIN_ATR_PCT = 0.005
INDEX_MIN_ATR_PCT = 0.0025
OPT_MIN_ATR_PCT = 0.04
MAX_ATR_PCT = 0.06                      # stocks / cash equity
INDEX_MAX_ATR_PCT = 0.04                # sectoral indices
OPT_MAX_ATR_PCT = 0.35                  # option premiums
RISK_ATR_TF = "360min"                  # ATR used ONLY for the SL / Target / R:R distances (overridden per mode in MODE_PROFILES). None = same ATR as engine.
ENV_RISK_ATR_TF = "RISK_ATR_TF"
ENV_ATR_BASIS_TF = "ATR_BASIS_TF"
ENV_ATR_BASIS_PERIOD = "ATR_BASIS_PERIOD"
ENV_ATR_FALLBACK_TF = "ATR_FALLBACK_TF"
ENV_MAX_ATR_PCT = "MAX_ATR_PCT"
ENV_INDEX_MAX_ATR_PCT = "INDEX_MAX_ATR_PCT"
ENV_OPT_MAX_ATR_PCT = "OPT_MAX_ATR_PCT"
ABSOLUTE_MIN_ATR = 0.01

HARD_STOP_ATR_MULT = 2.5
HARD_STOP_RUNNER_TRIGGER_R = 3.0        # Tighten trailing hard stop once peak R >= this level (0 = off)
HARD_STOP_RUNNER_ATR_MULT = 1.5         # Tighter trailing hard stop in ATR for runners
SOFT_STOP_ATR_MULT = 1.0
SOFT_STOP_OPPOSING_SCORE = 4
OPPOSING_VEL_ATR_MULT = 1.0
OPPOSING_BB_STD_MULT = 1.0
WARZONE_OPPOSING_SCORE = 2

SOFT_STOP_DISABLE_R = 1.5
PROFIT_LOCK_TRIGGER_R = 1.8
PROFIT_LOCK_FLOOR_R = 0.5
PROFIT_LOCK_STEPS = [(1.2, 0.1), (1.8, 0.5), (2.5, 1.5), (3.5, 2.5)]  # Stepped (peak_r_trigger, floor_r) ratchet
POST_LOSS_COOLDOWN_BARS = 5             # Wait N 1m-equiv bars after a losing exit before re-anchoring
MAX_CONSECUTIVE_LOSSES_PER_DAY = 3      # Stop re-anchoring a contract for the rest of the day after N consecutive losing stops (0 = unlimited)

TARGET_ATR_MULT = PROFIT_LOCK_TRIGGER_R
SL_ATR_MULT = SOFT_STOP_ATR_MULT
TARGET_SL_FROM_ANCHOR = True

# --- 0.7B RISK-TO-REWARD (R:R) ENGINE CONFIGURATION ---
RR_RATIO_TYPE = "RISK_TO_REWARD"        # Choices: "RISK_TO_REWARD" (Risk/Reward : 1) or "REWARD_TO_RISK" (Reward/Risk : 1)
RR_SORT_ORDER = "ASC"                   # Choices: "ASC" (e.g. 0.1:1, 0.28:1 ascending) or "DESC"
ENV_RR_RATIO_TYPE = "RR_RATIO_TYPE"
ENV_RR_SORT_ORDER = "RR_SORT_ORDER"

# --- 0.7C TARGET LADDER ---
TARGET_LADDER_ENABLED = True            # Master toggle (False = old behaviour: fixed T1, R:R 99.9:1 once achieved)
TARGET_LADDER_STEP = 0.5                # Increment of the multiplier per level (0.5 -> 1.0x, 1.5x, 2.0x, 2.5x ...)
TARGET_LADDER_MAX_LEVELS = 50           # Safety cap on the number of ladder levels (practically never reached)
SL_TRAIL_TO_LOCK_FLOOR = True           # Once the Profit-Lock is armed (peak >= PROFIT_LOCK_TRIGGER_R) show SL = lock floor
ENV_TARGET_LADDER_ENABLED = "TARGET_LADDER_ENABLED"
ENV_TARGET_LADDER_STEP = "TARGET_LADDER_STEP"
ENV_SL_TRAIL_TO_LOCK_FLOOR = "SL_TRAIL_TO_LOCK_FLOOR"

# --- 0.7D PRIOR-ATTEMPT CONTEXT ---
SHOW_PRIOR_ATTEMPTS_LINE = True
SHOW_ATTEMPTS_LINE_FOR_ALL_ROWS = True  # True: EVERY row (Bull or Bear, stock or index, any mode) gets the line, also first attempts
PRIOR_ATTEMPTS_MAX_SHOWN = 4            # Max earlier attempts listed per row (most recent ones, older ones are summarised)
ENV_SHOW_ATTEMPTS_LINE_FOR_ALL_ROWS = "SHOW_ATTEMPTS_LINE_FOR_ALL_ROWS"

# --- 0.7E ATTEMPT-TIER ROW ORDER ---
ATTEMPT_SORT_ENABLED = True             # False = legacy order (R:R first)
ATTEMPT_SORT_TIME_DESC = True           # True = most recent anchor first inside a tier, False = oldest first
ENV_ATTEMPT_SORT_ENABLED = "ATTEMPT_SORT_ENABLED"

# --- 0.7F TRAILING STOP LOSS ---
SL_TRAILING_ENABLED = True              # False = old behaviour (SL fixed at Anchor -/+ SL_ATR_MULT x ATR)
SL_TRAIL_ATR_MULT = None                # Distance behind the best price in ATR. None = SL_ATR_MULT (1.0).
ENV_SL_TRAILING_ENABLED = "SL_TRAILING_ENABLED"
ENV_SL_TRAIL_ATR_MULT = "SL_TRAIL_ATR_MULT"

ADVERSE_DRIFT_R_FLOOR = -0.5
ADVERSE_DRIFT_STOP_RATIO = 1.0
BANKED_R_PROTECTION_FLOOR = 0.5

# --- 0.8 SECTORAL INDICES & MULTI-INDEX WEIGHTAGE MATRIX ---
ENABLE_SECTORAL_INDICES = True
DEFAULT_UNMAPPED_SECTOR = "NIFTY 500"
DEFAULT_UNMAPPED_WEIGHT = 0.10

SECTORAL_INDEX_CONFIG = {
    "NIFTY BANK":        {"spot_key": "NSE_INDEX|Nifty Bank",              "aliases": {"NIFTY BANK", "BANKNIFTY", "NIFTYBANK"}},
    "NIFTY PSU BANK":    {"spot_key": "NSE_INDEX|Nifty PSU Bank",          "aliases": {"NIFTY PSU BANK", "NIFTYSUBANK", "PSUBANK"}},
    "NIFTY FIN SERVICE": {"spot_key": "NSE_INDEX|Nifty Fin Service",       "aliases": {"NIFTY FIN SERVICE", "FINNIFTY", "NIFTYFINSERVICE"}},
    "NIFTY IT":          {"spot_key": "NSE_INDEX|Nifty IT",                "aliases": {"NIFTY IT", "NIFTYIT", "CNXIT"}},
    "NIFTY AUTO":        {"spot_key": "NSE_INDEX|Nifty Auto",              "aliases": {"NIFTY AUTO", "NIFTYAUTO"}},
    "NIFTY PHARMA":      {"spot_key": "NSE_INDEX|Nifty Pharma",            "aliases": {"NIFTY PHARMA", "NIFTYPHARMA"}},
    "NIFTY METAL":       {"spot_key": "NSE_INDEX|Nifty Metal",             "aliases": {"NIFTY METAL", "NIFTYMETAL"}},
    "NIFTY FMCG":        {"spot_key": "NSE_INDEX|Nifty FMCG",              "aliases": {"NIFTY FMCG", "NIFTYFMCG"}},
    "NIFTY ENERGY":      {"spot_key": "NSE_INDEX|Nifty Energy",            "aliases": {"NIFTY ENERGY", "NIFTYENERGY"}},
    "NIFTY OIL & GAS":   {"spot_key": "NSE_INDEX|Nifty Oil and Gas",       "aliases": {"NIFTY OIL AND GAS", "NIFTY OIL & GAS", "NIFTYOILGAS"}},
    "NIFTY REALTY":      {"spot_key": "NSE_INDEX|Nifty Realty",            "aliases": {"NIFTY REALTY", "NIFTYREALTY"}},
    "NIFTY INFRA":       {"spot_key": "NSE_INDEX|Nifty Infra",             "aliases": {"NIFTY INFRA", "NIFTYINFRA"}},
    "NIFTY CONSUMPTION": {"spot_key": "NSE_INDEX|Nifty India Consumption", "aliases": {"NIFTY INDIA CONSUMPTION", "NIFTY CONSUMPTION", "NIFTYCONSUMPTION"}},
    "NIFTY PSE":         {"spot_key": "NSE_INDEX|Nifty PSE",               "aliases": {"NIFTY PSE", "NIFTYPSE", "NIFTY CPSE"}},
}

SECTOR_CONSTITUENT_WEIGHTS = {
    "NIFTY IT": {
        "INFY": 31.30, "TCS": 19.08, "HCLTECH": 11.81, "TECHM": 10.47, "COFORGE": 6.59,
        "PERSISTENT": 6.21, "WIPRO": 4.64, "LTM": 4.06, "LTIM": 4.06, "MPHASIS": 3.09,
        "OFSS": 2.75, "TATAELXSI": 2.40, "KPITTECH": 2.25, "CYIENT": 1.50, "BSOFT": 1.20,
    },
    "NIFTY PSU BANK": {
        "SBIN": 32.50, "BANKBARODA": 16.80, "PNB": 13.40, "CANBK": 12.60, "UNIONBANK": 9.20,
        "INDIANB": 6.80, "BANKINDIA": 5.10, "MAHABANK": 2.10, "IOB": 1.50,
    },
    "NIFTY BANK": {
        "HDFCBANK": 28.10, "ICICIBANK": 24.30, "SBIN": 9.80, "KOTAKBANK": 10.40, "AXISBANK": 9.90,
        "INDUSINDBK": 5.40, "FEDERALBNK": 3.10, "IDFCFIRSTB": 2.60, "AUBANK": 2.40, "BANDHANBNK": 1.60,
        "RBLBANK": 1.40, "BANKBARODA": 2.80, "PNB": 2.10, "CANBK": 1.90, "YESBANK": 1.20,
    },
    "NIFTY FIN SERVICE": {
        "HDFCBANK": 22.40, "ICICIBANK": 19.10, "KOTAKBANK": 8.10, "AXISBANK": 7.60, "SBIN": 6.50,
        "BAJFINANCE": 9.20, "BAJAJFINSV": 4.30, "SHRIRAMFIN": 3.80, "CHOLAFIN": 3.10, "PFC": 2.95,
        "RECLTD": 2.85, "HDFCLIFE": 2.75, "SBILIFE": 2.70, "ICICIGI": 2.15, "ICICIPRULI": 1.65,
        "HDFCAMC": 1.85, "MUTHOOTFIN": 1.75, "BSE": 2.40, "MCX": 1.60, "CDSL": 1.55,
        "SBICARD": 1.45, "LICHSGFIN": 1.35, "JIOFIN": 3.20, "IREDA": 1.30, "LICI": 1.95,
        "MFSL": 1.40, "ABCAPITAL": 1.25, "MANAPPURAM": 1.15, "LTF": 1.20, "PNBHOUSING": 1.10,
        "ANGELONE": 1.25, "CAMS": 1.20, "KFINTECH": 1.05, "360ONE": 1.15, "NAM-INDIA": 1.00,
        "PAYTM": 1.30, "POLICYBZR": 1.45, "MOTILALOFS": 1.05, "ANANDRATHI": 0.95, "IIFL": 0.90,
    },
    "NIFTY AUTO": {
        "M&M": 23.57, "MARUTI": 14.42, "TMPV": 11.50, "TATAMOTORS": 11.50, "BAJAJ-AUTO": 9.91,
        "EICHERMOT": 8.38, "TVSMOTOR": 7.87, "HEROMOTOCO": 5.43, "MOTHERSON": 5.17, "BHARATFORG": 4.55,
        "ASHOKLEY": 3.69, "BOSCHLTD": 3.20, "TIINDIA": 3.10, "MRF": 2.40, "BALKRISIND": 2.10,
        "SONACOMS": 2.05, "UNOMINDA": 1.95, "APOLLOTYRE": 1.65, "EXIDEIND": 1.55, "ATHERENERG": 1.50,
    },
    "NIFTY PHARMA": {
        "SUNPHARMA": 24.50, "DIVISLAB": 10.20, "CIPLA": 9.80, "DRREDDY": 9.40, "LUPIN": 6.80,
        "TORNTPHARM": 5.20, "AUROPHARMA": 5.10, "ZYDUSLIFE": 4.80, "MAXHEALTH": 6.40, "APOLLOHOSP": 6.10,
        "FORTIS": 3.90, "MANKIND": 4.10, "ALKEM": 3.60, "LAURUSLABS": 2.90, "GLENMARK": 2.70,
        "BIOCON": 2.50, "IPCALAB": 2.20, "SYNGENE": 1.90, "GRANULES": 1.60, "ABBOTINDIA": 1.80,
    },
    "NIFTY METAL": {
        "TATASTEEL": 21.40, "HINDALCO": 18.90, "JSWSTEEL": 16.50, "VEDL": 11.80, "ADANIENT": 10.60,
        "JINDALSTEL": 8.40, "NMDC": 4.90, "HINDZINC": 4.60, "SAIL": 3.80, "NATIONALUM": 3.50,
        "APLAPOLLO": 3.40, "JSL": 2.60, "HINDCOPPER": 2.10, "WELCORP": 1.70,
    },
    "NIFTY FMCG": {
        "ITC": 31.20, "HINDUNILVR": 21.80, "NESTLEIND": 8.90, "VBL": 7.60, "BRITANNIA": 6.40,
        "TATACONSUM": 6.10, "GODREJCP": 4.90, "MARICO": 4.10, "DABUR": 3.80, "COLPAL": 3.50,
        "UNITDSPR": 3.40, "RADICO": 2.20, "PATANJALI": 2.10, "GODFRYPHLP": 1.80, "UBL": 1.70, "JYOTHYLAB": 1.30,
    },
    "NIFTY OIL & GAS": {
        "RELIANCE": 33.20, "ONGC": 16.50, "IOC": 11.40, "BPCL": 10.80, "GAIL": 8.90,
        "HINDPETRO": 5.60, "OIL": 4.80, "PETRONET": 4.20, "IGL": 2.90, "MGL": 2.10, "GUJGASLTD": 1.95, "ATGL": 1.50,
    },
    "NIFTY ENERGY": {
        "RELIANCE": 27.40, "NTPC": 15.80, "POWERGRID": 11.90, "ONGC": 10.50, "COALINDIA": 9.40,
        "TATAPOWER": 6.80, "IOC": 5.20, "BPCL": 4.90, "ADANIGREEN": 4.60, "ADANIPOWER": 4.40,
        "JSWENERGY": 3.90, "SIEMENS": 4.80, "ABB": 4.20, "CGPOWER": 3.85, "BHEL": 3.40,
        "ADANIENSOL": 3.10, "NHPC": 2.90, "SUZLON": 3.20, "WAAREEENER": 2.80, "PREMIERENE": 2.60,
        "INOXWIND": 2.10, "TORNTPOWER": 2.70, "SJVN": 1.80, "CESC": 1.60, "GVT&D": 2.20, "ENRIN": 2.15,
    },
    "NIFTY REALTY": {
        "DLF": 26.80, "GODREJPROP": 17.40, "LODHA": 16.90, "MACROTECH": 16.90, "PHOENIXLTD": 13.50,
        "PRESTIGE": 11.80, "OBEROIRLTY": 10.60, "BRIGADE": 4.80, "SOBHA": 3.10, "RAYMOND": 2.40,
    },
    "NIFTY INFRA": {
        "LT": 28.50, "RELIANCE": 19.50, "BHARTIARTL": 18.40, "NTPC": 8.20, "ULTRACEMCO": 7.60,
        "POWERGRID": 6.40, "ADANIPORTS": 5.80, "GRASIM": 5.40, "INDIGO": 5.20, "AMBUJACEM": 4.50,
        "SHREECEM": 3.60, "ACC": 2.80, "DALBHARAT": 2.60, "CONCOR": 3.10, "INDUSTOWER": 3.40,
        "CUMMINSIND": 3.90, "GMRAIRPORT": 2.50, "DELHIVERY": 2.70, "IRB": 1.90, "JKCEMENT": 2.10,
        "RAMCOCEM": 1.80, "RVNL": 2.95,
    },
    "NIFTY CONSUMPTION": {
        "BHARTIARTL": 14.20, "ITC": 11.50, "TRENT": 9.80, "TITAN": 9.40, "ASIANPAINT": 8.60,
        "DMART": 7.90, "INDHOTEL": 6.80, "NAUKRI": 6.40, "HAVELLS": 5.90, "ZOMATO": 6.20,
        "ETERNAL": 6.20, "SWIGGY": 4.80, "NYKAA": 4.50, "JUBLFOOD": 4.40, "PIDILITIND": 5.60,
        "DIXON": 5.20, "POLYCAB": 4.90, "KEI": 4.30, "VOLTAS": 4.10, "BLUESTARCO": 3.80,
        "CROMPTON": 3.50, "SUPREMEIND": 3.90, "ASTRAL": 3.60, "PAGEIND": 3.20, "KAYNES": 3.40,
        "AMBER": 2.90, "BERGEPAINT": 2.50, "BATAINDIA": 2.10, "KALYANKJIL": 3.30, "UPL": 3.70,
        "PIIND": 4.00, "SRF": 4.20, "AARTIIND": 2.60, "DEEPAKNTR": 2.70, "TATACHEM": 2.50,
        "CHAMBLFERT": 2.30, "COROMANDEL": 2.80,
    },
    "NIFTY PSE": {
        "HAL": 16.40, "BEL": 15.20, "NTPC": 11.00, "POWERGRID": 9.50, "ONGC": 9.00,
        "COALINDIA": 8.20, "BDL": 5.80, "MAZDOCK": 5.60, "COCHINSHIP": 4.90, "IRCTC": 4.80,
        "IRFC": 4.50, "NBCC": 3.20, "HUDCO": 2.90, "NLCINDIA": 2.40,
    },
}

# --- 0.9 INDEX OPTIONS CONFIGURATION ---
OPTIONS_BUYING_ONLY = True
EXPIRY_SELECTION = "CURRENT"               # Choices: CURRENT (0), NEXT (1), NEXT2 (2), MONTHLY, NEXT_MONTHLY, or exact date (YYYY-MM-DD / DD-MMM-YYYY)
ENV_EXPIRY_SELECTION = "EXPIRY_SELECTION"
STOCK_FNO_AUTO_ROLL_ON_EXPIRY_DAY = True   # On monthly expiry day (0DTE), automatically roll STOCK_FNO/SWING_FNO to next expiry
INDEX_AUTO_ROLL_ON_EXPIRY_DAY = False      # Optional: keep False for index weekly 0DTE trading, True to roll index options on expiry day too
ENV_STOCK_FNO_AUTO_ROLL_ON_EXPIRY_DAY = "STOCK_FNO_AUTO_ROLL_ON_EXPIRY_DAY"
ENV_INDEX_AUTO_ROLL_ON_EXPIRY_DAY = "INDEX_AUTO_ROLL_ON_EXPIRY_DAY"
STRIKES_ABOVE_ATM = 2
STRIKES_BELOW_ATM = 2
STOCK_OPT_STRIKES_ABOVE_ATM = 0
STOCK_OPT_STRIKES_BELOW_ATM = 0
NO_OTM_STRIKES = True
INDEX_OPTIONS_BEST_STRIKE_ONLY = True      # Collapse correlated ATM/ITM strikes to the single highest-ranked strike per index/direction (override with --all-strikes)
ENV_INDEX_OPTIONS_BEST_STRIKE_ONLY = "INDEX_OPTIONS_BEST_STRIKE_ONLY"
STOCK_FNO_MAX_STRIKES = 120             # Top N most-liquid stock option strike pairs by turnover (0 = unlimited)
OPT_MIN_PRICE = 10.0
OPT_MIN_VOLUME = 5000
OPT_MIN_DAY_PCT = -15.0                 # Block option anchors if option is already crushed below -15% on the day (None = off)
OPT_MAX_DAY_PCT = 150.0                 # Block option anchors if option is already spent above +150% intraday (None = off)
STOCK_FNO_BLOCK_AGAINST_SECTOR = True   # In STOCK_FNO / SWING_FNO, block stock options whose direction opposes their active Sectoral Index
LIVE_QUOTE_FALLBACK_MIN = 3
SPOT_1M_LOOKBACK_DAYS = 5
SPOT_DAILY_LOOKBACK_DAYS = 10
INDEX_CONFIG = {
    "NIFTY":      {"spot_key": "NSE_INDEX|Nifty 50",          "aliases": {"NIFTY", "NIFTY 50"}},
    "BANKNIFTY": {"spot_key": "NSE_INDEX|Nifty Bank",        "aliases": {"BANKNIFTY", "NIFTY BANK"}},
    "FINNIFTY":  {"spot_key": "NSE_INDEX|Nifty Fin Service", "aliases": {"FINNIFTY", "NIFTY FIN SERVICE"}},
    "SENSEX":    {"spot_key": "BSE_INDEX|SENSEX",            "aliases": {"SENSEX", "BSE SENSEX", "BSESN"}},
}
ALIAS_TO_INDEX = {a: idx for idx, cfg in INDEX_CONFIG.items() for a in cfg["aliases"]}
PROBE_KEY = INDEX_CONFIG["NIFTY"]["spot_key"]

# --- 0.9B OPTION IMPLIED VOLATILITY (IV) & CRUSH PROTECTION ENGINE ---
ENABLE_OPTION_IV_ENGINE = True
IV_CRUSH_DROP_PCT = 0.15              # 15% drop from post-anchor peak IV triggers Kinetic SL
IV_CRUSH_VEL_MULT = 2.5              # Negative IV velocity > 2.5x ATR_vel (on smoothed IV) triggers IV crush kill
IV_CRUSH_VEL_MIN_DROP_PCT = 0.05     # Require >= 5% IV drop from post-anchor peak alongside velocity shock (and curr_r < 0)
IV_REQUIRE_NON_NEGATIVE_VEL = True   # Block anchor if IV velocity <= 0 at anchor candle
RISK_FREE_RATE = 0.065               # 6.5% standard Indian risk-free rate
ENV_ENABLE_OPTION_IV_ENGINE = "ENABLE_OPTION_IV_ENGINE"
ENV_IV_CRUSH_DROP_PCT = "IV_CRUSH_DROP_PCT"
ENV_IV_CRUSH_VEL_MULT = "IV_CRUSH_VEL_MULT"
ENV_IV_REQUIRE_NON_NEGATIVE_VEL = "IV_REQUIRE_NON_NEGATIVE_VEL"

# --- 0.9C CE/PE DUAL-BAND CONFLUENCE GATE (MANDATORY FOR EVERY OPTION ANCHOR) ---
# Runs on actual CE/PE option premiums (with log-return gap stitching when ISOLATE_SESSION_BANDS=False)
# so it genuinely detects IV-spike traps (both legs > Upper) and IV-crush traps (both legs < Lower).
DUAL_BAND_GATE_MANDATORY = True         # HARD-WIRED ON: no CLI / env switch by design. Missing / stale twin => no anchor (fail-closed).
DUAL_BAND_PERIOD = 20                   # Bollinger period (bars) used for BOTH legs
DUAL_BAND_STD_MULT = 2.0                # Band half-width in standard deviations (overridden per mode in MODE_PROFILES)
DUAL_BAND_MAX_Z = 2.5                   # Anti-climax cap: reject overextended > +2.5sigma blow-offs before mean-reversion drop (0 = off)
DELTA_BB_PERIOD = 20                    # Mandatory |Delta| Bollinger Band period (bars) for all option modes
DELTA_BB_STD_MULT = 1.0                 # Mandatory |Delta| Upper BB sigma multiplier (early entry requires |Delta| > +1.0sigma Upper BB)
DELTA_BB_MIN_SIGMA = 0.005              # Floor on |Delta| std to prevent zero-variance tick glitches
DELTA_MIN_ATR = 0.005                   # Floor on Black-Scholes |Delta| ATR used for option stops & R-multiples
DELTA_MAX_ATR = 0.15                    # Cap on Black-Scholes |Delta| ATR used for option stops & R-multiples
DELTA_ENTRY_MAX_ABS = 0.85              # Anti-chase guard: skip options whose |Delta| is already saturated (> 0.85)
ENABLE_INCEPTION_DELTA_REGIME = True    # Two-layer macro filter: rising gap-stitched |Delta| slope gate + prints inception-to-date Life(date) mu/z context
INCEPTION_LOOKBACK_CAL_DAYS = 95        # Single daily API call lookback to find the strike's first liquid trade day
INCEPTION_MIN_DAILY_VOLUME = 1000       # Ignore dormant far-OTM days below this daily volume when anchoring contract inception
INCEPTION_DELTA_MIN_Z = 0.0             # Informational lifecycle z-score reference (gate uses rising gap-stitched |Delta| slope)
INCEPTION_SLOPE_PERIOD = 10             # Short-term gap-stitched |Delta| mean period for rising-slope confirmation
STOCK_OPT_DUAL_BAND_STD_MULT = 1.5      # Relaxed band half-width for STOCK_FNO / SWING_FNO
DUAL_BAND_ALIGN_MODE = "ZSCORE"         # ZSCORE (default) | OFFSET (literal: PE(norm) = SMA(CE) + PE - SMA(PE))
DUAL_BAND_MIN_SIGMA = 0.10              # Rs floor on each leg's sigma (2 ticks): stops tick-noise breakouts
DUAL_BAND_TWIN_MAX_STALE_MIN = 5        # Twin max stale minutes (overridden per mode in MODE_PROFILES; 0 = same day only)
STOCK_OPT_DUAL_BAND_TWIN_MAX_STALE_MIN = 15  # Relaxed twin max stale minutes for STOCK_FNO / SWING_FNO
DUAL_BAND_REVERSAL_EXIT = True          # Kill a live option anchor when the OPPOSITE side is fully confirmed by the same arena
DUAL_BAND_REVERSAL_CONFIRM_BARS = 2     # Consecutive reversal bars required (or 1 bar with opposing kinetic score >= WARZONE_OPPOSING_SCORE)
ENV_DUAL_BAND_PERIOD = "DUAL_BAND_PERIOD"
ENV_DUAL_BAND_STD_MULT = "DUAL_BAND_STD_MULT"
ENV_DUAL_BAND_ALIGN_MODE = "DUAL_BAND_ALIGN_MODE"
ENV_DUAL_BAND_MIN_SIGMA = "DUAL_BAND_MIN_SIGMA"
ENV_DUAL_BAND_TWIN_MAX_STALE_MIN = "DUAL_BAND_TWIN_MAX_STALE_MIN"
ENV_DUAL_BAND_REVERSAL_EXIT = "DUAL_BAND_REVERSAL_EXIT"
_RUN_MODE = None                        # Set by run_screener (the legend only mentions the gate in option modes)

# --- 0.10 DASHBOARD DISPLAY & FORMATTING ---
TOP_N_BUYERS = 50
TOP_N_SELLERS = 50
TOP_N_GRAVEYARD = 100
MIN_SYMBOL_WIDTH = 12
MAX_SYMBOL_WIDTH = 30
SECTOR_COL_WIDTH = 18
BANKED_R_DISPLAY_EPSILON = 0.01

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

# --- 0.11 UPSTOX API & RATE LIMITER INFRASTRUCTURE ---
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

# ==============================================================================
# HELPERS, SECTOR RESOLVER, CE/PE COUNTERS, TARGET/SL, R:R & RATE LIMITERS
# ==============================================================================
def get_upstox_token():
    return os.environ.get(ENV_UPSTOX_TOKEN) or os.environ.get(ENV_FALLBACK_TOKEN)

def _is_swing_mode(mode=None):
    m = str(mode if mode is not None else (_RUN_MODE or TRADING_MODE)).upper()
    return m in ("SWING_FNO", "SWING_INDEX") or bool(SWING_MODE_ACTIVE)

def _is_option_mode(mode=None):
    m = str(mode if mode is not None else (_RUN_MODE or TRADING_MODE)).upper()
    return m in ("INDEX_OPTIONS", "STOCK_FNO", "SWING_FNO", "SWING_INDEX")

def _is_stock_option_mode(mode=None):
    m = str(mode if mode is not None else (_RUN_MODE or TRADING_MODE)).upper()
    return m in ("STOCK_FNO", "SWING_FNO")

def _is_index_option_mode(mode=None):
    m = str(mode if mode is not None else (_RUN_MODE or TRADING_MODE)).upper()
    return m in ("INDEX_OPTIONS", "SWING_INDEX")

def _apply_mode_profile(mode):
    """Apply mode-specific defaults (candle TF, ATR basis, strikes, liquidity, Dual-Band, Swing settings) unless explicitly overridden via CLI/env."""
    global CANDLE_TF_MIN, ATR_BASIS_TF, RISK_ATR_TF, MIN1_HISTORY_DAYS, STRIKES_ABOVE_ATM, STRIKES_BELOW_ATM
    global STOCK_OPT_STRIKES_ABOVE_ATM, STOCK_OPT_STRIKES_BELOW_ATM
    global OPT_MIN_PRICE, OPT_MIN_VOLUME, DUAL_BAND_STD_MULT, STOCK_OPT_DUAL_BAND_STD_MULT
    global DUAL_BAND_TWIN_MAX_STALE_MIN, STOCK_OPT_DUAL_BAND_TWIN_MAX_STALE_MIN
    global ISOLATE_SESSION_BANDS, SWING_MODE_ACTIVE
    m_up = str(mode or "").upper()
    prof = MODE_PROFILES.get(m_up) or {}
    SWING_MODE_ACTIVE = bool(prof.get("swing_mode", False))
    if "isolate_session_bands" in prof and "isolate_session_bands" not in _CLI_OVERRIDES:
        ISOLATE_SESSION_BANDS = bool(prof["isolate_session_bands"])
    if "candle_tf_min" in prof and "candle_tf" not in _CLI_OVERRIDES:
        CANDLE_TF_MIN = _tf_to_minutes(prof["candle_tf_min"], default=15)
    if "atr_basis_tf" in prof and "atr_tf" not in _CLI_OVERRIDES:
        ATR_BASIS_TF = str(prof["atr_basis_tf"])
    if "risk_atr_tf" in prof and "risk_atr_tf" not in _CLI_OVERRIDES:
        RISK_ATR_TF = str(prof["risk_atr_tf"])
    if "history_days" in prof and "history_days" not in _CLI_OVERRIDES:
        MIN1_HISTORY_DAYS = int(prof["history_days"])
    if "strikes_up" in prof and "strikes_up" not in _CLI_OVERRIDES:
        if m_up in ("STOCK_FNO", "SWING_FNO"):
            STOCK_OPT_STRIKES_ABOVE_ATM = int(prof["strikes_up"])
        else:
            STRIKES_ABOVE_ATM = int(prof["strikes_up"])
    if "strikes_down" in prof and "strikes_down" not in _CLI_OVERRIDES:
        if m_up in ("STOCK_FNO", "SWING_FNO"):
            STOCK_OPT_STRIKES_BELOW_ATM = int(prof["strikes_down"])
        else:
            STRIKES_BELOW_ATM = int(prof["strikes_down"])
    if "opt_min_price" in prof and "opt_min_price" not in _CLI_OVERRIDES:
        OPT_MIN_PRICE = float(prof["opt_min_price"])
    if "opt_min_volume" in prof and "opt_min_volume" not in _CLI_OVERRIDES:
        OPT_MIN_VOLUME = float(prof["opt_min_volume"])
    if "dual_band_std_mult" in prof and "dual_band_std" not in _CLI_OVERRIDES:
        if m_up in ("STOCK_FNO", "SWING_FNO"):
            STOCK_OPT_DUAL_BAND_STD_MULT = float(prof["dual_band_std_mult"])
        else:
            DUAL_BAND_STD_MULT = float(prof["dual_band_std_mult"])
    if "dual_band_twin_max_stale_min" in prof and "dual_band_max_stale" not in _CLI_OVERRIDES:
        if m_up in ("STOCK_FNO", "SWING_FNO"):
            STOCK_OPT_DUAL_BAND_TWIN_MAX_STALE_MIN = float(prof["dual_band_twin_max_stale_min"])
        else:
            DUAL_BAND_TWIN_MAX_STALE_MIN = float(prof["dual_band_twin_max_stale_min"])

def _parse_bool_env(env_key, default_val):
    raw = os.environ.get(env_key)
    if raw is None or str(raw).strip() == "":
        return bool(default_val)
    s = str(raw).strip().lower()
    return s in ("1", "true", "yes", "y", "on")

def _js_onclick_arg(val):
    """Safely serialize a string argument for inline HTML onclick handlers."""
    return html.escape(json.dumps(str(val or "")), quote=True)

def get_lookback_iterations_config():
    raw_iters = os.environ.get(ENV_NO_OF_ITERATIONS) or os.environ.get(ENV_LOOKBACK_ITERATIONS)
    raw_interval = os.environ.get(ENV_TIME_OF_INTERVAL_TO_LOOKBACK) or os.environ.get(ENV_LOOKBACK_INTERVAL_MIN)
    try:
        iters = int(raw_iters) if raw_iters and str(raw_iters).strip() != "" else int(NO_OF_ITERATIONS)
    except ValueError:
        iters = int(NO_OF_ITERATIONS)
    try:
        interval = int(raw_interval) if raw_interval and str(raw_interval).strip() != "" else int(TIME_OF_INTERVAL_TO_LOOKBACK)
    except ValueError:
        interval = int(TIME_OF_INTERVAL_TO_LOOKBACK)
    return max(0, iters), max(1, interval)

def now_ist():
    return datetime.now(IST).replace(tzinfo=None)

def _close_dt(day):
    return datetime.combine(day, dtime(SESSION_CLOSE_MIN // 60, SESSION_CLOSE_MIN % 60))

def _open_dt(day):
    return datetime.combine(day, dtime(SESSION_OPEN_MIN // 60, SESSION_OPEN_MIN % 60))

def _parse_snapshot_datetime(target_date_str, target_time_str):
    clean_d = str(target_date_str or "").strip()
    clean_t = str(target_time_str or "").strip()
    if clean_d in ("", "***", "NONE", "NULL", "TODAY", "LIVE"): clean_d = ""
    if clean_t in ("", "***", "NONE", "NULL", "NOW", "LIVE"): clean_t = ""

    if clean_d and ":" in clean_d and not clean_t:
        clean_t = clean_d
        clean_d = ""

    now = now_ist()
    if not clean_d and not clean_t:
        return now, False

    parsed_date = None
    if clean_d:
        m_iso = re.match(r"^(\d{4})[-/.](0?[1-9]|1[0-2])[-/.](0?[1-9]|[12]\d|3[01])$", clean_d)
        if m_iso:
            y, m, d = int(m_iso.group(1)), int(m_iso.group(2)), int(m_iso.group(3))
            parsed_date = datetime(y, m, d).date()
        else:
            parts = re.findall(r"\d+", clean_d)
            if len(parts) == 3:
                p1, p2, p3 = int(parts[0]), int(parts[1]), int(parts[2])
                if p3 >= 1000:
                    y = p3
                    if y == now.year and ((p1 == now.month and p2 == now.day) or (p1 == now.day and p2 == now.month)):
                        parsed_date = now.date()
                    elif p1 > 12 >= p2:
                        parsed_date = datetime(y, p2, p1).date()
                    elif p2 > 12 >= p1:
                        parsed_date = datetime(y, p1, p2).date()
                    else:
                        d1 = datetime(y, p2, p1).date()
                        d2 = datetime(y, p1, p2).date()
                        if d1 > now.date() >= d2: parsed_date = d2
                        elif d2 > now.date() >= d1: parsed_date = d1
                        else: parsed_date = d1
                elif p1 >= 1000:
                    y = p1
                    parsed_date = datetime(y, p2, p3).date()
            elif len(parts) == 2:
                p1, p2 = int(parts[0]), int(parts[1])
                y = now.year
                if (p1 == now.month and p2 == now.day) or (p1 == now.day and p2 == now.month):
                    parsed_date = now.date()
                elif p1 > 12 >= p2:
                    parsed_date = datetime(y, p2, p1).date()
                elif p2 > 12 >= p1:
                    parsed_date = datetime(y, p1, p2).date()
                else:
                    d1 = datetime(y, p2, p1).date()
                    d2 = datetime(y, p1, p2).date()
                    if d1 > now.date() >= d2: parsed_date = d2
                    elif d2 > now.date() >= d1: parsed_date = d1
                    else: parsed_date = d1

        if parsed_date is None:
            for dfmt in ("%Y-%m-%d", "%d-%m-%Y", "%d/%m/%Y", "%m-%d-%Y", "%m/%d/%Y", "%Y/%m/%d", "%d-%b-%Y", "%d-%B-%Y", "%d.%m.%Y", "%Y.%m.%d"):
                try:
                    parsed_date = datetime.strptime(clean_d, dfmt).date()
                    break
                except ValueError:
                    pass
            if parsed_date is None:
                parsed_date = pd.to_datetime(clean_d, dayfirst=True).date()
    else:
        parsed_date = now.date()

    time_s = clean_t or DEFAULT_SNAPSHOT_TIME
    parsed_time = None
    for tfmt in ("%H:%M", "%H:%M:%S", "%I:%M %p", "%I:%M%p", "%H.%M"):
        try:
            parsed_time = datetime.strptime(time_s.upper(), tfmt).time()
            break
        except ValueError:
            pass
    if parsed_time is None:
        parsed_time = pd.to_datetime(time_s).time()

    return datetime.combine(parsed_date, parsed_time.replace(second=0, microsecond=0)), True

def _norm_sym(s):
    return re.sub(r'[\s_\-]+', '', str(s).upper())

def _is_ce_symbol(symbol):
    s = str(symbol or "").upper().strip()
    return bool(re.search(r'(?:\s|\d)CE$', s) or re.search(r'\sCE\s', s))

def _is_pe_symbol(symbol):
    s = str(symbol or "").upper().strip()
    return bool(re.search(r'(?:\s|\d)PE$', s) or re.search(r'\sPE\s', s))

def _is_option_symbol(symbol):
    return _is_ce_symbol(symbol) or _is_pe_symbol(symbol)

def _ce_pe_counts(rows_list):
    ce_cnt = sum(1 for r in (rows_list or []) if _is_ce_symbol(r.get("Symbol", "")))
    pe_cnt = sum(1 for r in (rows_list or []) if _is_pe_symbol(r.get("Symbol", "")))
    return ce_cnt, pe_cnt

def _format_count_with_ce_pe(rows_list):
    total = len(rows_list or [])
    ce_cnt, pe_cnt = _ce_pe_counts(rows_list)
    return f"({total}) CE({ce_cnt}) PE({pe_cnt})"

def _active_profit_lock(max_r):
    """Returns (trig_r, floor_r) for the highest stepped Profit-Lock tier reached by max_r, or (None, None)."""
    if float(PROFIT_LOCK_TRIGGER_R or 0.0) <= 0:
        return None, None
    steps = PROFIT_LOCK_STEPS or [(float(PROFIT_LOCK_TRIGGER_R), float(PROFIT_LOCK_FLOOR_R))]
    best = (None, None)
    for trig, flr in sorted(steps, key=lambda x: float(x[0])):
        if float(trig) > 0 and max_r + 1e-9 >= float(trig):
            best = (float(trig), float(flr))
    return best

def _compute_target_sl(row):
    ap = float(row.get('AnchorPrice', 0.0) or 0.0)
    ltp = float(row.get('LTP', 0.0) or 0.0)
    anchored = bool(TARGET_SL_FROM_ANCHOR and ap > 0)
    ref_price = ap if anchored else ltp
    engine_atr = float(row.get('BaseAtr', 0.0) or 0.0)
    base_atr = float(row.get('RiskAtr', 0.0) or 0.0) or engine_atr
    if ref_price <= 0 or base_atr <= 0:
        return 0.0, 0.0

    sym = str(row.get('Symbol') or '')
    if OPTIONS_BUYING_ONLY and _is_option_symbol(sym):
        p_dir = 'BULL'
    else:
        p_dir = str(row.get('PremiumDir') or row.get('AnchorDir') or 'BULL').upper()

    t_mult = float(TARGET_ATR_MULT)
    s_mult = float(SL_ATR_MULT)
    try:
        trail_mult = float(SL_TRAIL_ATR_MULT) if (SL_TRAIL_ATR_MULT not in (None, "") and float(SL_TRAIL_ATR_MULT) > 0) else s_mult
    except Exception:
        trail_mult = s_mult

    peak, fav_r_risk, fav_r_eng = ltp, 0.0, 0.0
    if anchored:
        peak = min(ap, ltp) if p_dir == 'BEAR' else max(ap, ltp)
        try:
            pk, pk_for = row.get('PeakPrice'), row.get('PeakFor')
            if pk is not None and pk_for is not None and float(pk) > 0 and abs(float(pk_for) - ap) < 1e-6:
                peak = min(float(pk), ltp) if p_dir == 'BEAR' else max(float(pk), ltp)
        except Exception:
            pass
        fav_move = (ap - peak) if p_dir == 'BEAR' else (peak - ap)
        fav_r_risk = fav_move / base_atr
        fav_r_eng = fav_move / (engine_atr if engine_atr > 0 else base_atr)

    level = 1
    if TARGET_LADDER_ENABLED and anchored:
        step, max_lv = float(TARGET_LADDER_STEP), max(1, int(TARGET_LADDER_MAX_LEVELS))
        if t_mult > 0 and step > 0 and fav_r_risk + 1e-9 >= t_mult:
            level = min(max_lv, int(((fav_r_risk / t_mult - 1.0) / step + 1e-9) // 1) + 2)
    eff_t_mult = t_mult * (1.0 + (level - 1) * float(TARGET_LADDER_STEP)) if level > 1 else t_mult

    trailed = False
    if p_dir == 'BEAR':
        target_price = max(0.05, ref_price - (eff_t_mult * base_atr))
        init_sl = ref_price + (s_mult * base_atr)
        sl_price = init_sl
        if SL_TRAILING_ENABLED and anchored:
            sl_price = min(init_sl, peak + trail_mult * base_atr)
    else:
        target_price = ref_price + (eff_t_mult * base_atr)
        init_sl = max(0.05, ref_price - (s_mult * base_atr))
        sl_price = init_sl
        if SL_TRAILING_ENABLED and anchored:
            sl_price = max(init_sl, peak - trail_mult * base_atr)
    trailed = abs(sl_price - init_sl) > 1e-9

    # Profit-Lock arming & floor synced to engine_atr so displayed SL matches tripwire ratchet
    lock_atr = engine_atr if engine_atr > 0 else base_atr
    pl_trig, pl_floor = _active_profit_lock(fav_r_eng)
    if SL_TRAIL_TO_LOCK_FLOOR and anchored and lock_atr > 0 and pl_trig is not None:
        floor_r = pl_floor
        if p_dir == 'BEAR':
            lock_sl = ref_price - floor_r * lock_atr
            if lock_sl > ltp and lock_sl < sl_price:
                sl_price, trailed = lock_sl, True
        else:
            lock_sl = ref_price + floor_r * lock_atr
            if 0.05 < lock_sl < ltp and lock_sl > sl_price:
                sl_price, trailed = lock_sl, True

    if isinstance(row, dict):
        row['TargetLevel'] = level
        row['SLTrailed'] = bool(trailed)
    return round(target_price, 2), round(sl_price, 2)

def compute_risk_reward_ratio(row):
    tgt_p = float(row.get('TargetPrice', 0.0) or 0.0)
    sl_p = float(row.get('SLPrice', 0.0) or 0.0)
    if tgt_p <= 0 or sl_p <= 0:
        tgt_p, sl_p = _compute_target_sl(row)
    ltp = float(row.get('LTP', 0.0) or 0.0)
    if ltp <= 0 or tgt_p <= 0 or sl_p <= 0:
        return 999.0, "-:1"

    sym = str(row.get('Symbol') or '')
    if OPTIONS_BUYING_ONLY and _is_option_symbol(sym):
        p_dir = 'BULL'
    else:
        p_dir = str(row.get('PremiumDir') or row.get('AnchorDir') or 'BULL').upper()

    if p_dir == 'BEAR':
        reward = max(0.0, ltp - tgt_p)
        risk = max(0.0, sl_p - ltp)
    else:
        reward = max(0.0, tgt_p - ltp)
        risk = max(0.0, ltp - sl_p)

    if risk <= 0:
        return (0.0 if RR_RATIO_TYPE == "REWARD_TO_RISK" else 99.9), "SL hit"

    if RR_RATIO_TYPE == "REWARD_TO_RISK":
        val = (reward / risk) if risk > 0 else 99.9
    else:
        val = (risk / reward) if reward > 0 else 99.9

    val = round(min(val, 99.9), 2)
    return val, f"{val:.2f}:1"

def _target_tag(row):
    try:
        lvl = int(row.get('TargetLevel', 1) or 1)
    except Exception:
        lvl = 1
    return f"T{lvl}" if lvl > 1 else ""

def _target_cell_text(row):
    return f"{_target_tag(row):<3}{float(row.get('TargetPrice', 0.0) or 0.0):>9.2f}"

def _hhmm_to_min(s):
    try:
        clean = str(s).replace('*', '').strip()
        if ' ' in clean:
            clean = clean.split()[-1]
        h, m_ = clean.split(':')[:2]
        return int(h) * 60 + int(m_)
    except Exception:
        return 0

def _anchor_sort_key(row):
    """Sortable timestamp key for ActiveAnchor (handles multi-day swing dates + intraday HH:MM)."""
    ad = str(row.get('AnchorDate') or '').strip()
    tm = _hhmm_to_min(row.get('ActiveAnchor'))
    return (ad, tm)

def _format_anchor_label(dt_val, target_dt=None):
    """Returns 'HH:MM' for same-day anchors, or 'MM-DD HH:MM' for multi-day swing anchors."""
    ts = pd.to_datetime(dt_val)
    if target_dt is not None and ts.date() != target_dt:
        return ts.strftime("%m-%d %H:%M")
    return ts.strftime("%H:%M")

def _attempt_history(row):
    d_raw = str(row.get('PremiumDir') or row.get('AnchorDir') or '').upper()
    label = str(row.get('AnchorDir') or d_raw or 'BULL').upper()
    cur_anchor = str(row.get('ActiveAnchor') or '').replace('*', '').strip()
    atr = float(row.get('BaseAtr', 0.0) or 0.0)
    ks = [k for k in (row.get('Kills') or []) if str(k.get('dir', '')).upper() == d_raw and str(k.get('anchor_time', '')) != cur_anchor]
    ks.sort(key=lambda k: (str(k.get('anchor_date', '')), str(k.get('anchor_time', ''))))
    prior = [(str(k.get('anchor_time', '?')), str(k.get('killed_time', '?')), float(_kill_r_multiple(k, atr))) for k in ks]
    peaks = []
    for k in ks:
        try:
            ap, pk = float(k.get('anchor_price', 0.0) or 0.0), float(k.get('peak_price', 0.0) or 0.0)
            k_atr = float(k.get('eff_atr') or atr or 0.0)
            peaks.append(None if (ap <= 0 or pk <= 0 or k_atr <= 0) else ((pk - ap) / k_atr if d_raw != 'BEAR' else (ap - pk) / k_atr))
        except Exception:
            peaks.append(None)
    return {"label": label, "cur_anchor": cur_anchor, "prior": prior, "peaks": peaks, "kills": ks, "cur_r": float(_r_multiple(row, include_banked=False))}

def _attempt_tier(row):
    try:
        h = _attempt_history(row)
        try: floor_r = max(0.0, float(PRIOR_ATTEMPT_MIN_POSITIVE_R or 0.0))
        except Exception: floor_r = 0.0
        prior_rs = [r for _, _, r in h["prior"]]
        rs = prior_rs + [h["cur_r"]]
        if all(round(r, 2) >= max(round(floor_r, 2), 0.01) - 1e-9 for r in prior_rs) and round(h["cur_r"], 2) >= 0:
            return 1
        return 2 if round(sum(rs), 2) >= 0 else 3
    except Exception:
        return 3

def _prior_attempts_qualify(prior_rs):
    if not PURE_ALLOW_REANCHOR_WITH_POSITIVE_PRIOR or not prior_rs:
        return False
    try:
        floor_r = max(0.0, float(PRIOR_ATTEMPT_MIN_POSITIVE_R or 0.0))
    except Exception:
        floor_r = 0.0
    rs = [round(float(r), 2) for r in prior_rs]
    pos = lambda r: (r >= round(floor_r, 2) - 1e-9) and r > 0
    mode = str(PURE_REANCHOR_PRIOR_MODE or "ANY").strip().upper()
    if mode == "NONE":
        return True
    if mode == "NET":
        return pos(round(sum(rs), 2))
    if mode == "ALL":
        return all(pos(r) for r in rs)
    return any(pos(r) for r in rs)

def _pure_label():
    if _is_swing_mode():
        return f"Fresh Swing ({int(SWING_FRESH_SESSIONS)}d)"
    if not PURE_ALLOW_REANCHOR_WITH_POSITIVE_PRIOR:
        return "Pure 1st-Time (No *)"
    return "Pure 1st-Time + Re-anchor" if PURE_INCLUDE_PLAIN_FIRST_TIME else "Re-anchor only"

def _pure_note_text():
    if _is_swing_mode():
        return (
            f"Swing Freshness: anchors triggered within the last {int(SWING_FRESH_SESSIONS)} trading session(s) "
            f"(multi-day carryover window: {int(SWING_LOOKBACK_SESSIONS)} sessions, gap-stitched {int(CANDLE_TF_MIN)}m |Δ| bars, "
            f"{ATR_BASIS_TF} ATR basis, {RISK_ATR_TF} Risk ATR, causal Charm/Vanna-neutralized |Δ| signal, live |Δ| zone {SWING_DELTA_MIN_ABS:.2f}-{SWING_DELTA_MAX_ABS:.2f})."
        )
    if not PURE_ALLOW_REANCHOR_WITH_POSITIVE_PRIOR:
        return "Pure list = first anchor of the day only (rows with '*' are kept out)."
    min_r = POPUP_MIN_R_MULTIPLE if PURE_ANCHOR_MIN_R is None else PURE_ANCHOR_MIN_R
    base = f"All rows: still holding, current R >= {float(min_r):g}"
    if PURE_ANCHOR_REQUIRE_GRID and POPUP_REQUIRE_NON_ZERO_GRID:
        base += ", ATR/BB grid not all 0"
    mode = str(PURE_REANCHOR_PRIOR_MODE or "ANY").strip().upper()
    try: mr = float(PRIOR_ATTEMPT_MIN_POSITIVE_R or 0.0)
    except Exception: mr = 0.0
    pos = f"+R (>= {mr:g}R)" if mr > 0 else "+R"
    rule = {"NET": f"earlier attempts net {pos}", "ALL": f"every earlier attempt {pos}", "NONE": "any earlier attempt"}.get(mode, f"at least one earlier attempt {pos}")
    gap_txt = ""
    if not PURE_ANCHOR_IGNORE_GAP and ENABLE_TIMESTAMP_GAP_CHECK and MAX_TIMESTAMP_GAP_MINUTES > 0:
        gap_txt = f" (Seen-Anchor gap <= {MAX_TIMESTAMP_GAP_MINUTES}m)"
    lb = int(PURE_ANCHOR_LOOKBACK_MIN or 0)
    first_txt = f"1ST-TIME = first anchor of the day, Anchor column inside the last {lb}m{gap_txt}" if PURE_INCLUDE_PLAIN_FIRST_TIME else "first anchors of the day are not listed here"
    re_when = "re-anchored at ANY time after the first anchor today" if PURE_REANCHOR_ANY_TIME else f"re-anchor inside the same {lb}-minute window"
    return f"{base}. {first_txt}. RE-ANCHOR+ = {re_when}, with {rule} (see the ↳ line)."

def _popup_row_ok(row, any_time=False):
    if _is_swing_mode():
        return _passes_strict_popup_quality(row, check_gap=False, check_grid=False, min_r=PURE_ANCHOR_MIN_R)
    return _passes_strict_popup_quality(row, check_gap=(not PURE_ANCHOR_IGNORE_GAP and not any_time), check_grid=PURE_ANCHOR_REQUIRE_GRID, min_r=PURE_ANCHOR_MIN_R)

ATTEMPT_TIER_ICONS = {1: "✅", 2: "🟡", 3: "🔻"}

def _delta_bb_text(row):
    d = row.get("DeltaBB")
    if not isinstance(d, dict): return ""
    try:
        _ad = float(row.get("AnchorDelta", 0.0) or 0.0)
        _cd = float(row.get("CurrDelta", 0.0) or 0.0)
        _pkd = float(row.get("PeakDelta", 0.0) or 0.0)
        if _ad > 0 and _cd > 0:
            prog = f" [|Δ| {_ad:.3f}→pk {_pkd:.3f}→{_cd:.3f}]" if (_pkd > _cd + 0.015) else f" [|Δ| {_ad:.3f}→{_cd:.3f}]"
        else:
            prog = ""
        inc = row.get("InceptionDelta")
        inc_txt = ""
        if isinstance(inc, dict) and inc.get("first_date"):
            inc_txt = f" | Life({inc['first_date']}) μ={float(inc['mean']):.3f} z={float(inc['z']):+.1f}"
        return f"Δ-BB ✔ |Δ| {float(d['delta']):.3f} > U {float(d['upper']):.3f} (z {float(d['z']):+.1f}){prog}{inc_txt}"
    except Exception:
        return ""

def _prior_attempts_text(row):
    """The '↳' line under every row: attempts on the script + Dual-Band & Delta-BB proofs."""
    parts = [p for p in (_prior_attempts_core(row), _dual_band_text(row), _delta_bb_text(row)) if p]
    return " | ".join(parts)

def _prior_attempts_core(row):
    if not SHOW_PRIOR_ATTEMPTS_LINE:
        return ""
    try:
        h = _attempt_history(row)
        prior, lab, cur, cur_r = h["prior"], h["label"], h["cur_anchor"], h["cur_r"]
        if not prior and not SHOW_ATTEMPTS_LINE_FOR_ALL_ROWS:
            return ""
        icon = ATTEMPT_TIER_ICONS.get(_attempt_tier(row), "")
        day = float(row.get('DayChangePct', 0.0) or 0.0)
        open_pct = row.get('OpenChangePct')
        day_txt = f"Day {day:+.2f}%"
        if open_pct is not None and abs(float(open_pct) - day) >= 2.0:
            day_txt = f"Day {day:+.2f}% (Open {float(open_pct):+.2f}%)"
        days_held = int(row.get('DaysHeld', 0) or 0)
        held_txt = f", {days_held}d held" if (_is_swing_mode() and days_held > 0) else ""
        scope_lbl = f"last {int(SWING_LOOKBACK_SESSIONS)}d" if _is_swing_mode() else "today"
        if not prior:
            return f"{icon} Earlier {lab} attempts (0) → first {lab} anchor ({scope_lbl}) {cur} (now {cur_r:+.2f}R{held_txt}) | {day_txt}"
        net = sum(r for _, _, r in prior)
        n_show = int(PRIOR_ATTEMPTS_MAX_SHOWN)
        peaks = list(h.get("peaks") or [None] * len(prior))
        ks_by_ix = {i_: k_ for i_, k_ in enumerate(h.get("kills") or [])}
        idx_shown = list(range(len(prior)))[-n_show:] if n_show > 0 else list(range(len(prior)))
        fmt_r = lambda v: (f"{v:+.2f}R" if abs(v) < 1.0 else f"{v:+.1f}R")
        parts = []
        for ix in idx_shown:
            a, k, r = prior[ix]
            pk = peaks[ix] if ix < len(peaks) else None
            kd = ks_by_ix.get(ix) or {}
            try: px_txt = f"px {float(kd.get('anchor_price')):.2f}→{float(kd.get('killed_price')):.2f}" if kd else ""
            except Exception: px_txt = ""
            pk_txt = f"peak {pk:+.1f}R" if (pk is not None and pk - r >= 0.25) else ""
            extra = ", ".join(x for x in (px_txt, pk_txt) if x)
            parts.append(f"{a}>{k} {fmt_r(r)}" + (f" ({extra})" if extra else ""))
        more = f"{len(prior) - len(idx_shown)} older, " if len(prior) > len(idx_shown) else ""
        return (f"{icon} Earlier {lab} attempts ({len(prior)}, net {fmt_r(net)}): {more}{' | '.join(parts)} "
                f"→ re-anchored {cur} (now {cur_r:+.2f}R{held_txt}) | first {lab} anchor {prior[0][0]} | {day_txt}")
    except Exception:
        return ""

def _report_legend_text():
    parts = []
    if _is_swing_mode():
        parts.append(
            f"SWING Option-Delta Engine: {int(CANDLE_TF_MIN)}m gap-stitched candles (overnight gaps ignored) | "
            f"Carryover: {int(SWING_LOOKBACK_SESSIONS)} sessions (Fresh <= {int(SWING_FRESH_SESSIONS)}d) | "
            f"Causal Charm/Vanna-neutralized |Δ| signal, live |Δ| {SWING_DELTA_MIN_ABS:.2f}-{SWING_DELTA_MAX_ABS:.2f}, "
            f"|Δ| trail {SWING_DELTA_TRAIL_ATR_MULT:g}x Δ-ATR"
        )
    if ATTEMPT_SORT_ENABLED:
        parts.append("Order: ✅ every earlier attempt +R → 🟡 net +R (an earlier attempt was flat or lost) → 🔻 net -R, newest anchor first inside each group")
    if SL_TRAILING_ENABLED:
        try: _tm = float(SL_TRAIL_ATR_MULT) if (SL_TRAIL_ATR_MULT not in (None, "") and float(SL_TRAIL_ATR_MULT) > 0) else float(SL_ATR_MULT)
        except Exception: _tm = float(SL_ATR_MULT)
        parts.append(f"SL trails the best price since the anchor ({_tm:g} ATR behind, never loosens; 'SL hit' = LTP already through it)")
    if TARGET_LADDER_ENABLED:
        parts.append("T2/T3.. = target stepped up (earlier target achieved)")
    _rt = str(RISK_ATR_TF if RISK_ATR_TF is not None else "").strip()
    _gap_lbl = "gap-free " if ATR_IGNORE_GAPS else ""
    if _rt and _rt.lower() not in ("none", "off", "same", "0", "false") and _rt != str(ATR_BASIS_TF):
        parts.append(f"R = move ÷ {_gap_lbl}{ATR_BASIS_TF} ATR (engine); Target / SL distances use {_gap_lbl}{_rt} ATR")
    else:
        parts.append(f"R = move ÷ {_gap_lbl}{ATR_BASIS_TF} ATR (also used for Target / SL)")
    parts.append(f"Candles: {int(CANDLE_TF_MIN or 1)}min")
    parts.append("↳ = attempts on this script (earlier stopped-out anchors with their prices, current one, Day %)")
    if _dual_gate_required(None, _RUN_MODE):
        parts.append(
            f"CE/PE Dual-Band gate: MANDATORY (premium-based IV-trap guard, "
            f"{int(DUAL_BAND_PERIOD)}-bar {'session-isolated' if ISOLATE_SESSION_BANDS else 'gap-stitched'} BB ±{_dual_eff_std_mult():g}σ on active leg, "
            f"twin normalised by {str(DUAL_BAND_ALIGN_MODE).upper()}, twin max stale {_dual_eff_max_stale():g}m, "
            f"reversal exit {'ON' if DUAL_BAND_REVERSAL_EXIT else 'OFF'}, missing twin = no anchor) | "
            f"Indicators/BB-KC/Δ-BB: Black-Scholes |Δ| (MANDATORY {int(DELTA_BB_PERIOD)}-bar {'session-isolated' if ISOLATE_SESSION_BANDS else 'gap-stitched'} |Δ| > Upper BB +{float(DELTA_BB_STD_MULT):g}σ)"
        )
    return "Legend: " + " | ".join(parts)

def _get_timestamp_gap_min(row):
    """Computes elapsed minutes between CheckpointTime (Seen Time) and ActiveAnchor time."""
    try:
        anc_str = str(row.get('ActiveAnchor') or '').replace('*', '').strip()
        if not anc_str or anc_str == '-':
            return 9999
        if _is_swing_mode() and int(row.get('DaysHeld', 0) or 0) > 0:
            return int(row.get('DaysHeld', 0)) * 375
        cp_dt = row.get('CheckpointTime')
        if not cp_dt:
            return 0
        if ' ' in anc_str:
            anc_str = anc_str.split()[-1]
        anc_t = datetime.strptime(anc_str, "%H:%M").time()
        if hasattr(cp_dt, 'hour'):
            cp_min = cp_dt.hour * 60 + cp_dt.minute
        else:
            cp_t = datetime.strptime(str(cp_dt), "%H:%M").time()
            cp_min = cp_t.hour * 60 + cp_t.minute
        anc_min = anc_t.hour * 60 + anc_t.minute
        gap = cp_min - anc_min
        return gap if gap >= 0 else gap + 1440
    except Exception:
        return 0

def resolve_primary_sector(symbol):
    clean_sym = str(symbol).upper().strip()
    matches = []
    for sec_name, weights_map in SECTOR_CONSTITUENT_WEIGHTS.items():
        if clean_sym in weights_map:
            matches.append((sec_name, float(weights_map[clean_sym])))
    if not matches:
        return DEFAULT_UNMAPPED_SECTOR, float(DEFAULT_UNMAPPED_WEIGHT), [(DEFAULT_UNMAPPED_SECTOR, float(DEFAULT_UNMAPPED_WEIGHT))]
    matches.sort(key=lambda x: (-x[1], x[0]))
    best_sector, best_weight = matches[0]
    return best_sector, best_weight, matches

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
_MASTER_CACHE = {}
_MASTER_CACHE_LOCK = threading.Lock()
_MASTER_1M_CACHE = {}
_MASTER_1M_LOCK = threading.Lock()

def _limiter_for(url):
    if "market-quote/quotes" in url: return LIMITERS["quotes"]
    if "/intraday/" in url: return LIMITERS["intraday"]
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
        self.total = max(1, total)
        self.n = 0
        self.step = max(1, self.total // PROGRESS_STEPS)
        self.lock = threading.Lock()

    def tick(self):
        with self.lock:
            self.n += 1
            if self.n == self.total or self.n % self.step == 0:
                print(f"\r   {self.label}: {self.n}/{self.total}", end="", file=sys.stderr, flush=True)

    def done(self):
        print("", file=sys.stderr)

# ==============================================================================
# 1. UPSTOX API & INSTRUMENT MASTER RESOLUTION
# ==============================================================================
def _get(url, params=None, retries=None):
    max_retries = API_MAX_RETRIES if retries is None else retries
    token = get_upstox_token()
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
            if code == 200: return 200, r.json()
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
    if not candles: return None
    df = pd.DataFrame(candles).iloc[:, :6]
    df.columns = ['Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume']
    ts = df['Timestamp'].astype(str)
    try:
        df['Datetime'] = pd.to_datetime(ts.str.slice(0, 19), format="%Y-%m-%dT%H:%M:%S")
    except ValueError:
        df['Datetime'] = pd.to_datetime(ts, utc=True).dt.tz_convert(IST).dt.tz_localize(None)
    for c in ('Open', 'High', 'Low', 'Close', 'Volume'):
        df[c] = df[c].astype(float)
    return df.drop(columns='Timestamp').drop_duplicates(subset='Datetime', keep='last').sort_values('Datetime').reset_index(drop=True)

def _candles(url):
    status, js = _get(url)
    if status != 200 or not js: return status, None
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
    if cutoff_dt <= session_open: return [cutoff_dt]
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
        if status != 200 or not js: return {}
        by_ts = {f"{x['key'].split('|')[0]}:{x['symbol']}": x['key'] for x in b}
        data, out = js.get('data') or {}, {}
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
    if d.weekday() >= 5: return False
    if d == today:
        df = fetch_today(PROBE_KEY)
        return df is not None and not df.empty
    status, df = _candles(_url_range_1m(PROBE_KEY, d, d + timedelta(days=1)))
    return bool(status == 200 and df is not None and (df['Datetime'].dt.date == d).any())

def resolve_session(cutoff_dt, explicit_time=False):
    notes = []
    req_time = cutoff_dt.time()
    now = now_ist().replace(second=0, microsecond=0)
    if cutoff_dt > now:
        notes.append(f"snapshot {cutoff_dt:%Y-%m-%d %H:%M} is in future -> using current {now:%Y-%m-%d %H:%M}")
        cutoff_dt = now
    target = cutoff_dt.date()

    if cutoff_dt <= _open_dt(target):
        why = f"{cutoff_dt:%Y-%m-%d %H:%M} is before {_open_dt(target):%H:%M} open"
    elif target.weekday() >= 5:
        why = f"{target} is a weekend"
    elif _session_traded(target, now.date()):
        return min(cutoff_dt, _close_dt(target)), notes
    elif STATS.auth_failed:
        return cutoff_dt, notes
    else:
        why = f"no trading data for {target}"

    d = target - timedelta(days=1)
    for _ in range(MAX_SESSION_LOOKBACK_DAYS):
        if STATS.auth_failed: return cutoff_dt, notes
        if _session_traded(d, now.date()):
            if explicit_time:
                cand = datetime.combine(d, req_time)
                resolved = min(max(cand, _open_dt(d)), _close_dt(d))
            else:
                resolved = _close_dt(d)
            notes.append(f"{why} -> using last completed session: {resolved:%Y-%m-%d %H:%M}")
            return resolved, notes
        d -= timedelta(days=1)
    return cutoff_dt, notes

def prepare_master(dfs):
    master = (
        pd.concat([d[['Datetime', 'Open', 'High', 'Low', 'Close', 'Volume']] for d in dfs], ignore_index=True)
        .drop_duplicates(subset='Datetime', keep='last')
        .sort_values('Datetime')
        .reset_index(drop=True)
    )
    dt = master['Datetime']
    day = dt.dt.normalize()
    master['Min'] = dt.dt.hour.values * 60 + dt.dt.minute.values
    master['SessId'] = day.values.astype('datetime64[D]').astype('int64')
    master['Session'] = day.dt.date
    return master

def _resample_master(master, tf_min, cutoff_dt=None):
    """Session-aligned OHLCV resampling from 1-minute bars to tf_min (e.g. 3m, 15m).
    Stamps every bar with its canonical bucket OPEN timestamp and CloseDt so option/twin/spot align 1-to-1 and unfinished bars never fire."""
    tf = max(1, int(tf_min or 1))
    if master is None or master.empty:
        return master
    if tf <= 1:
        out = master.copy()
        out['CloseDt'] = out['Datetime'] + pd.to_timedelta(1, unit='m')
        return out
    sid = master['SessId'].values
    mins = np.clip(master['Min'].values, SESSION_OPEN_MIN, SESSION_CLOSE_MIN - 1)
    bidx = (mins - SESSION_OPEN_MIN) // tf
    bucket = sid * SESSION_BUCKET_MULT + bidx
    starts = np.concatenate(([0], np.flatnonzero(np.diff(bucket)) + 1))
    ends = np.concatenate((starts[1:], [len(bucket)])) - 1

    base_days = master['Datetime'].iloc[starts].dt.normalize().values
    bopen_min = SESSION_OPEN_MIN + bidx[starts] * tf
    bclose_min = SESSION_OPEN_MIN + (bidx[starts] + 1) * tf
    dt_vals = base_days + pd.to_timedelta(bopen_min, unit='m').values
    close_dt_vals = base_days + pd.to_timedelta(bclose_min, unit='m').values
    op_vals = master['Open'].values[starts].astype(float)
    hi_vals = np.maximum.reduceat(master['High'].values.astype(float), starts)
    lo_vals = np.minimum.reduceat(master['Low'].values.astype(float), starts)
    cl_vals = master['Close'].values[ends].astype(float)
    vol_vals = np.add.reduceat(master['Volume'].values.astype(float), starts)

    out = pd.DataFrame({
        'Datetime': dt_vals, 'CloseDt': close_dt_vals, 'Open': op_vals, 'High': hi_vals,
        'Low': lo_vals, 'Close': cl_vals, 'Volume': vol_vals,
    })
    if cutoff_dt is not None and not out.empty:
        cut_d = cutoff_dt.date()
        cut_min = cutoff_dt.hour * 60 + cutoff_dt.minute
        keep = ~((out['Datetime'].dt.date.values == cut_d) & (bclose_min > cut_min))
        out = out.loc[keep].reset_index(drop=True)
    dt = out['Datetime']
    day = dt.dt.normalize()
    out['Min'] = dt.dt.hour.values * 60 + dt.dt.minute.values
    out['SessId'] = day.values.astype('datetime64[D]').astype('int64')
    out['Session'] = day.dt.date
    return out

def _unwrap_master(data):
    if isinstance(data, list): return data
    if isinstance(data, dict):
        if isinstance(data.get("data"), list): return data["data"]
        for v in data.values():
            if isinstance(v, list): return v
    return []

def _download_master(name, _is_fallback=False):
    with _MASTER_CACHE_LOCK:
        if name in _MASTER_CACHE: return _MASTER_CACHE[name]
    url = f"{MASTER_BASE_URL}/{name}.json.gz"
    for _ in range(MASTER_DOWNLOAD_RETRIES):
        try:
            resp = requests.get(url, timeout=MASTER_DOWNLOAD_TIMEOUT_SEC)
            if resp.status_code == 200:
                rows = _unwrap_master(json.loads(gzip.decompress(resp.content).decode('utf-8')))
                if rows:
                    with _MASTER_CACHE_LOCK:
                        _MASTER_CACHE[name] = rows
                    return rows
            elif resp.status_code == 404: break
        except Exception:
            pass
        time.sleep(MASTER_RETRY_SLEEP_SEC)
    if not _is_fallback and name.upper() in ("NSE", "BSE"):
        full = _download_master("complete", _is_fallback=True)
        rows = [r for r in full if str(r.get("exchange", "")).upper() == name.upper() or str(r.get("segment", "")).upper().startswith(name.upper())]
        if rows:
            with _MASTER_CACHE_LOCK:
                _MASTER_CACHE[name] = rows
            return rows
    return []

def resolve_pins(cands, needles, label="symbol"):
    pinned = set()
    norm = [(_norm_sym(sym), sym, key) for sym, key in cands]
    for raw in needles:
        n = _norm_sym(raw)
        if not n: continue
        hits = [c for c in norm if c[0] == n] or [c for c in norm if n in c[0]]
        if not hits: continue
        hits.sort(key=lambda c: c[2])
        pinned.update(h[1] for h in hits[:MAX_PINNED_MATCHES])
    return pinned

def _sectoral_indices_universe():
    nse = _download_master("NSE")
    master_idx_map = {}
    for row in (nse or []):
        if str(row.get("segment", "")).upper() == "NSE_INDEX":
            ik = row.get("instrument_key")
            for field_name in ("name", "trading_symbol", "tradingsymbol"):
                val = row.get(field_name)
                if val and ik: master_idx_map[_norm_sym(val)] = ik
            if ik and "|" in ik: master_idx_map[_norm_sym(ik.split("|", 1)[1])] = ik

    out = []
    for sec_name, cfg in SECTORAL_INDEX_CONFIG.items():
        resolved_key = None
        for alias in [sec_name] + list(cfg.get("aliases", [])):
            norm_a = _norm_sym(alias)
            if norm_a in master_idx_map:
                resolved_key = master_idx_map[norm_a]
                break
        if not resolved_key: resolved_key = cfg["spot_key"]
        out.append({"symbol": sec_name, "key": resolved_key, "pinned": True, "is_sector_index": True, "sector": sec_name, "sector_weight": 100.0})
    return out

def _equity_universe(mode, needles=()):
    nse = _download_master("NSE")
    if not nse: return []

    def ts_of(i): return i.get("tradingsymbol", i.get("trading_symbol"))
    def plain(i):
        return (
            i.get("segment") == "NSE_EQ" and ts_of(i) and i.get("instrument_key")
            and (INCLUDE_NON_EQ_SERIES or (i.get("instrument_type") or "EQ") == "EQ")
        )

    fno = {i.get("underlying_symbol") for i in nse if i.get("segment") == "NSE_FO" and i.get("underlying_symbol")}
    everything = {}
    for i in nse:
        if not plain(i): continue
        sym = ts_of(i)
        if not sym: continue
        k = i["instrument_key"]
        if sym not in everything or (i.get("series") == "EQ" and everything[sym].get("series") != "EQ"):
            everything[sym] = {"symbol": sym, "key": k, "series": i.get("series", "EQ")}

    in_mode = {sym: v for sym, v in everything.items() if ((sym in fno) if mode == "STOCK_FNO" else (sym not in fno))}
    pinned = set()
    if needles:
        pinned = resolve_pins([(v["symbol"], (len(v["symbol"]), v["symbol"])) for v in everything.values()], needles, "equity symbol")
    out = {}
    for sym, v in in_mode.items():
        sec_name, sec_wt, all_sec = resolve_primary_sector(v["symbol"])
        out[v["key"]] = dict(v, pinned=v["symbol"] in pinned, is_sector_index=False, sector=sec_name, sector_weight=sec_wt, all_sectors=all_sec)
    for sym, v in everything.items():
        if v["symbol"] in pinned and v["key"] not in out:
            sec_name, sec_wt, all_sec = resolve_primary_sector(v["symbol"])
            out[v["key"]] = dict(v, pinned=True, is_sector_index=False, sector=sec_name, sector_weight=sec_wt, all_sectors=all_sec)
    return list(out.values())

def _resolve_index_spot_price(key, name, cutoff_dt, use_intraday):
    target_dt = cutoff_dt.date()
    if use_intraday:
        df = fetch_today(key)
        if df is not None and not df.empty:
            sub = df[df['Datetime'] < cutoff_dt]
            if not sub.empty: return float(sub['Close'].iloc[-1]), "today's 1-min close"
        if now_ist() - cutoff_dt <= timedelta(minutes=LIVE_QUOTE_FALLBACK_MIN):
            q = fetch_quotes([{"key": key, "symbol": name}], batch=1)
            ltp = q.get(key, {}).get('ltp', 0.0)
            if ltp and ltp > 0: return float(ltp), "live quote"
    status, df = _candles(_url_range_1m(key, target_dt - timedelta(days=SPOT_1M_LOOKBACK_DAYS), target_dt + timedelta(days=1)))
    if status == 200 and df is not None and not df.empty:
        sub = df[df['Datetime'] < cutoff_dt]
        if not sub.empty: return float(sub['Close'].iloc[-1]), "1-min close"
    status, df = _candles(_url_range_daily(key, target_dt - timedelta(days=SPOT_DAILY_LOOKBACK_DAYS), target_dt + timedelta(days=1)))
    if status == 200 and df is not None and not df.empty:
        limit = target_dt if cutoff_dt >= _close_dt(target_dt) else target_dt - timedelta(days=1)
        sub = df[df['Datetime'].dt.date <= limit]
        if not sub.empty: return float(sub['Close'].iloc[-1]), "daily close"
    return 0.0, "no candles returned"

def _field(row, *names):
    for n in names:
        v = row.get(n)
        if v is not None and v != "" and not (isinstance(v, float) and v != v): return v
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
    except Exception: pass
    return None

def _resolve_target_expiry(expiries, selection):
    """Resolve target expiry date from sorted available expiry dates.
    Supports: CURRENT (0), NEXT (1), NEXT2 (2), any integer index, MONTHLY, NEXT_MONTHLY, or explicit date (YYYY-MM-DD / DD-MMM-YYYY)."""
    if not expiries:
        return None
    sel = str(selection or "CURRENT").strip().upper()
    if sel in ("CURRENT", "NEAR", "0"):
        return expiries[0]
    if sel in ("NEXT", "1"):
        return expiries[min(1, len(expiries) - 1)]
    if sel in ("NEXT2", "2"):
        return expiries[min(2, len(expiries) - 1)]
    if sel.isdigit():
        idx = int(sel)
        return expiries[min(max(0, idx), len(expiries) - 1)]
    if sel in ("MONTHLY", "NEXT_MONTHLY"):
        by_ym = {}
        for d in expiries:
            by_ym[(d.year, d.month)] = max(d, by_ym.get((d.year, d.month), d))
        monthlies = sorted(by_ym.values())
        if not monthlies:
            return expiries[0]
        return monthlies[0] if sel == "MONTHLY" else monthlies[min(1, len(monthlies) - 1)]
    parsed = _expiry_to_date(sel)
    if parsed is not None:
        if parsed in expiries:
            return parsed
        ge = [d for d in expiries if d >= parsed]
        return ge[0] if ge else expiries[-1]
    return expiries[0]

def _attach_twin_keys(universe, lookup):
    """Give every option of the universe the instrument key of its CE<->PE twin (same underlying / expiry / strike)."""
    for it in universe:
        twin_type = "PE" if str(it.get('option_type') or '').upper() == "CE" else "CE"
        it['twin_key'] = lookup.get((it.get('underlying'), it.get('expiry'), float(it.get('strike') or 0.0), twin_type))
    return universe

def _select_option_strikes(strikes, spot_price, strikes_down, strikes_up, no_otm=True):
    """Return (ce_window, pe_window) strike sets around spot_price.
    When no_otm=True: CE keeps only ATM/ITM strikes (strike <= spot_price) and PE keeps only ATM/ITM strikes (strike >= spot_price)."""
    if len(strikes) == 0:
        return set(), set()
    if not no_otm:
        atm_i = int(np.abs(strikes - spot_price).argmin())
        lo = max(0, atm_i - max(0, int(strikes_down)))
        hi = min(len(strikes), atm_i + max(0, int(strikes_up)) + 1)
        w = set(strikes[lo:hi].tolist())
        return w, w
    ce_le = np.where(strikes <= spot_price)[0]
    if len(ce_le) > 0:
        ce_atm_i = int(ce_le[-1])
        ce_lo = max(0, ce_atm_i - max(0, int(strikes_down)))
        ce_win = set(strikes[ce_lo:ce_atm_i + 1].tolist())
    else:
        ce_win = {float(strikes[0])}
    pe_ge = np.where(strikes >= spot_price)[0]
    if len(pe_ge) > 0:
        pe_atm_i = int(pe_ge[0])
        pe_hi = min(len(strikes), pe_atm_i + max(0, int(strikes_up)) + 1)
        pe_win = set(strikes[pe_atm_i:pe_hi].tolist())
    else:
        pe_win = {float(strikes[-1])}
    return ce_win, pe_win


def _index_options_universe(cutoff_dt, use_intraday, needles=()):
    target_dt = cutoff_dt.date()
    spot = {}
    for idx, cfg in INDEX_CONFIG.items():
        try: price, _ = _resolve_index_spot_price(cfg["spot_key"], idx, cutoff_dt, use_intraday)
        except Exception: price = 0.0
        if price > 0: spot[idx] = price
    if not spot and not needles: return []

    rows = _download_master("NSE") + _download_master("BSE")
    if not rows: return []

    by_idx = {idx: [] for idx in INDEX_CONFIG}
    for row in rows:
        segment = str(_field(row, "segment") or "").upper()
        if segment and segment not in ("NSE_FO", "BSE_FO"): continue
        ts = str(_field(row, "trading_symbol", "tradingsymbol") or "").upper()
        itype = str(_field(row, "option_type", "instrument_type") or "").upper()
        if itype not in ("CE", "PE"):
            m = re.search(r'(CE|PE)', ts) or re.search(r'\d(CE|PE)$', ts)
            if not m: continue
            itype = m.group(1)
        cands = [str(_field(row, "underlying_symbol") or ""), str(_field(row, "name") or ""), ts.split()[0] if ts else ""]
        idx = next((ALIAS_TO_INDEX[c.upper().strip()] for c in cands if c.upper().strip() in ALIAS_TO_INDEX), None)
        if idx is None: continue
        exp_date = _expiry_to_date(_field(row, "expiry"))
        if exp_date is None: continue
        strike_raw = _field(row, "strike_price", "strike")
        try: strike = float(strike_raw) if strike_raw is not None else 0.0
        except Exception: strike = 0.0
        if strike <= 0:
            m = re.search(r'(\d+(?:\.\d+)?)\s+(?:CE|PE)', ts) or re.search(r'(\d+)(?:CE|PE)$', ts)
            strike = float(m.group(1)) if m else 0.0
        if strike <= 0 or not _field(row, "instrument_key"): continue
        by_idx[idx].append((exp_date, strike, itype, row))

    universe, seen = [], set()
    if needles:
        flat = []
        for idx, contracts in by_idx.items():
            for exp_date, strike, itype, row in contracts:
                if exp_date < target_dt: continue
                key = _field(row, "instrument_key")
                sym = str(_field(row, "trading_symbol", "tradingsymbol") or key)
                flat.append((sym, (exp_date, strike, itype, sym, idx), key))
        sym_to_item = {sym: (key, exp_date, strike, itype, idx_) for sym, (exp_date, strike, itype, _, idx_), key in flat}
        for sym in sorted(resolve_pins([(s, k) for s, _, k in flat], needles, "option contract")):
            key, exp_date, strike, itype, idx = sym_to_item[sym]
            if key not in seen:
                seen.add(key)
                universe.append({
                    "key": key, "symbol": sym, "pinned": True, "is_sector_index": False,
                    "sector": "INDEX_OPTIONS", "sector_weight": 100.0,
                    "strike": float(strike), "option_type": itype, "expiry": exp_date, "underlying": idx
                })

    for idx, contracts in by_idx.items():
        if idx not in spot or not contracts: continue
        min_exp_dt = (target_dt + timedelta(days=1)) if INDEX_AUTO_ROLL_ON_EXPIRY_DAY else target_dt
        expiries = sorted({c[0] for c in contracts if c[0] >= min_exp_dt}) or sorted({c[0] for c in contracts if c[0] >= target_dt})
        if not expiries: continue
        target_exp = _resolve_target_expiry(expiries, EXPIRY_SELECTION)
        chain = [c for c in contracts if c[0] == target_exp]
        strikes = np.array(sorted({c[1] for c in chain}))
        ce_win, pe_win = _select_option_strikes(strikes, spot[idx], STRIKES_BELOW_ATM, STRIKES_ABOVE_ATM, no_otm=NO_OTM_STRIKES)
        for exp_date, strike, itype, row in chain:
            key = _field(row, "instrument_key")
            target_win = ce_win if itype == "CE" else pe_win
            if strike in target_win and key not in seen:
                seen.add(key)
                universe.append({
                    "key": key, "symbol": str(_field(row, "trading_symbol", "tradingsymbol") or key),
                    "pinned": False, "is_sector_index": False, "sector": idx, "sector_weight": 100.0,
                    "strike": float(strike), "option_type": itype, "expiry": exp_date, "underlying": idx
                })
    return _attach_twin_keys(universe, {(i_, e_, float(k_), t_): _field(r_, "instrument_key") for i_, cs_ in by_idx.items() for e_, k_, t_, r_ in cs_})


def _stock_options_universe(cutoff_dt, use_intraday, needles=()):
    target_dt = cutoff_dt.date()
    nse = _download_master("NSE")
    if not nse: return []

    def ts_of(i): return i.get("tradingsymbol", i.get("trading_symbol"))

    # 1. Map underlying F&O stocks to their cash equity keys
    eq_keys = {}
    for r in nse:
        if r.get("segment") == "NSE_EQ" and (INCLUDE_NON_EQ_SERIES or (r.get("instrument_type") or "EQ") == "EQ"):
            sym = str(ts_of(r) or "").upper().strip()
            if sym and r.get("instrument_key"):
                eq_keys[sym] = r["instrument_key"]

    # 2. Group all stock option contracts in NSE_FO by underlying
    by_stock = {}
    for row in nse:
        seg = str(_field(row, "segment") or "").upper()
        if seg != "NSE_FO": continue
        itype = str(_field(row, "option_type", "instrument_type") or "").upper()
        ts = str(_field(row, "trading_symbol", "tradingsymbol") or "").upper()
        if itype not in ("CE", "PE", "OPTSTK"):
            if not (ts.endswith("CE") or ts.endswith("PE") or " CE " in ts or " PE " in ts):
                continue
        if itype not in ("CE", "PE"):
            itype = "CE" if ("CE" in ts.split()[-1] or ts.endswith("CE")) else "PE"

        u_sym = str(_field(row, "underlying_symbol") or "").upper().strip()
        if not u_sym:
            m_pre = re.match(r'^([A-Z0-9_\-]+)', ts)
            if m_pre: u_sym = m_pre.group(1)
        if not u_sym or u_sym not in eq_keys: continue

        exp_date = _expiry_to_date(_field(row, "expiry"))
        if exp_date is None or exp_date < target_dt: continue
        strike_raw = _field(row, "strike_price", "strike")
        try: strike = float(strike_raw) if strike_raw is not None else 0.0
        except Exception: strike = 0.0
        if strike <= 0:
            m_stk = re.search(r'(\d+(?:\.\d+)?)\s*(?:CE|PE)', ts)
            if m_stk:
                try: strike = float(m_stk.group(1))
                except Exception: strike = 0.0
        if strike <= 0 or not _field(row, "instrument_key"): continue
        by_stock.setdefault(u_sym, []).append((exp_date, strike, itype, row))

    if not by_stock: return []

    # 3. Resolve spot price for each F&O stock
    spot = {}
    cands_to_fetch = [{"key": eq_keys[u], "symbol": u} for u in by_stock if u in eq_keys]
    if cands_to_fetch:
        if use_intraday:
            q_map = fetch_quotes(cands_to_fetch)
            for c in cands_to_fetch:
                ltp = q_map.get(c["key"], {}).get("ltp", 0.0)
                if ltp and ltp > 0:
                    spot[c["symbol"]] = float(ltp)
        else:
            is_eod = cutoff_dt >= _close_dt(target_dt)
            def _hist_spot_one(c):
                status, df = _candles(_url_range_daily(c["key"], target_dt - timedelta(days=SPOT_DAILY_LOOKBACK_DAYS), target_dt + timedelta(days=1)))
                if status == 200 and df is not None and not df.empty:
                    dates = df['Datetime'].dt.date
                    if is_eod:
                        sub = df[dates <= target_dt]
                        if not sub.empty:
                            return c["symbol"], float(sub['Close'].iloc[-1])
                    else:
                        sub_today = df[dates == target_dt]
                        if not sub_today.empty and float(sub_today['Open'].iloc[-1]) > 0:
                            return c["symbol"], float(sub_today['Open'].iloc[-1])
                        sub_prev = df[dates < target_dt]
                        if not sub_prev.empty:
                            return c["symbol"], float(sub_prev['Close'].iloc[-1])
                return c["symbol"], 0.0
            with concurrent.futures.ThreadPoolExecutor(max_workers=WORKERS) as ex:
                for sym_, px_ in ex.map(_hist_spot_one, cands_to_fetch):
                    if px_ > 0:
                        spot[sym_] = px_
            missing_cands = [c for c in cands_to_fetch if c["symbol"] not in spot]
            if missing_cands:
                q_map = fetch_quotes(missing_cands)
                for c in missing_cands:
                    ltp = q_map.get(c["key"], {}).get("ltp", 0.0)
                    if ltp and ltp > 0:
                        spot[c["symbol"]] = float(ltp)

    universe, seen = [], set()

    # 4. Handle pinned / needles contracts (scoped to target expiry & near-ATM when tracking an underlying symbol)
    if needles:
        min_exp_dt = (target_dt + timedelta(days=1)) if STOCK_FNO_AUTO_ROLL_ON_EXPIRY_DAY else target_dt
        pinned_cands = []
        for u_sym, contracts in by_stock.items():
            expiries = sorted({c[0] for c in contracts if c[0] >= min_exp_dt}) or sorted({c[0] for c in contracts if c[0] >= target_dt})
            target_exp = _resolve_target_expiry(expiries, EXPIRY_SELECTION) if expiries else None
            stk_spot = spot.get(u_sym, 0.0)
            for exp_date, strike, itype, row in contracts:
                key = _field(row, "instrument_key")
                opt_sym = str(_field(row, "trading_symbol", "tradingsymbol") or key)
                for needle in needles:
                    n_clean = _norm_sym(needle)
                    if not n_clean: continue
                    exact_contract = (n_clean == _norm_sym(opt_sym))
                    underlying_match = (n_clean == _norm_sym(u_sym))
                    if exact_contract or (underlying_match and target_exp is not None and exp_date == target_exp):
                        dist = abs(float(strike) - stk_spot) if stk_spot > 0 else 0.0
                        pinned_cands.append((0 if exact_contract else 1, dist, exp_date, strike, itype, key, opt_sym, u_sym))
        pinned_cands.sort(key=lambda x: (x[0], x[1], x[2], x[3], x[4]))
        for _, _, exp_date, strike, itype, key, opt_sym, u_sym in pinned_cands[:MAX_PINNED_MATCHES]:
            if key not in seen:
                seen.add(key)
                sec_name, sec_wt, _ = resolve_primary_sector(u_sym)
                stk_spot_ref = spot.get(u_sym, 0.0) or float(strike)
                universe.append({
                    "key": key, "symbol": opt_sym, "pinned": True, "is_sector_index": False,
                    "sector": sec_name, "sector_weight": sec_wt,
                    "strike": float(strike), "option_type": itype, "expiry": exp_date, "underlying": u_sym,
                    "underlying_key": eq_keys.get(u_sym), "spot_price": float(stk_spot_ref)
                })

    # 5. Build ATM options chain for all F&O stocks (current expiry, auto-roll only on expiry day)
    for u_sym, contracts in by_stock.items():
        if not contracts: continue
        min_exp_dt = (target_dt + timedelta(days=1)) if STOCK_FNO_AUTO_ROLL_ON_EXPIRY_DAY else target_dt
        expiries = sorted({c[0] for c in contracts if c[0] >= min_exp_dt}) or sorted({c[0] for c in contracts if c[0] >= target_dt})
        if not expiries: continue
        target_exp = _resolve_target_expiry(expiries, EXPIRY_SELECTION)
        chain = [c for c in contracts if c[0] == target_exp]
        strikes = np.array(sorted({c[1] for c in chain}))

        stock_spot = spot.get(u_sym, 0.0)
        if stock_spot <= 0:
            stock_spot = float(np.median(strikes))

        ce_win, pe_win = _select_option_strikes(strikes, stock_spot, STOCK_OPT_STRIKES_BELOW_ATM, STOCK_OPT_STRIKES_ABOVE_ATM, no_otm=NO_OTM_STRIKES)

        sec_name, sec_wt, _ = resolve_primary_sector(u_sym)
        for exp_date, strike, itype, row in chain:
            key = _field(row, "instrument_key")
            target_win = ce_win if itype == "CE" else pe_win
            if strike in target_win and key not in seen:
                seen.add(key)
                universe.append({
                    "key": key, "symbol": str(_field(row, "trading_symbol", "tradingsymbol") or key),
                    "pinned": False, "is_sector_index": False, "sector": sec_name, "sector_weight": sec_wt,
                    "strike": float(strike), "option_type": itype, "expiry": exp_date, "underlying": u_sym,
                    "underlying_key": eq_keys.get(u_sym), "spot_price": float(stock_spot)
                })
    return _attach_twin_keys(universe, {(u_, e_, float(k_), t_): _field(r_, "instrument_key") for u_, cs_ in by_stock.items() for e_, k_, t_, r_ in cs_})

def get_dynamic_universe(mode, cutoff_dt, use_intraday, needles=()):
    if mode == "CASH_EQUITY": return _equity_universe(mode, needles)
    if mode in ("STOCK_FNO", "SWING_FNO"): return _stock_options_universe(cutoff_dt, use_intraday, needles)
    if mode in ("INDEX_OPTIONS", "SWING_INDEX"): return _index_options_universe(cutoff_dt, use_intraday, needles)
    return []

# ==============================================================================
# 2. CONTINUOUS KINETIC TRIPWIRE ENGINE (SHARED BY INDICES, OPTIONS & SWING)
# ==============================================================================
def _tf_to_minutes(tf, default=15):
    t = str(tf if tf is not None else "").strip().lower()
    try:
        if t.endswith("h") or t.endswith("hr") or t.endswith("hour") or t.endswith("hours"):
            return max(1, int(round(float(t.rstrip("hours").rstrip("hr").rstrip("h") or 0) * 60)))
        for suf in ("minutes", "minute", "mins", "min", "m"):
            if t.endswith(suf):
                t = t[:-len(suf)]
                break
        return max(1, int(float(t)))
    except Exception:
        return int(default)

def _atr_fallback_minutes():
    t = str(ATR_FALLBACK_TF if ATR_FALLBACK_TF is not None else "").strip().lower()
    if t in ("", "off", "none", "0", "false", "no"):
        return 0
    return _tf_to_minutes(t, default=15)

def compute_base_atr(m, tf=None, ignore_gaps=None):
    if m is None or len(m) == 0: return None, 0
    if ignore_gaps is None:
        ignore_gaps = bool(ATR_IGNORE_GAPS)
    hi, lo, cl = m['High'].values, m['Low'].values, m['Close'].values
    sid = m['SessId'].values
    mins = np.clip(m['Min'].values, SESSION_OPEN_MIN, SESSION_CLOSE_MIN - 1)
    tf_minutes = _tf_to_minutes(ATR_BASIS_TF if tf is None else tf)
    last_idx = max((int(SESSION_CLOSE_MIN) - int(SESSION_OPEN_MIN)) // tf_minutes - 1, 0)
    bidx = np.minimum((mins - SESSION_OPEN_MIN) // tf_minutes, last_idx)
    bucket = sid * SESSION_BUCKET_MULT + bidx
    starts = np.concatenate(([0], np.flatnonzero(np.diff(bucket)) + 1))
    ends = np.concatenate((starts[1:], [len(bucket)])) - 1
    b_high, b_low, b_close = np.maximum.reduceat(hi, starts), np.minimum.reduceat(lo, starts), cl[ends]
    b_sid = sid[starts]
    prev_close = np.concatenate(([np.nan], b_close[:-1]))
    tr = np.fmax(np.fmax(b_high - b_low, np.abs(b_high - prev_close)), np.abs(b_low - prev_close))
    if ignore_gaps and len(b_sid) > 1:
        b_sess_start = np.zeros(len(b_sid), dtype=bool)
        b_sess_start[0] = True
        b_sess_start[1:] = b_sid[1:] != b_sid[:-1]
        tr = np.where(b_sess_start, b_high - b_low, tr)
    tr_win = tr[-ATR_BASIS_PERIOD:]
    if len(tr_win) >= 5:
        med_tr = float(np.median(tr_win))
        if med_tr > 0:
            tr_win = np.minimum(tr_win, 3.0 * med_tr)
    return float(tr_win.mean()), len(b_close)

def _ewm(x, alpha, sess_starts=None):
    xs = x.tolist()
    n = len(xs)
    if n == 0:
        return np.asarray([], dtype=float)
    out = [0.0] * n
    prev = out[0] = xs[0]
    keep = 1.0 - alpha
    if sess_starts is not None:
        ss = sess_starts.tolist()
        for i in range(1, n):
            if ss[i]:
                prev = out[i] = xs[i]
            else:
                out[i] = prev = keep * prev + alpha * xs[i]
    else:
        for i in range(1, n):
            out[i] = prev = keep * prev + alpha * xs[i]
    return np.asarray(out)


# ==============================================================================
# 1B. OPTION IMPLIED VOLATILITY (IV) MATHEMATICAL ENGINE & KINEMATICS
# ==============================================================================
_SPOT_1M_CACHE = {}
_SPOT_DAILY_CACHE = {}
_OPT_DAILY_CACHE = {}
_SPOT_CACHE_LOCK = threading.Lock()

def _norm_cdf(x):
    return 0.5 * (1.0 + math.erf(x / 1.4142135623730951))

def _bs_price(S, K, T, r, sigma, is_call=True):
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return max(0.0, S - K) if is_call else max(0.0, K - S)
    d1 = (math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    if is_call:
        return S * _norm_cdf(d1) - K * math.exp(-r * T) * _norm_cdf(d2)
    else:
        return K * math.exp(-r * T) * _norm_cdf(-d2) - S * _norm_cdf(-d1)

def _bs_vega(S, K, T, r, sigma):
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0: return 0.0
    d1 = (math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))
    return S * math.sqrt(T) * (1.0 / math.sqrt(2 * math.pi)) * math.exp(-0.5 * d1 * d1)

def _bs_delta(S, K, T, r, sigma, is_call=True):
    """Absolute Black-Scholes Delta |Δ| in [0.0, 1.0]: N(d1) for CE, N(-d1) for PE."""
    if S <= 0 or K <= 0: return 0.0
    if T <= 0 or sigma <= 0:
        return (1.0 if S > K else 0.0) if is_call else (1.0 if K > S else 0.0)
    d1 = (math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))
    return _norm_cdf(d1) if is_call else _norm_cdf(-d1)

def _compute_bs_delta_ohlc(frame, spot_df, strike, exp_dt, is_call, cutoff_dt=None, swing_mode=False):
    """Compute per-bar Black-Scholes IV and |Delta| OHLC aligned to frame's bars using underlying spot OHLC.
    Uses strictly causal IV carry-forward and causal expanding median IV in swing mode (zero future lookahead).
    Returns (iv_arr, d_o, d_h, d_l, d_c, sig_o, sig_h, sig_l, sig_c)."""
    n = len(frame)
    close = frame['Close'].values.astype(float)
    spot_rs = _resample_master(spot_df, CANDLE_TF_MIN, cutoff_dt=cutoff_dt) if int(CANDLE_TF_MIN or 1) > 1 else spot_df
    if spot_rs is None or spot_rs.empty:
        spot_rs = spot_df
    s_dt = spot_rs['Datetime'].values
    s_o = spot_rs['Open'].values.astype(float)
    s_h = spot_rs['High'].values.astype(float)
    s_l = spot_rs['Low'].values.astype(float)
    s_c = spot_rs['Close'].values.astype(float)
    raw_pos = np.searchsorted(s_dt, frame['Datetime'].values, side='right') - 1
    has_spot = raw_pos >= 0
    pos = np.clip(raw_pos, 0, len(s_c) - 1)
    sp_o = np.where(has_spot, s_o[pos], np.nan)
    sp_h = np.where(has_spot, s_h[pos], np.nan)
    sp_l = np.where(has_spot, s_l[pos], np.nan)
    sp_c = np.where(has_spot, s_c[pos], np.nan)

    exp_datetime = datetime.combine(exp_dt, dtime(15, 30))
    candle_dts = frame['Datetime'].dt.to_pydatetime()
    iv_arr = np.zeros(n, dtype=float)
    t_yr_arr = np.zeros(n, dtype=float)
    causal_med_iv = np.full(n, 0.25, dtype=float)
    valid_ivs_so_far = []
    last_valid_iv = 0.25

    for k_i in range(n):
        t_sec = max(60.0, (exp_datetime - candle_dts[k_i]).total_seconds())
        t_yr = t_sec / (365.0 * 86400.0)
        t_yr_arr[k_i] = t_yr
        if has_spot[k_i] and np.isfinite(sp_c[k_i]) and sp_c[k_i] > 0:
            iv_v = calc_implied_volatility(close[k_i], sp_c[k_i], strike, t_yr, r=RISK_FREE_RATE, is_call=is_call)
        else:
            iv_v = 0.0
        if iv_v > 0:
            last_valid_iv = iv_v
            valid_ivs_so_far.append(iv_v)
            iv_arr[k_i] = iv_v
        else:
            # Strictly causal carry-forward (never backfill from future bars)
            iv_arr[k_i] = last_valid_iv
        if valid_ivs_so_far:
            causal_med_iv[k_i] = float(np.median(valid_ivs_so_far))

    d_o = np.full(n, 0.5, dtype=float)
    d_h = np.full(n, 0.5, dtype=float)
    d_l = np.full(n, 0.5, dtype=float)
    d_c = np.full(n, 0.5, dtype=float)
    for k_i in range(n):
        if not has_spot[k_i] or not np.isfinite(sp_c[k_i]):
            if k_i > 0:
                d_o[k_i], d_h[k_i], d_l[k_i], d_c[k_i] = d_o[k_i - 1], d_h[k_i - 1], d_l[k_i - 1], d_c[k_i - 1]
            continue
        t_yr = t_yr_arr[k_i]
        eff_iv = iv_arr[k_i] if iv_arr[k_i] > 0 else causal_med_iv[k_i]
        vo = _bs_delta(sp_o[k_i], strike, t_yr, RISK_FREE_RATE, eff_iv, is_call=is_call)
        vc = _bs_delta(sp_c[k_i], strike, t_yr, RISK_FREE_RATE, eff_iv, is_call=is_call)
        if is_call:
            vh = _bs_delta(sp_h[k_i], strike, t_yr, RISK_FREE_RATE, eff_iv, is_call=True)
            vl = _bs_delta(sp_l[k_i], strike, t_yr, RISK_FREE_RATE, eff_iv, is_call=True)
        else:
            vh = _bs_delta(sp_l[k_i], strike, t_yr, RISK_FREE_RATE, eff_iv, is_call=False)
            vl = _bs_delta(sp_h[k_i], strike, t_yr, RISK_FREE_RATE, eff_iv, is_call=False)
        d_o[k_i] = min(max(vo, 1e-4), 0.9999)
        d_c[k_i] = min(max(vc, 1e-4), 0.9999)
        d_h[k_i] = min(max(max(vh, vo, vc), 1e-4), 0.9999)
        d_l[k_i] = min(max(min(vl, vo, vc), 1e-4), 0.9999)

    if not swing_mode:
        return iv_arr, d_o, d_h, d_l, d_c, d_o, d_h, d_l, d_c

    # Charm & Vanna neutralized signal |Delta| (fixed constant reference horizon T_ref and causal median IV)
    t_ref = float(SWING_CHARM_REF_DTE_DAYS or 15.0) / 365.0
    sig_o = np.full(n, 0.5, dtype=float)
    sig_h = np.full(n, 0.5, dtype=float)
    sig_l = np.full(n, 0.5, dtype=float)
    sig_c = np.full(n, 0.5, dtype=float)
    for k_i in range(n):
        if not has_spot[k_i] or not np.isfinite(sp_c[k_i]):
            if k_i > 0:
                sig_o[k_i], sig_h[k_i], sig_l[k_i], sig_c[k_i] = sig_o[k_i - 1], sig_h[k_i - 1], sig_l[k_i - 1], sig_c[k_i - 1]
            continue
        m_iv = causal_med_iv[k_i]
        vo = _bs_delta(sp_o[k_i], strike, t_ref, RISK_FREE_RATE, m_iv, is_call=is_call)
        vc = _bs_delta(sp_c[k_i], strike, t_ref, RISK_FREE_RATE, m_iv, is_call=is_call)
        if is_call:
            vh = _bs_delta(sp_h[k_i], strike, t_ref, RISK_FREE_RATE, m_iv, is_call=True)
            vl = _bs_delta(sp_l[k_i], strike, t_ref, RISK_FREE_RATE, m_iv, is_call=True)
        else:
            vh = _bs_delta(sp_l[k_i], strike, t_ref, RISK_FREE_RATE, m_iv, is_call=False)
            vl = _bs_delta(sp_h[k_i], strike, t_ref, RISK_FREE_RATE, m_iv, is_call=False)
        sig_o[k_i] = min(max(vo, 1e-4), 0.9999)
        sig_c[k_i] = min(max(vc, 1e-4), 0.9999)
        sig_h[k_i] = min(max(max(vh, vo, vc), 1e-4), 0.9999)
        sig_l[k_i] = min(max(min(vl, vo, vc), 1e-4), 0.9999)

    return iv_arr, d_o, d_h, d_l, d_c, sig_o, sig_h, sig_l, sig_c


def calc_implied_volatility(price, S, K, T, r=0.065, is_call=True):
    intrinsic = max(0.0, S - K) if is_call else max(0.0, K - S)
    if price <= intrinsic or price <= 0.05 or S <= 0 or K <= 0 or T <= 0:
        return 0.0
    sigma = 0.20
    for _ in range(10):
        p = _bs_price(S, K, T, r, sigma, is_call)
        diff = p - price
        if abs(diff) < 1e-3: return sigma
        v = _bs_vega(S, K, T, r, sigma)
        if v < 1e-4: break
        sigma -= diff / v
        if sigma <= 0.01 or sigma > 5.0: break
    low, high = 0.01, 3.5
    mid = 0.25
    for _ in range(12):
        mid = 0.5 * (low + high)
        p = _bs_price(S, K, T, r, mid, is_call)
        if abs(p - price) < 1e-2: return mid
        if p < price: low = mid
        else: high = mid
    return mid

def _parse_contract_specs(symbol, item=None):
    sym = str(symbol or "").upper().strip()
    is_ce = _is_ce_symbol(sym)
    is_pe = _is_pe_symbol(sym)
    if not (is_ce or is_pe): return None
    itype = "CE" if is_ce else "PE"

    strike = 0.0
    if item and item.get("strike"):
        try: strike = float(item["strike"])
        except Exception: pass
    if strike <= 0:
        m = re.search(r'(\d+(?:\.\d+)?)\s*(?:CE|PE)', sym)
        if m:
            try: strike = float(m.group(1))
            except Exception: pass

    underlying = "NIFTY"
    if item and item.get("underlying"):
        underlying = item["underlying"]
    else:
        for idx in ("BANKNIFTY", "FINNIFTY", "SENSEX", "NIFTY"):
            if idx in sym:
                underlying = idx
                break

    expiry_dt = None
    if item and item.get("expiry"):
        expiry_dt = item["expiry"]
    if expiry_dt is None:
        months = {'JAN': 1, 'FEB': 2, 'MAR': 3, 'APR': 4, 'MAY': 5, 'JUN': 6,
                  'JUL': 7, 'AUG': 8, 'SEP': 9, 'OCT': 10, 'NOV': 11, 'DEC': 12}
        m = re.search(r'(\d{1,2})\s*([A-Z]{3})\s*(\d{2,4})?', sym)
        if m:
            day = int(m.group(1))
            mon_str = m.group(2)
            yr_str = m.group(3)
            if mon_str in months:
                yr = int(yr_str) if yr_str else now_ist().year
                if yr < 100: yr += 2000
                try: expiry_dt = datetime(yr, months[mon_str], day).date()
                except Exception: pass
    if expiry_dt is None:
        expiry_dt = now_ist().date()

    return {"symbol": sym, "option_type": itype, "strike": strike, "underlying": underlying, "expiry": expiry_dt}


def _get_cached_spot_1m(idx, target_dt, history_days=MIN1_HISTORY_DAYS, use_intraday=True, spot_key=None):
    if not spot_key:
        cfg = INDEX_CONFIG.get(idx)
        spot_key = cfg.get("spot_key") if cfg else None
    if not spot_key: return None
    cache_key = (idx, target_dt, int(history_days))
    with _SPOT_CACHE_LOCK:
        if cache_key in _SPOT_1M_CACHE:
            return _SPOT_1M_CACHE[cache_key]
    try:
        frames = []
        for c_start, c_end in _date_chunks(target_dt - timedelta(days=history_days), target_dt + timedelta(days=1), span=CHUNK_SPAN_1M_DAYS):
            status, df = _candles(_url_range_1m(spot_key, c_start, c_end))
            if status == 200 and df is not None: frames.append(df)
        if use_intraday:
            today_df = fetch_today(spot_key)
            if today_df is not None: frames.append(today_df)
        if frames:
            master_spot = prepare_master(frames)
            with _SPOT_CACHE_LOCK:
                _SPOT_1M_CACHE[cache_key] = master_spot
            return master_spot
    except Exception as e:
        ERRORS.add(f"spot_1m:{idx}", e)
    return None

def _get_cached_daily(cache_dict, cache_key, inst_key, target_dt, lookback_days=INCEPTION_LOOKBACK_CAL_DAYS):
    if not inst_key:
        return None
    with _SPOT_CACHE_LOCK:
        if cache_key in cache_dict:
            return cache_dict[cache_key]
    try:
        start_d = target_dt - timedelta(days=int(lookback_days or 95))
        status, df = _candles(_url_range_daily(inst_key, start_d, target_dt))
        res = df if (status == 200 and df is not None and not df.empty) else None
    except Exception as e:
        ERRORS.add(f"daily:{cache_key}", e)
        res = None
    with _SPOT_CACHE_LOCK:
        cache_dict[cache_key] = res
    return res

def _build_inception_delta_regime(master, d_c, target_dt, opt_item, opt_specs, spot_df_1m, sig_o=None, sig_c=None):
    """Build Two-Layer Inception-to-Date Macro |Delta| context + rising gap-stitched |Delta| slope gate."""
    n = len(d_c)
    if n == 0 or not ENABLE_INCEPTION_DELTA_REGIME:
        return None
    prior_deltas = []
    first_liq_date = None
    try:
        opt_key = (opt_item or {}).get("key")
        u_idx = (opt_specs or {}).get("underlying", "NIFTY")
        u_key = (opt_item or {}).get("underlying_key") or (INDEX_CONFIG.get(u_idx) or {}).get("spot_key")
        stk = float((opt_specs or {}).get("strike") or 0.0)
        exp_dt = (opt_specs or {}).get("expiry") or target_dt
        is_call = ((opt_specs or {}).get("option_type") == "CE")

        opt_daily = _get_cached_daily(_OPT_DAILY_CACHE, (opt_key, target_dt), opt_key, target_dt) if opt_key else None
        spot_daily = _get_cached_daily(_SPOT_DAILY_CACHE, (u_idx, target_dt), u_key, target_dt) if u_key else None

        if opt_daily is not None and spot_daily is not None and stk > 0:
            od = opt_daily[opt_daily["Datetime"].dt.date < target_dt].copy()
            sd = spot_daily[spot_daily["Datetime"].dt.date < target_dt].copy()
            if not od.empty and not sd.empty:
                liq_mask = od["Volume"].values >= float(INCEPTION_MIN_DAILY_VOLUME or 0)
                if liq_mask.any():
                    first_idx = int(np.argmax(liq_mask))
                    od = od.iloc[first_idx:].reset_index(drop=True)
                if not od.empty:
                    sd_map = {d_: float(c_) for d_, c_ in zip(sd["Datetime"].dt.date, sd["Close"].values)}
                    exp_datetime = datetime.combine(exp_dt, dtime(15, 30))
                    ivs, raw_rows = [], []
                    for dt_v, cl_v in zip(od["Datetime"].dt.date, od["Close"].values):
                        sp_v = sd_map.get(dt_v)
                        if sp_v and sp_v > 0 and cl_v > 0:
                            t_sec = max(60.0, (exp_datetime - datetime.combine(dt_v, dtime(15, 30))).total_seconds())
                            t_yr = t_sec / (365.0 * 86400.0)
                            iv_v = calc_implied_volatility(float(cl_v), float(sp_v), stk, t_yr, r=RISK_FREE_RATE, is_call=is_call)
                            ivs.append(iv_v)
                            raw_rows.append((dt_v, float(sp_v), t_yr, iv_v))
                    if raw_rows:
                        first_liq_date = raw_rows[0][0]
                        valid_ivs = [v for v in ivs if v > 0]
                        med_iv = float(np.median(valid_ivs)) if valid_ivs else 0.25
                        for _, sp_v, t_yr, iv_v in raw_rows:
                            eff_iv = iv_v if iv_v > 0 else med_iv
                            prior_deltas.append(min(max(_bs_delta(sp_v, stk, t_yr, RISK_FREE_RATE, eff_iv, is_call=is_call), 1e-4), 0.9999))
    except Exception as e:
        ERRORS.add("inception_delta_regime", e)

    if not prior_deltas and master is not None and not master.empty:
        prior_mask = (master["Datetime"].dt.date < target_dt).values
        if prior_mask.any():
            p_df = pd.DataFrame({"d": master["Datetime"].dt.date.values[prior_mask], "dc": d_c[prior_mask]})
            daily_last = p_df.groupby("d")["dc"].last()
            prior_deltas = [float(v) for v in daily_last.values if np.isfinite(v)]
            if len(daily_last) > 0:
                first_liq_date = daily_last.index[0]

    min_sig = max(float(DELTA_BB_MIN_SIGMA or 0.005), 1e-6)
    inc_mean = np.full(n, np.nan, dtype=float)
    inc_std = np.full(n, np.nan, dtype=float)
    inc_z = np.full(n, 0.0, dtype=float)

    if prior_deltas:
        base_arr = np.asarray(prior_deltas, dtype=float)
        base_n = len(base_arr)
        base_sum = float(np.sum(base_arr))
        base_sq = float(np.sum(base_arr * base_arr))
        for i in range(n):
            v = float(d_c[i])
            cnt = base_n + 1
            mu = (base_sum + v) / cnt
            var = max(0.0, (base_sq + v * v) / cnt - mu * mu)
            sd = max(math.sqrt(var), min_sig)
            inc_mean[i] = mu
            inc_std[i] = sd
            inc_z[i] = (v - mu) / sd
    else:
        first_liq_date = target_dt
        s_d = pd.Series(d_c)
        inc_mean = s_d.expanding(min_periods=1).mean().values
        inc_std = np.maximum(s_d.expanding(min_periods=1).std(ddof=0).fillna(min_sig).values, min_sig)
        inc_z = (d_c - inc_mean) / inc_std

    # Rising gap-stitched |Delta| slope gate: passes whenever |Delta| is rising above its
    # gap-stitched short-term mean, regardless of whether 2-week lifecycle z is negative.
    sc = np.asarray(sig_c if sig_c is not None else d_c, dtype=float)
    so = np.asarray(sig_o if sig_o is not None else sc, dtype=float)
    sess_starts = np.zeros(n, dtype=bool)
    sess_starts[0] = True
    if master is not None and not master.empty and 'SessId' in master.columns and len(master) == n:
        sid_v = master['SessId'].values
        sess_starts[1:] = sid_v[1:] != sid_v[:-1]
    d_diff = np.diff(sc, prepend=sc[0])
    d_diff = np.where(sess_starts, sc - so, d_diff)
    stitched_sc = np.cumsum(d_diff) + so[0]
    slope_p = max(3, int(INCEPTION_SLOPE_PERIOD or 10))
    st_mean = pd.Series(stitched_sc).rolling(slope_p, min_periods=min(3, slope_p)).mean().values
    st_slope = np.diff(st_mean, prepend=st_mean[0])
    st_above = stitched_sc - st_mean
    ok_bull = np.isfinite(st_mean) & (st_above > 0.0) & (st_slope >= 0.0)

    return {
        "enabled": bool(ENABLE_INCEPTION_DELTA_REGIME),
        "first_date": first_liq_date.strftime("%Y-%m-%d") if first_liq_date else target_dt.strftime("%Y-%m-%d"),
        "prior_days": len(prior_deltas),
        "mean": inc_mean,
        "std": inc_std,
        "z": inc_z,
        "st_above": st_above,
        "st_slope": st_slope,
        "ok_bull": ok_bull,
    }

# ==============================================================================
# 1C. CE/PE DUAL-BAND CONFLUENCE GATE ENGINE (MANDATORY FOR EVERY OPTION ANCHOR)
# ==============================================================================
# Runs on actual CE/PE option premiums (with log-return gap stitching when ISOLATE_SESSION_BANDS=False)
# so it genuinely detects IV-spike traps (both legs > Upper) and IV-crush traps (both legs < Lower).
_DUAL_CODE_TEXT = {
    0: "twin/band data not ready",
    3: "TRAP - CE & PE both above the Upper band (IV spike)",
    4: "TRAP - CE & PE both below the Lower band (IV crush)",
    5: "CHOP - CE & PE inside the band",
    6: "NOT CONFIRMED - only one leg broke its band",
}

def _scaled_warmup_bars():
    """Scale ANCHOR_OPEN_WARMUP_BARS (defined in 1m bars) to the active CANDLE_TF_MIN."""
    tf_scale = max(1, int(CANDLE_TF_MIN or 1))
    return max(2, int(math.ceil(int(ANCHOR_OPEN_WARMUP_BARS or 5) / tf_scale)))

def _is_true_option_item(item):
    """True only for real CE/PE contracts of the option universe."""
    return bool(item) and not item.get('is_sector_index') and str(item.get('option_type') or '').upper() in ("CE", "PE")

def _dual_eff_std_mult():
    return float(STOCK_OPT_DUAL_BAND_STD_MULT if _is_stock_option_mode(_RUN_MODE) else DUAL_BAND_STD_MULT)

def _dual_eff_max_stale():
    base = float(STOCK_OPT_DUAL_BAND_TWIN_MAX_STALE_MIN if _is_stock_option_mode(_RUN_MODE) else DUAL_BAND_TWIN_MAX_STALE_MIN)
    return max(base, float(CANDLE_TF_MIN or 1)) if base > 0 else 0.0

def _dual_gate_required(item, mode, is_sector_index=False):
    """Mandatory for every non-index instrument of an option run, and for any true option contract wherever it comes from."""
    if not DUAL_BAND_GATE_MANDATORY or is_sector_index:
        return False
    return _is_option_mode(mode) or _is_true_option_item(item)

def _closed_dual_gate(own_type, reason):
    """A gate that can never pass: used whenever the twin leg (or the CE/PE identity) is missing. Fail-closed by design."""
    return {'required': True, 'available': False, 'own_type': str(own_type or "").upper(), 'reason': reason}

def _dual_leg_arrays(frame, min_sigma_override=None):
    """Per-leg scale-invariant Bollinger arrays computed strictly on the leg's OWN native bars and OWN SessId.
    Uses multiplicative (log-return) gap stitching across sessions when ISOLATE_SESSION_BANDS=False so overnight gaps
    never leak into the rolling band while preserving multi-day continuity."""
    open_ = np.maximum(frame['Open'].values.astype(float), EPSILON)
    close = np.maximum(frame['Close'].values.astype(float), EPSILON)
    sid = frame['SessId'].values
    n = len(close)
    period = max(2, int(DUAL_BAND_PERIOD))
    _ms = DUAL_BAND_MIN_SIGMA if min_sigma_override is None else min_sigma_override
    floor_rs = max(float(_ms or 0.0), EPSILON)
    log_c = np.log(close)
    log_o = np.log(open_)
    min_p = min(period, _scaled_warmup_bars())

    if ISOLATE_SESSION_BANDS:
        grouped = pd.DataFrame({'lc': log_c, 'sid': sid}).groupby('sid')['lc']
        mu_log = grouped.transform(lambda x: x.rolling(period, min_periods=min_p).mean()).values
        sd_log = grouped.transform(lambda x: x.rolling(period, min_periods=min_p).std(ddof=0)).values
    else:
        sess_starts = np.zeros(n, dtype=bool)
        sess_starts[0] = True
        sess_starts[1:] = sid[1:] != sid[:-1]
        log_ret = np.diff(log_c, prepend=log_c[0])
        log_ret = np.where(sess_starts, log_c - log_o, log_ret)
        stitched_log = np.cumsum(log_ret) + log_o[0]
        log_offset = log_c - stitched_log
        s_log = pd.Series(stitched_log)
        mu_log = s_log.rolling(period, min_periods=min_p).mean().values + log_offset
        sd_log = s_log.rolling(period, min_periods=min_p).std(ddof=0).values

    with np.errstate(all='ignore'):
        rel_floor = np.maximum(floor_rs / close, 1e-4)
        sd_eff = np.maximum(sd_log, rel_floor)
        mid = np.exp(mu_log)
        std_rs = np.maximum(mid * sd_eff, floor_rs)
        z = (log_c - mu_log) / sd_eff

    return {
        'dt': frame['Datetime'].values.astype('datetime64[ns]'),
        'close': close, 'mu_log': mu_log, 'sd_eff': sd_eff,
        'mid': mid, 'std_rs': std_rs, 'z': z,
    }

def _dual_align_twin_arrays(own_dt, twin_arr):
    """Sample the twin leg's ALREADY-COMPUTED native Bollinger stats AS-OF each bar of the host leg."""
    tw_dt = twin_arr['dt']
    pos = np.searchsorted(tw_dt, own_dt, side='right') - 1
    have = pos >= 0
    pos_safe = np.clip(pos, 0, len(tw_dt) - 1)
    last_dt = tw_dt[pos_safe]
    fresh = have & (own_dt.astype('datetime64[D]') == last_dt.astype('datetime64[D]'))
    max_stale = _dual_eff_max_stale()
    if max_stale > 0:
        age_min = (own_dt - last_dt).astype('timedelta64[s]').astype(float) / 60.0
        fresh = fresh & (age_min <= max_stale)

    aligned = {
        k: twin_arr[k][pos_safe]
        for k in ('close', 'mu_log', 'sd_eff', 'mid', 'std_rs', 'z')
    }
    return aligned, fresh

def build_dual_band_gate(master, twin_master, own_type, min_sigma_override=None):
    """Per-bar symmetric CE<->PE confluence arrays on actual option premiums."""
    own_type = str(own_type).upper()
    if own_type not in ("CE", "PE"):
        return _closed_dual_gate(own_type, "contract is neither CE nor PE (fail-closed)")
    if master is None or len(master) == 0 or twin_master is None or len(twin_master) == 0:
        return _closed_dual_gate(own_type, "twin CE/PE candles unavailable (fail-closed)")

    n = len(master)
    own = _dual_leg_arrays(master, min_sigma_override=min_sigma_override)
    twin_native = _dual_leg_arrays(twin_master, min_sigma_override=min_sigma_override)
    twin, fresh = _dual_align_twin_arrays(own['dt'], twin_native)

    mode = str(DUAL_BAND_ALIGN_MODE).strip().upper()
    k = _dual_eff_std_mult()
    is_ce = (own_type == "CE")
    twin_type = "PE" if is_ce else "CE"

    h_close, h_mu, h_sd, h_mid, h_std_rs, h_z = (
        own['close'], own['mu_log'], own['sd_eff'], own['mid'], own['std_rs'], own['z']
    )
    t_close, t_mu, t_sd, t_mid, t_z = (
        twin['close'], twin['mu_log'], twin['sd_eff'], twin['mid'], twin['z']
    )

    with np.errstate(all='ignore'):
        valid = fresh & np.isfinite(h_mu) & np.isfinite(h_sd) & np.isfinite(t_mu) & np.isfinite(t_sd)
        if mode == "OFFSET":
            upper = h_mid + k * h_std_rs
            lower = np.maximum(0.0, h_mid - k * h_std_rs)
            twin_aligned = h_mid + (t_close - t_mid)
            host_up, host_dn = (h_close > upper), (h_close < lower)
            twin_up, twin_dn = (twin_aligned > upper), (twin_aligned < lower)
        else:
            upper = np.exp(h_mu + k * h_sd)
            lower = np.exp(h_mu - k * h_sd)
            twin_aligned = np.exp(h_mu + t_z * h_sd)
            host_up, host_dn = (h_z > k), (h_z < -k)
            twin_up, twin_dn = (t_z > k), (t_z < -k)

    ce_up, ce_dn = (host_up, host_dn) if is_ce else (twin_up, twin_dn)
    pe_up, pe_dn = (twin_up, twin_dn) if is_ce else (host_up, host_dn)
    z_ce, z_pe = (h_z, t_z) if is_ce else (t_z, h_z)
    ce_close, pe_close = (h_close, t_close) if is_ce else (t_close, h_close)

    up_ok, dn_ok = valid & ce_up & pe_dn, valid & pe_up & ce_dn
    trap_up, trap_dn = valid & ce_up & pe_up, valid & ce_dn & pe_dn
    chop = valid & ~(ce_up | ce_dn | pe_up | pe_dn)

    code = np.zeros(n, dtype=np.int8)
    code[valid] = 6
    code[chop] = 5
    code[trap_dn] = 4
    code[trap_up] = 3
    code[dn_ok] = 2
    code[up_ok] = 1

    return {
        'required': True, 'available': True, 'reason': "",
        'own_type': own_type, 'twin_type': twin_type,
        'mode': "OFFSET" if mode == "OFFSET" else "ZSCORE", 'code': code,
        'ok_bull': up_ok if is_ce else dn_ok,
        'ok_bear': dn_ok if is_ce else up_ok,
        'host_close': h_close, 'twin_close': t_close, 'twin_aligned': twin_aligned,
        'ce_close': ce_close, 'pe_close': pe_close,
        'mid': h_mid, 'upper': upper, 'lower': lower,
        'z_ce': z_ce, 'z_pe': z_pe, 'z_host': h_z, 'z_twin': t_z,
    }

def _dual_gate_pass(gate, i, direction):
    """(passed, why). Rows without a required gate (equity / index) always pass; a required gate with no data fails closed."""
    if not gate or not gate.get('required'):
        return True, ""
    if not gate.get('available'):
        return False, str(gate.get('reason') or "twin CE/PE candles unavailable")
    ok = gate['ok_bull'] if direction == "BULL" else gate['ok_bear']
    if i < len(ok) and bool(ok[i]):
        max_z = float(DUAL_BAND_MAX_Z or 0.0)
        if max_z > 0 and direction == "BULL" and i < len(gate['z_host']) and float(gate['z_host'][i]) > max_z:
            return False, f"CLIMAX OVEREXTENDED (z {float(gate['z_host'][i]):+.1f} > +{max_z:g})"
        return True, ""
    code = int(gate['code'][i]) if i < len(gate['code']) else 0
    if code in (1, 2):
        return False, "OPPOSITE side confirmed (CE/PE direction conflict)"
    return False, _DUAL_CODE_TEXT.get(code, "not confirmed")

def _dual_gate_snapshot(gate, i, direction):
    """Levels that confirmed the anchor (host leg on its own BB, twin leg normalised onto host BB)."""
    if not gate or not gate.get('required') or not gate.get('available'):
        return None
    return {
        'own_type': gate['own_type'], 'twin_type': gate['twin_type'],
        'dir': direction, 'side': "UP" if int(gate['code'][i]) == 1 else "DOWN",
        'mode': gate.get('mode', "ZSCORE"),
        'host': float(gate['host_close'][i]), 'twin': float(gate['twin_close'][i]),
        'twin_aligned': float(gate['twin_aligned'][i]),
        'ce': float(gate['ce_close'][i]), 'pe': float(gate['pe_close'][i]),
        'mid': float(gate['mid'][i]), 'upper': float(gate['upper'][i]), 'lower': float(gate['lower'][i]),
        'z_host': float(gate['z_host'][i]), 'z_twin': float(gate['z_twin'][i]),
        'z_ce': float(gate['z_ce'][i]), 'z_pe': float(gate['z_pe'][i]),
    }

def _dual_reversal_reason(gate, j, is_bull):
    """Kill reason when the arena FULLY confirms the opposite side of a live option anchor; None otherwise."""
    if not (DUAL_BAND_REVERSAL_EXIT and gate and gate.get('required') and gate.get('available')):
        return None
    rev = gate['ok_bear'] if is_bull else gate['ok_bull']
    if j < len(rev) and bool(rev[j]):
        return f"Kinetic SL (Dual-Band Reversal: {'CE' if int(gate['code'][j]) == 1 else 'PE'} BUY confirmed)"
    return None

def _dual_block_reason(dual_blocks, dt_1m):
    last_i, last_why = dual_blocks[-1]
    try: last_t = pd.to_datetime(dt_1m[last_i]).strftime('%H:%M')
    except Exception: last_t = '-'
    return f"Dual-Band Gate: {last_why} ({len(dual_blocks)}x, last {last_t})"

def _closed_delta_bb_gate(reason):
    return {'required': True, 'available': False, 'reason': reason}

def _build_delta_bb_gate(abs_delta, sid, abs_delta_open=None):
    """Mandatory |Delta| Bollinger Band gate (fail-closed). Entry requires |Delta| > Upper BB(|Delta|).
    When ISOLATE_SESSION_BANDS=False, gap-stitches across sessions so overnight gap-ups/gap-downs are ignored."""
    d_arr = np.asarray(abs_delta, dtype=float)
    n = len(d_arr)
    if n == 0 or not np.isfinite(d_arr).any():
        return _closed_delta_bb_gate("delta series unavailable (fail-closed)")
    period = max(2, int(DELTA_BB_PERIOD or 20))
    std_mult = float(DELTA_BB_STD_MULT or 1.0)
    min_sig = max(float(DELTA_BB_MIN_SIGMA or 0.005), 1e-6)
    min_p = min(period, _scaled_warmup_bars())
    if ISOLATE_SESSION_BANDS:
        grp = pd.DataFrame({'d': d_arr, 'sid': sid}).groupby('sid')['d']
        d_mean = grp.transform(lambda x: x.rolling(period, min_periods=min_p).mean()).values
        d_std = grp.transform(lambda x: x.rolling(period, min_periods=min_p).std(ddof=0)).values
    else:
        sess_starts = np.zeros(n, dtype=bool)
        sess_starts[0] = True
        sess_starts[1:] = sid[1:] != sid[:-1]
        d_open = np.asarray(abs_delta_open, dtype=float) if abs_delta_open is not None else d_arr
        d_diff = np.diff(d_arr, prepend=d_arr[0])
        d_diff = np.where(sess_starts, d_arr - d_open, d_diff)
        stitched_d = np.cumsum(d_diff) + d_open[0]
        d_offset = d_arr - stitched_d
        s_d = pd.Series(stitched_d)
        d_mean = s_d.rolling(period, min_periods=min_p).mean().values + d_offset
        d_std = s_d.rolling(period, min_periods=min_p).std(ddof=0).values
    with np.errstate(all='ignore'):
        sd_eff = np.maximum(d_std, min_sig)
        upper = d_mean + std_mult * sd_eff
        lower = d_mean - std_mult * sd_eff
        z = (d_arr - d_mean) / sd_eff
        ok_bull = np.isfinite(upper) & np.isfinite(d_arr) & (d_arr > upper)
        ok_bear = np.isfinite(lower) & np.isfinite(d_arr) & (d_arr < lower)
    return {
        'required': True, 'available': True,
        'delta': d_arr, 'mean': d_mean, 'upper': upper, 'lower': lower, 'z': z,
        'ok_bull': ok_bull, 'ok_bear': ok_bear,
    }

def _delta_bb_pass(gate, i, direction):
    """(passed, why). Non-option rows (gate is None) always pass; option rows fail-closed when unavailable."""
    if not gate or not gate.get('required'):
        return True, ""
    if not gate.get('available'):
        return False, str(gate.get('reason') or "delta BB unavailable")
    ok = gate['ok_bull'] if direction == "BULL" else gate['ok_bear']
    if i < len(ok) and bool(ok[i]):
        return True, ""
    if i < len(gate['delta']) and np.isfinite(gate['upper'][i]):
        return False, f"|Δ| {float(gate['delta'][i]):.3f} <= Upper BB {float(gate['upper'][i]):.3f}"
    return False, "delta BB warming up"

def _delta_bb_snapshot(gate, i):
    if not gate or not gate.get('required') or not gate.get('available') or i >= len(gate['delta']):
        return None
    return {
        'delta': float(gate['delta'][i]),
        'mean': float(gate['mean'][i]),
        'upper': float(gate['upper'][i]),
        'lower': float(gate['lower'][i]),
        'z': float(gate['z'][i]),
    }

def _delta_block_reason(delta_blocks, dt_1m):
    if not delta_blocks: return ""
    last_i, last_why = delta_blocks[-1]
    try: last_t = pd.to_datetime(dt_1m[last_i]).strftime('%H:%M')
    except Exception: last_t = '-'
    return f"Delta-BB Gate: {last_why} ({len(delta_blocks)}x, last {last_t})"

def _inception_block_reason(inception_blocks, dt_1m):
    if not inception_blocks: return ""
    last_i, last_why = inception_blocks[-1]
    try: last_t = pd.to_datetime(dt_1m[last_i]).strftime('%H:%M')
    except Exception: last_t = '-'
    return f"Inception |Δ| Gate: {last_why} ({len(inception_blocks)}x, last {last_t})"

def _iv_block_reason(iv_blocks, dt_1m):
    if not iv_blocks: return ""
    last_i, last_why = iv_blocks[-1]
    try: last_t = pd.to_datetime(dt_1m[last_i]).strftime('%H:%M')
    except Exception: last_t = '-'
    return f"IV Gate: {last_why} ({len(iv_blocks)}x, last {last_t})"

def _dual_band_text(row):
    d = row.get('DualBand')
    if not d:
        return ""
    try:
        own_t = d['own_type']
        twin_t = d.get('twin_type') or ("PE" if own_t == "CE" else "CE")
        is_bull = (d.get('dir') == 'BULL')
        lbl = f"{own_t} {'BUY' if is_bull else 'SELL'}"
        tw_lbl = f"{twin_t}(al)" if d.get('mode') == "OFFSET" else f"{twin_t}(norm)"
        _is_del = (float(d['host']) <= 1.05 and float(d['upper']) <= 1.05)
        _dec = 3 if _is_del else 2
        _tag = "|Δ| " if _is_del else ""
        raw = f" [raw {d['twin']:.{_dec}f}]" if abs(d['twin'] - d['twin_aligned']) >= (0.001 if _is_del else 0.01) else ""
        if is_bull:
            body = f"{own_t} {_tag}{d['host']:.{_dec}f} > U {d['upper']:.{_dec}f} & {tw_lbl} {d['twin_aligned']:.{_dec}f}{raw} < L {d['lower']:.{_dec}f}"
        else:
            body = f"{tw_lbl} {d['twin_aligned']:.{_dec}f}{raw} > U {d['upper']:.{_dec}f} & {own_t} {_tag}{d['host']:.{_dec}f} < L {d['lower']:.{_dec}f}"
        return f"Dual-Band ✔ {lbl} @anchor: {body} (z {own_t} {d['z_host']:+.1f} / {twin_t} {d['z_twin']:+.1f})"
    except Exception:
        return ""

def _evaluate_iv_arrays(iv, sess_starts, sid):
    n = len(iv)
    # Reset IV smoothing at session boundaries so yesterday's IV never leaks into today's open
    iv_smooth = _ewm(iv, 2.0 / 4.0, sess_starts=sess_starts if ISOLATE_SESSION_BANDS else None) if n > 1 else iv
    delta_iv = np.diff(iv_smooth, prepend=iv_smooth[0])
    delta_iv = np.where(sess_starts, 0.0, delta_iv)

    alpha_vel = 1.0 / float(VEL_ATR_PERIOD)
    atr_iv = _ewm(np.abs(delta_iv), alpha_vel)

    min_p = min(IND_BB_PERIOD, max(1, int(ROLLING_MIN_PERIODS or 1)))
    if ISOLATE_SESSION_BANDS:
        grp = pd.DataFrame({'iv': iv, 'sid': sid}).groupby('sid')['iv']
        iv_mean = grp.transform(lambda x: x.rolling(IND_BB_PERIOD, min_periods=min_p).mean()).values
        iv_std = grp.transform(lambda x: x.rolling(IND_BB_PERIOD, min_periods=min_p).std(ddof=0)).values
    else:
        iv_diff = np.diff(iv, prepend=iv[0])
        iv_diff = np.where(sess_starts, 0.0, iv_diff)
        stitched_iv = np.cumsum(iv_diff) + iv[0]
        iv_offset = iv - stitched_iv
        s_iv = pd.Series(stitched_iv)
        iv_mean = s_iv.rolling(IND_BB_PERIOD, min_periods=min_p).mean().values + iv_offset
        iv_std = s_iv.rolling(IND_BB_PERIOD, min_periods=min_p).std(ddof=0).values

    w = max(1, int(SHOCK_WINDOW_BARS))
    win_viv_max = pd.Series(delta_iv).rolling(w, min_periods=min(w, max(1, int(ROLLING_MIN_PERIODS or 1)))).max().values
    win_viv_min = pd.Series(delta_iv).rolling(w, min_periods=min(w, max(1, int(ROLLING_MIN_PERIODS or 1)))).min().values

    return {
        'iv': iv,
        'iv_mean': iv_mean,
        'iv_std': iv_std,
        'vel_iv': delta_iv,
        'atr_iv': atr_iv,
        'win_viv_max': win_viv_max,
        'win_viv_min': win_viv_min,
        'iv_bb_upper': iv_mean + 2.0 * iv_std,
        'iv_bb_lower': iv_mean - 2.0 * iv_std,
        'iv_kc_upper': iv_mean + 1.5 * atr_iv,
        'iv_kc_lower': iv_mean - 1.5 * atr_iv
    }

def _evaluate_kinetic_arrays(open_, close, high, low, vol, sess_starts, sid, obv_close=None, obv_open=None):
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

    eff_vol = vol
    if INDEX_SYNTHETIC_VOL_FALLBACK and float(np.sum(vol)) <= 0.0:
        eff_vol = np.maximum(high - low, np.abs(delta)) + EPSILON

    # Use actual tradable price direction for OBV when provided (avoids crediting option volume during IV-crush price drops)
    if obv_close is not None:
        oc = np.asarray(obv_close, dtype=float)
        oo = np.asarray(obv_open if obv_open is not None else oc, dtype=float)
        obv_delta = np.diff(oc, prepend=oc[0])
        obv_delta = np.where(sess_starts, oc - oo, obv_delta)
    else:
        obv_delta = delta
    direction = np.where(obv_delta > 0, 1, np.where(obv_delta < 0, -1, 0))
    obv = np.cumsum(direction * eff_vol)

    if ISOLATE_SESSION_BANDS:
        df = pd.DataFrame({'rsi': rsi, 'hist': hist, 'plus_di': plus_di, 'minus_di': minus_di, 'adx': adx, 'obv': obv, 'sid': sid})
        grouped = df.groupby('sid')
        rsi_mean, rsi_std = grouped['rsi'].transform(lambda x: x.rolling(IND_BB_PERIOD, min_periods=min(IND_BB_PERIOD, max(1, int(ROLLING_MIN_PERIODS or 1)))).mean()).values, grouped['rsi'].transform(lambda x: x.rolling(IND_BB_PERIOD, min_periods=min(IND_BB_PERIOD, max(1, int(ROLLING_MIN_PERIODS or 1)))).std(ddof=0)).values
        h_mean, h_std = grouped['hist'].transform(lambda x: x.rolling(IND_BB_PERIOD, min_periods=min(IND_BB_PERIOD, max(1, int(ROLLING_MIN_PERIODS or 1)))).mean()).values, grouped['hist'].transform(lambda x: x.rolling(IND_BB_PERIOD, min_periods=min(IND_BB_PERIOD, max(1, int(ROLLING_MIN_PERIODS or 1)))).std(ddof=0)).values
        p_mean, p_std = grouped['plus_di'].transform(lambda x: x.rolling(IND_BB_PERIOD, min_periods=min(IND_BB_PERIOD, max(1, int(ROLLING_MIN_PERIODS or 1)))).mean()).values, grouped['plus_di'].transform(lambda x: x.rolling(IND_BB_PERIOD, min_periods=min(IND_BB_PERIOD, max(1, int(ROLLING_MIN_PERIODS or 1)))).std(ddof=0)).values
        m_mean, m_std = grouped['minus_di'].transform(lambda x: x.rolling(IND_BB_PERIOD, min_periods=min(IND_BB_PERIOD, max(1, int(ROLLING_MIN_PERIODS or 1)))).mean()).values, grouped['minus_di'].transform(lambda x: x.rolling(IND_BB_PERIOD, min_periods=min(IND_BB_PERIOD, max(1, int(ROLLING_MIN_PERIODS or 1)))).std(ddof=0)).values
        a_mean, a_std = grouped['adx'].transform(lambda x: x.rolling(IND_BB_PERIOD, min_periods=min(IND_BB_PERIOD, max(1, int(ROLLING_MIN_PERIODS or 1)))).mean()).values, grouped['adx'].transform(lambda x: x.rolling(IND_BB_PERIOD, min_periods=min(IND_BB_PERIOD, max(1, int(ROLLING_MIN_PERIODS or 1)))).std(ddof=0)).values
        o_mean, o_std = grouped['obv'].transform(lambda x: x.rolling(IND_BB_PERIOD, min_periods=min(IND_BB_PERIOD, max(1, int(ROLLING_MIN_PERIODS or 1)))).mean()).values, grouped['obv'].transform(lambda x: x.rolling(IND_BB_PERIOD, min_periods=min(IND_BB_PERIOD, max(1, int(ROLLING_MIN_PERIODS or 1)))).std(ddof=0)).values
    else:
        rsi_s, hist_s = pd.Series(rsi), pd.Series(hist)
        pdi_s, mdi_s = pd.Series(plus_di), pd.Series(minus_di)
        adx_s, obv_s = pd.Series(adx), pd.Series(obv)
        rsi_mean, rsi_std = rsi_s.rolling(IND_BB_PERIOD, min_periods=min(IND_BB_PERIOD, max(1, int(ROLLING_MIN_PERIODS or 1)))).mean().values, rsi_s.rolling(IND_BB_PERIOD, min_periods=min(IND_BB_PERIOD, max(1, int(ROLLING_MIN_PERIODS or 1)))).std(ddof=0).values
        h_mean, h_std = hist_s.rolling(IND_BB_PERIOD, min_periods=min(IND_BB_PERIOD, max(1, int(ROLLING_MIN_PERIODS or 1)))).mean().values, hist_s.rolling(IND_BB_PERIOD, min_periods=min(IND_BB_PERIOD, max(1, int(ROLLING_MIN_PERIODS or 1)))).std(ddof=0).values
        p_mean, p_std = pdi_s.rolling(IND_BB_PERIOD, min_periods=min(IND_BB_PERIOD, max(1, int(ROLLING_MIN_PERIODS or 1)))).mean().values, pdi_s.rolling(IND_BB_PERIOD, min_periods=min(IND_BB_PERIOD, max(1, int(ROLLING_MIN_PERIODS or 1)))).std(ddof=0).values
        m_mean, m_std = mdi_s.rolling(IND_BB_PERIOD, min_periods=min(IND_BB_PERIOD, max(1, int(ROLLING_MIN_PERIODS or 1)))).mean().values, mdi_s.rolling(IND_BB_PERIOD, min_periods=min(IND_BB_PERIOD, max(1, int(ROLLING_MIN_PERIODS or 1)))).std(ddof=0).values
        a_mean, a_std = adx_s.rolling(IND_BB_PERIOD, min_periods=min(IND_BB_PERIOD, max(1, int(ROLLING_MIN_PERIODS or 1)))).mean().values, adx_s.rolling(IND_BB_PERIOD, min_periods=min(IND_BB_PERIOD, max(1, int(ROLLING_MIN_PERIODS or 1)))).std(ddof=0).values
        o_mean, o_std = obv_s.rolling(IND_BB_PERIOD, min_periods=min(IND_BB_PERIOD, max(1, int(ROLLING_MIN_PERIODS or 1)))).mean().values, obv_s.rolling(IND_BB_PERIOD, min_periods=min(IND_BB_PERIOD, max(1, int(ROLLING_MIN_PERIODS or 1)))).std(ddof=0).values

    vel_hist = np.where(sess_starts, 0.0, np.diff(hist, prepend=hist[0]))
    vel_obv = np.where(sess_starts, 0.0, np.diff(obv, prepend=obv[0]))
    vel_pdi = np.where(sess_starts, 0.0, np.diff(plus_di, prepend=plus_di[0]))
    vel_mdi = np.where(sess_starts, 0.0, np.diff(minus_di, prepend=minus_di[0]))
    vel_adx = np.where(sess_starts, 0.0, np.diff(adx, prepend=adx[0]))

    alpha_vel = 1.0 / float(VEL_ATR_PERIOD)
    atr_hist, atr_obv = _ewm(np.abs(vel_hist), alpha_vel), _ewm(np.abs(vel_obv), alpha_vel)
    atr_pdi, atr_mdi, atr_adx = _ewm(np.abs(vel_pdi), alpha_vel), _ewm(np.abs(vel_mdi), alpha_vel), _ewm(np.abs(vel_adx), alpha_vel)

    w = max(1, int(SHOCK_WINDOW_BARS))
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
        'win_vh_max': pd.Series(vel_hist).rolling(w, min_periods=min(w, max(1, int(ROLLING_MIN_PERIODS or 1)))).max().values,
        'win_vh_min': pd.Series(vel_hist).rolling(w, min_periods=min(w, max(1, int(ROLLING_MIN_PERIODS or 1)))).min().values,
        'win_vo_max': pd.Series(vel_obv).rolling(w, min_periods=min(w, max(1, int(ROLLING_MIN_PERIODS or 1)))).max().values,
        'win_vo_min': pd.Series(vel_obv).rolling(w, min_periods=min(w, max(1, int(ROLLING_MIN_PERIODS or 1)))).min().values,
        'win_vp_max': pd.Series(vel_pdi).rolling(w, min_periods=min(w, max(1, int(ROLLING_MIN_PERIODS or 1)))).max().values,
        'win_vm_max': pd.Series(vel_mdi).rolling(w, min_periods=min(w, max(1, int(ROLLING_MIN_PERIODS or 1)))).max().values,
        'win_va_max': pd.Series(vel_adx).rolling(w, min_periods=min(w, max(1, int(ROLLING_MIN_PERIODS or 1)))).max().values
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

    b_shock_cnt = (
        int(w_vh_max > KINETIC_SHOCK_ATR_MULT * ah) + int(w_vo_max > KINETIC_SHOCK_ATR_MULT * ao) +
        int(w_vp_max > KINETIC_SHOCK_ATR_MULT * ap) + int(w_va_max > KINETIC_SHOCK_ATR_MULT * aa)
    )
    br_shock_cnt = (
        int(w_vh_min < -KINETIC_SHOCK_ATR_MULT * ah) + int(w_vo_min < -KINETIC_SHOCK_ATR_MULT * ao) +
        int(w_vm_max > KINETIC_SHOCK_ATR_MULT * am) + int(w_va_max > KINETIC_SHOCK_ATR_MULT * aa)
    )

    bull_shock = (b_shock_cnt >= SHOCK_QUORUM) and (vh > SHOCK_FLOW_MIN_THRESHOLD or vo > SHOCK_FLOW_MIN_THRESHOLD)
    bear_shock = (br_shock_cnt >= SHOCK_QUORUM) and (vh < -SHOCK_FLOW_MIN_THRESHOLD or vo < -SHOCK_FLOW_MIN_THRESHOLD)

    bull_score = (
        int(p_std > 0 and p_di > p_mean + OPPOSING_BB_STD_MULT * p_std) +
        int(vh > OPPOSING_VEL_ATR_MULT * ah) + int(vo > OPPOSING_VEL_ATR_MULT * ao) +
        int(h_std > 0 and hist > h_mean + OPPOSING_BB_STD_MULT * h_std)
    )
    bear_score = (
        int(m_std > 0 and m_di > m_mean + OPPOSING_BB_STD_MULT * m_std) +
        int(vh < -OPPOSING_VEL_ATR_MULT * ah) + int(vo < -OPPOSING_VEL_ATR_MULT * ao) +
        int(h_std > 0 and hist < h_mean - OPPOSING_BB_STD_MULT * h_std)
    )

    return (
        bull_score, bear_score,
        bull_score >= WARZONE_OPPOSING_SCORE, bear_score >= WARZONE_OPPOSING_SCORE,
        bull_struct, bear_struct, bull_shock, bear_shock
    )

def calculate_grids(kin_1m, idx, direction, iv_kin=None):
    hist, h_mean, h_std = kin_1m['hist'][idx], kin_1m['h_mean'][idx], kin_1m['h_std'][idx]
    obv, o_mean, o_std = kin_1m['obv'][idx], kin_1m['o_mean'][idx], kin_1m['o_std'][idx]
    p_di, p_mean, p_std = kin_1m['plus_di'][idx], kin_1m['p_mean'][idx], kin_1m['p_std'][idx]
    m_di, m_mean, m_std = kin_1m['minus_di'][idx], kin_1m['m_mean'][idx], kin_1m['m_std'][idx]
    adx, a_mean, a_std = kin_1m['adx'][idx], kin_1m['a_mean'][idx], kin_1m['a_std'][idx]

    ah, ao, ap, am, aa = kin_1m['atr_hist'][idx], kin_1m['atr_obv'][idx], kin_1m['atr_pdi'][idx], kin_1m['atr_mdi'][idx], kin_1m['atr_adx'][idx]
    vh = max(kin_1m['vel_hist'][idx], kin_1m['win_vh_max'][idx]) if direction == "BULL" else min(kin_1m['vel_hist'][idx], kin_1m['win_vh_min'][idx])
    vo = max(kin_1m['vel_obv'][idx], kin_1m['win_vo_max'][idx]) if direction == "BULL" else min(kin_1m['vel_obv'][idx], kin_1m['win_vo_min'][idx])
    vp = max(kin_1m['vel_pdi'][idx], kin_1m['win_vp_max'][idx])
    vm = max(kin_1m['vel_mdi'][idx], kin_1m['win_vm_max'][idx])
    va = max(kin_1m['vel_adx'][idx], kin_1m['win_va_max'][idx])

    has_iv = bool(iv_kin and ENABLE_OPTION_IV_ENGINE and idx < len(iv_kin.get('iv', [])))
    if has_iv:
        v_iv = max(iv_kin['vel_iv'][idx], iv_kin['win_viv_max'][idx]) if direction == "BULL" else min(iv_kin['vel_iv'][idx], iv_kin['win_viv_min'][idx])
        a_iv = max(iv_kin['atr_iv'][idx], 1e-6)
        iv_v = iv_kin['iv'][idx]
        iv_m = iv_kin['iv_mean'][idx]
        iv_s = iv_kin['iv_std'][idx]

    atr_grid = {m: 0 for m in GRID_MULTIPLIERS}
    bb_grid = {m: 0 for m in GRID_MULTIPLIERS}

    def check_bb(val, mean, std, mult, is_bull_ind):
        if std <= 0 or not np.isfinite(std): return False
        return (val > mean + (std * mult)) if is_bull_ind else (val < mean - (std * mult))

    for mult in GRID_MULTIPLIERS:
        a_count, b_count = 0, 0
        if direction == "BULL":
            if vh > mult * ah: a_count += 1
            if vo > mult * ao: a_count += 1
            if vp > mult * ap: a_count += 1
            if va > mult * aa: a_count += 1
            if has_iv and v_iv > mult * a_iv: a_count += 1
            if check_bb(hist, h_mean, h_std, mult, True): b_count += 1
            if check_bb(obv, o_mean, o_std, mult, True): b_count += 1
            if check_bb(p_di, p_mean, p_std, mult, True): b_count += 1
            if check_bb(adx, a_mean, a_std, mult, True): b_count += 1
            if has_iv and check_bb(iv_v, iv_m, iv_s, mult, True): b_count += 1
        else:
            if vh < -mult * ah: a_count += 1
            if vo < -mult * ao: a_count += 1
            if vm > mult * am: a_count += 1
            if va > mult * aa: a_count += 1
            if has_iv and v_iv < -mult * a_iv: a_count += 1
            if check_bb(hist, h_mean, h_std, mult, False): b_count += 1
            if check_bb(obv, o_mean, o_std, mult, False): b_count += 1
            if check_bb(m_di, m_mean, m_std, mult, True): b_count += 1
            if check_bb(adx, a_mean, a_std, mult, True): b_count += 1
            if has_iv and check_bb(iv_v, iv_m, iv_s, mult, False): b_count += 1
        atr_grid[mult], bb_grid[mult] = a_count, b_count
    return atr_grid, bb_grid

def evaluate_anchor_tripwire(
    close, dt_1m, kin_1m, base_atr, bb_upper, bb_lower, kc_upper, kc_lower,
    scan_start, is_late_session, end=None, allow_bull=True, allow_bear=True,
    iv_kin=None, is_option=False, dual_gate=None, delta_gate=None, day_ref=0.0,
    day_open_ref=0.0, ind_close=None, live_delta=None, delta_base_atr=None,
    inception_gate=None, is_expiry_day=False, swing_mode=False, target_dt=None
):
    n = len(close) if end is None else min(int(end), len(close))
    if scan_start >= n:
        return None, {'dir': 'NONE', 'anchor_time': '-', 'anchor_price': 0.0, 'killed_time': '-', 'killed_price': 0.0, 'reason': 'No data for scan window', 'kills': []}

    hard_mult = float(HARD_STOP_ATR_MULT or 0.0)
    soft_mult = float(SOFT_STOP_ATR_MULT)
    opp_thresh = int(SOFT_STOP_OPPOSING_SCORE)
    soft_disable_r = float(SOFT_STOP_DISABLE_R)

    _tf_scale = max(1, int(CANDLE_TF_MIN or 1))
    warmup = 0 if swing_mode else max(0, int(math.ceil(int(ANCHOR_OPEN_WARMUP_BARS or 0) / _tf_scale)))
    i = scan_start + warmup
    if i >= n:
        return None, {'dir': 'NONE', 'anchor_time': '-', 'anchor_price': 0.0, 'killed_time': '-', 'killed_price': 0.0, 'reason': f'Open warm-up ({warmup} bars)', 'kills': []}

    _live_d = live_delta if live_delta is not None else ind_close
    survived_anchor = None
    last_killed_info = {'dir': 'NONE', 'anchor_time': '-', 'anchor_price': 0.0, 'killed_time': '-', 'killed_price': 0.0, 'reason': 'Failed Kinetic Alignment'}
    kills, warzone_kills, consec_losses = [], 0, 0
    dual_blocks, delta_blocks, inception_blocks, iv_blocks = [], [], [], []
    last_kill_was_win = False

    while i < n:
        anchor = None
        while i < n:
            (bull_score, bear_score, r_bull, r_bear, bull_struct, bear_struct, bull_shock, bear_shock) = get_kinetics(kin_1m, i)
            _ic = close[i] if ind_close is None else ind_close[i]
            bb_kc_bull_fire = (bb_upper[i] > kc_upper[i]) or (_ic > bb_upper[i] and _ic > kc_upper[i])
            bb_kc_bear_fire = (bb_lower[i] < kc_lower[i]) or (_ic < bb_lower[i] and _ic < kc_lower[i])

            iv_ok_bull = True
            iv_ok_bear = True
            iv_why = ""
            if is_option and iv_kin and ENABLE_OPTION_IV_ENGINE and (not is_expiry_day) and i < len(iv_kin.get('iv', [])):
                vel_iv_i = iv_kin['vel_iv'][i]
                atr_iv_i = max(iv_kin['atr_iv'][i], 1e-6)
                iv_i = iv_kin['iv'][i]
                iv_mean_i = iv_kin['iv_mean'][i]
                iv_std_i = iv_kin['iv_std'][i]
                if IV_REQUIRE_NON_NEGATIVE_VEL and vel_iv_i < -0.2 * atr_iv_i:
                    iv_ok_bull = False
                    iv_why = "falling IV velocity"
                if np.isfinite(iv_std_i) and iv_std_i > 0 and iv_i < iv_mean_i - 1.5 * iv_std_i:
                    iv_ok_bull = False
                    iv_why = "IV below -1.5σ band"

            # Only relax Lock 3 / OPT_MAX_DAY_PCT on re-anchor when the prior attempt was a profitable exit (never after a losing stop)
            is_winning_reanchor = (len(kills) > 0 and last_kill_was_win)
            opt_px_ok = True
            if is_option:
                if close[i] < float(OPT_MIN_PRICE or 0.0):
                    opt_px_ok = False
                elif _live_d is not None:
                    ld_i = float(_live_d[i])
                    if swing_mode:
                        if ld_i < float(SWING_DELTA_MIN_ABS or 0.0) or ld_i > float(SWING_DELTA_MAX_ABS or 1.0):
                            opt_px_ok = False
                    elif DELTA_ENTRY_MAX_ABS and ld_i > float(DELTA_ENTRY_MAX_ABS):
                        opt_px_ok = False
                if opt_px_ok and (not swing_mode):
                    # Use intraday open reference for chase cap when available so far-month stale prev_close doesn't distort gating
                    chk_ref = day_open_ref if (day_open_ref and day_open_ref > 0) else day_ref
                    if chk_ref and chk_ref > 0:
                        dp_i = ((close[i] - chk_ref) / chk_ref) * 100.0
                        if OPT_MIN_DAY_PCT is not None and dp_i < float(OPT_MIN_DAY_PCT):
                            opt_px_ok = False
                        elif (not is_winning_reanchor) and OPT_MAX_DAY_PCT is not None and dp_i > float(OPT_MAX_DAY_PCT):
                            opt_px_ok = False

            # In Swing mode, multi-day |Delta| structure + BB/KC + Delta-BB + Dual-Band suffices without needing an intraday shock explosion
            bull_trigger_ok = bull_struct and (swing_mode or bull_shock or is_winning_reanchor) and bb_kc_bull_fire
            bear_trigger_ok = bear_struct and (swing_mode or bear_shock or is_winning_reanchor) and bb_kc_bear_fire

            if opt_px_ok and allow_bull and (swing_mode or not is_late_session[i]) and bull_trigger_ok:
                if not iv_ok_bull:
                    iv_blocks.append((i, iv_why or "IV decay"))
                else:
                    dual_pass, dual_why = _dual_gate_pass(dual_gate, i, "BULL")
                    if not dual_pass: dual_blocks.append((i, dual_why))
                    else:
                        delta_pass, delta_why = _delta_bb_pass(delta_gate, i, "BULL")
                        if not delta_pass: delta_blocks.append((i, delta_why))
                        elif inception_gate and inception_gate.get("enabled") and i < len(inception_gate["ok_bull"]) and not bool(inception_gate["ok_bull"][i]):
                            inception_blocks.append((i, f"|Δ| slope not rising (vs mean {float(inception_gate['st_above'][i]):+.4f}, slope {float(inception_gate['st_slope'][i]):+.4f}, life z {float(inception_gate['z'][i]):+.2f})"))
                        elif r_bear: warzone_kills += 1
                        else:
                            inc_snap = (
                                {"first_date": inception_gate["first_date"], "mean": float(inception_gate["mean"][i]), "z": float(inception_gate["z"][i]), "prior_days": int(inception_gate["prior_days"])}
                                if (inception_gate and i < len(inception_gate["mean"])) else None
                            )
                            anchor = {
                                "dir": "BULL", "idx": i, "time": dt_1m[i], "price": float(close[i]),
                                "delta": float(_live_d[i]) if _live_d is not None else float(close[i]),
                                "sig_delta": float(ind_close[i]) if ind_close is not None else float(close[i]),
                                "dual": _dual_gate_snapshot(dual_gate, i, "BULL"),
                                "delta_bb": _delta_bb_snapshot(delta_gate, i),
                                "inception_delta": inc_snap,
                            }
                            break
            elif opt_px_ok and allow_bear and (swing_mode or not is_late_session[i]) and bear_trigger_ok and iv_ok_bear:
                dual_pass, dual_why = _dual_gate_pass(dual_gate, i, "BEAR")
                if not dual_pass: dual_blocks.append((i, dual_why))
                else:
                    delta_pass, delta_why = _delta_bb_pass(delta_gate, i, "BEAR")
                    if not delta_pass: delta_blocks.append((i, delta_why))
                    elif r_bull: warzone_kills += 1
                    else:
                        anchor = {
                            "dir": "BEAR", "idx": i, "time": dt_1m[i], "price": float(close[i]),
                            "delta": float(_live_d[i]) if _live_d is not None else float(close[i]),
                            "sig_delta": float(ind_close[i]) if ind_close is not None else float(close[i]),
                            "dual": _dual_gate_snapshot(dual_gate, i, "BEAR"),
                            "delta_bb": _delta_bb_snapshot(delta_gate, i),
                        }
                        break
            i += 1

        if not anchor: break
        survived, is_bull = True, anchor['dir'] == "BULL"
        eff_atr = max(float(base_atr), float(anchor['price']) * float(OPT_MIN_ATR_PCT)) if is_option else float(base_atr)
        anchor['eff_atr'] = eff_atr
        d_eff_atr = float(delta_base_atr) if (is_option and delta_base_atr) else 0.0
        anchor['delta_eff_atr'] = d_eff_atr
        peak_val = float(anchor['price'])
        anchor_eval = float(anchor['price'])
        peak_live_delta = float(anchor.get('delta', 0.0))
        peak_sig_delta = float(anchor.get('sig_delta', peak_live_delta))
        j = anchor['idx'] + 1
        peak_iv = iv_kin['iv'][anchor['idx']] if (is_option and iv_kin and anchor['idx'] < len(iv_kin.get('iv', []))) else 0.0
        dual_rev_streak = 0

        while j < n:
            (bull_score, bear_score, _, _, _, _, _, _) = get_kinetics(kin_1m, j)
            ev_j = float(close[j])
            if is_bull:
                peak_val = max(peak_val, ev_j)
                adverse, opposing = peak_val - ev_j, bear_score
                soft_reason = f"Kinetic SL ({soft_mult:g} ATR Pullback)"
            else:
                peak_val = min(peak_val, ev_j)
                adverse, opposing = ev_j - peak_val, bull_score
                soft_reason = f"Kinetic SL ({soft_mult:g} ATR Rally)"

            max_r = (peak_val - anchor_eval) / eff_atr if is_bull else (anchor_eval - peak_val) / eff_atr
            curr_r = (ev_j - anchor_eval) / eff_atr if is_bull else (anchor_eval - ev_j) / eff_atr

            reason = None
            pl_trig, pl_floor = _active_profit_lock(max_r)
            eff_hard_mult = (
                float(HARD_STOP_RUNNER_ATR_MULT)
                if (hard_mult > 0 and float(HARD_STOP_RUNNER_TRIGGER_R or 0.0) > 0 and max_r >= float(HARD_STOP_RUNNER_TRIGGER_R))
                else hard_mult
            )

            # Swing Option-Delta structural invalidation (Delta trail + saturation profit-lock + floor)
            if swing_mode and is_option and is_bull and _live_d is not None and d_eff_atr > 0:
                ld_j = float(_live_d[j])
                sd_j = float(ind_close[j]) if ind_close is not None else ld_j
                peak_live_delta = max(peak_live_delta, ld_j)
                peak_sig_delta = max(peak_sig_delta, sd_j)
                d_trail_mult = (
                    float(SWING_DELTA_PROFIT_LOCK_TRAIL_MULT)
                    if peak_live_delta >= float(SWING_DELTA_PROFIT_LOCK_ABS or 0.82)
                    else float(SWING_DELTA_TRAIL_ATR_MULT or 2.0)
                )
                if ld_j < float(SWING_DELTA_EXIT_FLOOR or 0.40):
                    reason = f"Delta Swing Floor (|Δ| {ld_j:.3f} < {float(SWING_DELTA_EXIT_FLOOR):.2f})"
                elif (peak_sig_delta - sd_j) >= d_trail_mult * d_eff_atr:
                    lock_tag = " [Sat-Lock]" if peak_live_delta >= float(SWING_DELTA_PROFIT_LOCK_ABS or 0.82) else ""
                    reason = f"Delta Trail SL{lock_tag} (|Δ| {peak_live_delta:.3f}→{ld_j:.3f}, -{d_trail_mult:g}×Δ-ATR)"

            if reason is None:
                if eff_hard_mult > 0 and adverse >= eff_hard_mult * eff_atr:
                    reason = f"Hard stop ({eff_hard_mult:g} ATR trailing floor)"
                elif pl_trig is not None and curr_r <= pl_floor:
                    reason = f"Profit Lock (+{pl_trig:g}R Peak -> +{pl_floor:g}R Floor)"
                elif max_r < soft_disable_r and adverse >= soft_mult * eff_atr and opposing >= opp_thresh:
                    reason = soft_reason
                elif is_option and iv_kin and ENABLE_OPTION_IV_ENGINE and (not is_expiry_day) and j < len(iv_kin.get('iv', [])):
                    cur_iv = iv_kin['iv'][j]
                    peak_iv = max(peak_iv, cur_iv)
                    vel_iv_j = iv_kin['vel_iv'][j]
                    atr_iv_j = max(iv_kin['atr_iv'][j], 1e-6)
                    iv_dd = ((peak_iv - cur_iv) / peak_iv) if peak_iv > 0 else 0.0
                    if curr_r < 0.25 and iv_dd >= float(IV_CRUSH_DROP_PCT):
                        reason = f"Kinetic SL (IV Crush -{iv_dd*100:.1f}% from peak)"
                    elif curr_r < 0.0 and iv_dd >= float(IV_CRUSH_VEL_MIN_DROP_PCT) and vel_iv_j < -float(IV_CRUSH_VEL_MULT) * atr_iv_j:
                        reason = f"Kinetic SL (IV Crush Velocity Shock)"

            if reason is None:
                rev_msg = _dual_reversal_reason(dual_gate, j, is_bull)
                if rev_msg:
                    dual_rev_streak += 1
                    req_bars = max(1, int(DUAL_BAND_REVERSAL_CONFIRM_BARS or 1))
                    if dual_rev_streak >= req_bars or opposing >= WARZONE_OPPOSING_SCORE:
                        reason = rev_msg
                else:
                    dual_rev_streak = 0

            if reason:
                last_killed_info = {
                    'dir': anchor['dir'],
                    'anchor_date': pd.to_datetime(anchor['time']).strftime('%Y-%m-%d'),
                    'anchor_time': _format_anchor_label(anchor['time'], target_dt if swing_mode else None),
                    'anchor_price': float(anchor['price']),
                    'killed_time': _format_anchor_label(dt_1m[j], target_dt if swing_mode else None),
                    'killed_price': float(close[j]), 'reason': reason, 'peak_price': float(peak_val),
                    'anchor_delta': float(anchor.get('delta', 0.0)),
                    'killed_delta': float(_live_d[j]) if (is_option and _live_d is not None) else 0.0,
                    'peak_delta': float(peak_live_delta) if is_option else 0.0,
                    'delta_eff_atr': float(d_eff_atr),
                    'eff_atr': float(anchor['eff_atr']), 'dual': anchor.get('dual')
                }
                kills.append(last_killed_info)
                survived = False
                last_kill_was_win = (curr_r >= max(0.0, float(PRIOR_ATTEMPT_MIN_POSITIVE_R or 0.0)))
                if curr_r < 0:
                    consec_losses += 1
                    if (not swing_mode) and MAX_CONSECUTIVE_LOSSES_PER_DAY > 0 and consec_losses >= int(MAX_CONSECUTIVE_LOSSES_PER_DAY):
                        i = n
                    else:
                        _cd_bars = int(round(max(0, int(POST_LOSS_COOLDOWN_BARS or 0)) / _tf_scale))
                        i = j + 1 + _cd_bars
                else:
                    consec_losses = 0
                    i = j + 1
                break
            j += 1

        if survived:
            survived_anchor = anchor
            break

    if survived_anchor: return survived_anchor, {"status": "Survived", "kills": kills}
    info = dict(last_killed_info)
    if warzone_kills > 0 and info['dir'] == 'NONE': info['reason'] = f"Warzone Inversion Chop ({warzone_kills}x)"
    elif inception_blocks and info['dir'] == 'NONE': info['reason'] = _inception_block_reason(inception_blocks, dt_1m)
    elif delta_blocks and info['dir'] == 'NONE': info['reason'] = _delta_block_reason(delta_blocks, dt_1m)
    elif dual_blocks and info['dir'] == 'NONE': info['reason'] = _dual_block_reason(dual_blocks, dt_1m)
    elif iv_blocks and info['dir'] == 'NONE': info['reason'] = _iv_block_reason(iv_blocks, dt_1m)
    info['kills'] = kills
    return None, info

def prepare_series(master, target_dt, is_sector_index=False, is_option=False, opt_item=None, twin_master=None, dual_required=None, cutoff_dt=None, swing_mode=False):
    if master is None or master.empty: return None, "no_data"
    raw_day_mask = (master['Datetime'].dt.date == target_dt).values
    if not raw_day_mask.any(): return None, "no_data"
    raw_ts = int(np.argmax(raw_day_mask))

    # Compute gap-free Base ATR & Risk ATR on raw 1m history before resampling engine candles
    atr_ignore_gaps = bool(ATR_IGNORE_GAPS or swing_mode)
    base_atr_calc, atr_bars = compute_base_atr(master.iloc[:raw_ts], ignore_gaps=atr_ignore_gaps)
    atr_tf_used = f"{_tf_to_minutes(ATR_BASIS_TF)}min"
    if base_atr_calc is None or atr_bars < ATR_MIN_BARS:
        fb_min = _atr_fallback_minutes()
        if fb_min and fb_min < _tf_to_minutes(ATR_BASIS_TF):
            fb_atr, fb_bars = compute_base_atr(master.iloc[:raw_ts], tf=fb_min, ignore_gaps=atr_ignore_gaps)
            if fb_atr is not None and fb_bars >= ATR_MIN_BARS:
                base_atr_calc, atr_bars, atr_tf_used = fb_atr, fb_bars, f"{fb_min}min*"
    if base_atr_calc is None or atr_bars < ATR_MIN_BARS: return None, "short_history"
    raw_first_close = float(master['Close'].values[raw_ts])
    raw_day_open = float(master['Open'].values[raw_ts])
    raw_prev_close = float(master['Close'].values[raw_ts - 1]) if raw_ts > 0 else 0.0
    atr_pct_floor = float(INDEX_MIN_ATR_PCT) if is_sector_index else (float(OPT_MIN_ATR_PCT) if is_option else float(MIN_ATR_PCT))
    atr_pct_cap = float(INDEX_MAX_ATR_PCT or 0.0) if is_sector_index else (float(OPT_MAX_ATR_PCT or 0.0) if is_option else float(MAX_ATR_PCT or 0.0))
    floor_v = max(raw_first_close * atr_pct_floor, float(ABSOLUTE_MIN_ATR))
    cap_v = raw_first_close * atr_pct_cap if atr_pct_cap > 0 else float('inf')
    base_atr = max(min(float(base_atr_calc), cap_v), floor_v)
    risk_atr = base_atr
    _rt = str(RISK_ATR_TF if RISK_ATR_TF is not None else "").strip().lower()
    if _rt and _rt not in ("none", "off", "same", "0", "false"):
        r_calc, r_bars = compute_base_atr(master.iloc[:raw_ts], tf=_rt, ignore_gaps=atr_ignore_gaps)
        if r_calc is not None and r_bars >= ATR_MIN_BARS:
            risk_atr = max(min(float(r_calc), cap_v), floor_v)
            _base_min = max(1, _tf_to_minutes(ATR_BASIS_TF))
            _risk_min = max(1, _tf_to_minutes(_rt))
            if _risk_min > _base_min and base_atr > 0:
                _max_ratio = max(1.5, math.sqrt(_risk_min / float(_base_min)) * 1.25)
                risk_atr = min(risk_atr, base_atr * _max_ratio)

    # Resample engine candles (and twin leg) to CANDLE_TF_MIN with canonical bar open/close timestamps
    master = _resample_master(master, CANDLE_TF_MIN, cutoff_dt=cutoff_dt)
    if twin_master is not None and not twin_master.empty:
        twin_master = _resample_master(twin_master, CANDLE_TF_MIN, cutoff_dt=cutoff_dt)

    if master is None or master.empty: return None, "no_data"
    day_mask = (master['Datetime'].dt.date == target_dt).values
    if not day_mask.any():
        if not swing_mode:
            return None, "no_data"
        # Before first 75m bar closes (09:15-10:29), keep completed prior-session bars alive for multi-day swings
        ts = max(0, len(master) - 1)
    else:
        ts = int(np.argmax(day_mask))

    # Multi-day scan start for Swing mode (carryover across last SWING_LOOKBACK_SESSIONS trading sessions)
    unique_sessions = sorted(master['Session'].unique())
    if swing_mode and len(unique_sessions) > 1:
        lb_sess = max(1, int(SWING_LOOKBACK_SESSIONS or 10))
        start_sess = unique_sessions[max(0, len(unique_sessions) - lb_sess)]
        scan_start = int(np.argmax((master['Session'].values == start_sess)))
        # Ensure at least IND_BB_PERIOD bars of warm-up before first swing anchor
        scan_start = max(scan_start, min(ts, int(IND_BB_PERIOD)))
    else:
        scan_start = ts

    open_ = master['Open'].values.astype(float)
    close = master['Close'].values.astype(float)
    high = master['High'].values.astype(float)
    low = master['Low'].values.astype(float)
    vol = master['Volume'].values.astype(float)
    sid, n = master['SessId'].values, len(close)

    sess_starts = np.zeros(n, dtype=bool)
    sess_starts[0] = True
    sess_starts[1:] = sid[1:] != sid[:-1]
    late_cut = int(SWING_LAST_NEW_ANCHOR_MIN if swing_mode else LAST_NEW_ANCHOR_MIN)
    is_late_session = (master['Min'].values >= late_cut)

    iv_kin = None
    delta_base_atr = None
    inception_delta = None
    is_expiry_day = False
    delta_bb = _closed_delta_bb_gate("underlying spot 1m candles unavailable (fail-closed)") if is_option else None
    ind_open, ind_high, ind_low, ind_close = open_, high, low, close
    live_delta = None

    if is_option:
        delta_ready = False
        try:
            opt_specs = _parse_contract_specs(opt_item.get('symbol') if opt_item else None, opt_item)
            if opt_specs and opt_specs.get('strike') and opt_specs['strike'] > 0:
                u_idx = opt_specs.get('underlying', 'NIFTY')
                u_key = (opt_item or {}).get('underlying_key')
                spot_df = _get_cached_spot_1m(u_idx, target_dt, MIN1_HISTORY_DAYS, target_dt == now_ist().date(), spot_key=u_key)
                if spot_df is not None and not spot_df.empty:
                    exp_dt = opt_specs.get('expiry') or target_dt
                    is_expiry_day = (exp_dt == target_dt)
                    is_call_opt = (opt_specs.get('option_type') == 'CE')
                    stk_val = float(opt_specs['strike'])

                    iv_arr, d_o, d_h, d_l, d_c, sig_o, sig_h, sig_l, sig_c = _compute_bs_delta_ohlc(
                        master, spot_df, stk_val, exp_dt, is_call_opt, cutoff_dt=cutoff_dt, swing_mode=swing_mode
                    )
                    if ENABLE_OPTION_IV_ENGINE:
                        iv_kin = _evaluate_iv_arrays(iv_arr, sess_starts, sid)
                    # Gap-stitched |Delta| Bollinger Band (runs on causal Charm/Vanna-neutralized sig_c in swing mode)
                    delta_bb = _build_delta_bb_gate(sig_c, sid, abs_delta_open=sig_o)
                    if ENABLE_INCEPTION_DELTA_REGIME:
                        inception_delta = _build_inception_delta_regime(master, d_c, target_dt, opt_item, opt_specs, spot_df, sig_o=sig_o, sig_c=sig_c)

                    # Directional indicators & BB/KC run on Black-Scholes |Delta| OHLC (Charm/Vanna-neutralized in swing mode)
                    ind_open, ind_high, ind_low, ind_close = sig_o, sig_h, sig_l, sig_c
                    live_delta = d_c
                    delta_ohlc_df = pd.DataFrame({
                        'Datetime': master['Datetime'].values, 'SessId': sid, 'Min': master['Min'].values,
                        'Open': sig_o, 'High': sig_h, 'Low': sig_l, 'Close': sig_c,
                    })
                    d_atr_calc, d_atr_bars = compute_base_atr(delta_ohlc_df.iloc[:ts], ignore_gaps=atr_ignore_gaps)
                    if d_atr_calc is None or d_atr_bars < ATR_MIN_BARS:
                        fb_min = _atr_fallback_minutes()
                        if fb_min:
                            d_atr_calc, d_atr_bars = compute_base_atr(delta_ohlc_df.iloc[:ts], tf=fb_min, ignore_gaps=atr_ignore_gaps)
                    if d_atr_calc is None or d_atr_bars < 1:
                        d_tr = np.maximum(sig_h[:max(1, ts)] - sig_l[:max(1, ts)], np.abs(np.diff(sig_c[:max(1, ts)], prepend=sig_c[0])))
                        d_atr_calc = float(np.mean(d_tr[-ATR_BASIS_PERIOD:])) if len(d_tr) else float(DELTA_MIN_ATR)
                    delta_base_atr = max(min(float(d_atr_calc), float(DELTA_MAX_ATR)), float(DELTA_MIN_ATR))
                    delta_ready = True
        except Exception as e:
            ERRORS.add("prepare_series:iv_kin", e)
            delta_bb = _closed_delta_bb_gate(f"delta BB error ({type(e).__name__}) (fail-closed)")
        if not delta_ready:
            return None, "no_spot_for_delta"

    tr = np.maximum(ind_high - ind_low, np.maximum(np.abs(ind_high - np.roll(ind_close, 1)), np.abs(ind_low - np.roll(ind_close, 1))))
    tr = np.where(sess_starts, ind_high - ind_low, tr)

    if ISOLATE_SESSION_BANDS:
        df_p = pd.DataFrame({'close': ind_close, 'tr': tr, 'sid': sid}).groupby('sid')
        sma_bb = df_p['close'].transform(lambda x: x.rolling(PRICE_BB_PERIOD, min_periods=min(PRICE_BB_PERIOD, max(1, int(ROLLING_MIN_PERIODS or 1)))).mean()).values
        std_bb = df_p['close'].transform(lambda x: x.rolling(PRICE_BB_PERIOD, min_periods=min(PRICE_BB_PERIOD, max(1, int(ROLLING_MIN_PERIODS or 1)))).std(ddof=0)).values
        sma_kc = df_p['close'].transform(lambda x: x.rolling(PRICE_KC_PERIOD, min_periods=min(PRICE_KC_PERIOD, max(1, int(ROLLING_MIN_PERIODS or 1)))).mean()).values
        atr_kc = df_p['tr'].transform(lambda x: x.rolling(PRICE_KC_PERIOD, min_periods=min(PRICE_KC_PERIOD, max(1, int(ROLLING_MIN_PERIODS or 1)))).mean()).values
    else:
        delta_p = np.diff(ind_close, prepend=ind_close[0])
        delta_p = np.where(sess_starts, ind_close - ind_open, delta_p)
        stitched_close = np.cumsum(delta_p) + ind_open[0]
        stitched_offset = ind_close - stitched_close
        sma_bb = pd.Series(stitched_close).rolling(PRICE_BB_PERIOD, min_periods=min(PRICE_BB_PERIOD, max(1, int(ROLLING_MIN_PERIODS or 1)))).mean().values + stitched_offset
        std_bb = pd.Series(stitched_close).rolling(PRICE_BB_PERIOD, min_periods=min(PRICE_BB_PERIOD, max(1, int(ROLLING_MIN_PERIODS or 1)))).std(ddof=0).values
        sma_kc = pd.Series(stitched_close).rolling(PRICE_KC_PERIOD, min_periods=min(PRICE_KC_PERIOD, max(1, int(ROLLING_MIN_PERIODS or 1)))).mean().values + stitched_offset
        atr_kc = pd.Series(tr).rolling(PRICE_KC_PERIOD, min_periods=min(PRICE_KC_PERIOD, max(1, int(ROLLING_MIN_PERIODS or 1)))).mean().values

    # MANDATORY CE/PE Dual-Band gate on ACTUAL OPTION PREMIUMS (catches IV-spike and IV-crush traps cleanly)
    dual_band = None
    if dual_required is None:
        dual_required = _dual_gate_required(opt_item, None, is_sector_index)
    if dual_required:
        own_type = str((opt_item or {}).get('option_type') or '').upper()
        try:
            dual_band = build_dual_band_gate(master, twin_master, own_type, min_sigma_override=None)
        except Exception as e:
            ERRORS.add("prepare_series:dual_band", e)
            dual_band = _closed_dual_gate(own_type, f"gate error ({type(e).__name__}) (fail-closed)")

    return {
        'close': close, 'ind_close': ind_close, 'live_delta': live_delta,
        'dt': master['Datetime'].values, 'close_dt': master['CloseDt'].values,
        'sessions': master['Session'].values, 'unique_sessions': unique_sessions,
        'today_start': ts, 'scan_start': scan_start, 'target_dt': target_dt, 'swing_mode': bool(swing_mode),
        'is_late_session': is_late_session,
        'day_open': float(open_[ts]) if day_mask.any() else raw_day_open, 'prev_close': (float(close[ts - 1]) if ts > 0 else 0.0) if day_mask.any() else raw_prev_close,
        'base_atr': base_atr, 'risk_atr': risk_atr, 'delta_base_atr': delta_base_atr, 'atr_bars': int(atr_bars), 'atr_tf': atr_tf_used,
        'kin': _evaluate_kinetic_arrays(
            ind_open, ind_close, ind_high, ind_low, vol, sess_starts, sid,
            obv_close=(close if is_option else None), obv_open=(open_ if is_option else None)
        ),
        'iv_kin': iv_kin, 'dual_band': dual_band, 'delta_bb': delta_bb, 'inception_delta': inception_delta, 'is_expiry_day': is_expiry_day,
        'bb_upper': sma_bb + PRICE_BB_STD_MULT * std_bb, 'bb_lower': sma_bb - PRICE_BB_STD_MULT * std_bb,
        'kc_upper': sma_kc + PRICE_KC_ATR_MULT * atr_kc, 'kc_lower': sma_kc - PRICE_KC_ATR_MULT * atr_kc,
    }, None

def row_at(S, end, symbol, sector="", sector_weight=0.0, is_sector_index=False):
    ts = S['today_start']
    scan_start = S.get('scan_start', ts)
    swing_mode = bool(S.get('swing_mode', False))
    target_dt = S.get('target_dt')
    end = min(int(end), len(S['close']))
    if end <= ts: return None
    close, base_atr, kin = S['close'], S['base_atr'], S['kin']
    last = end - 1
    ltp = float(close[last])
    ref = S['prev_close'] if (DAY_PCT_BASIS == "PREV_CLOSE" and S['prev_close'] > 0) else S['day_open']
    day_open_v = float(S.get('day_open', 0.0) or 0.0)
    open_chg_pct = ((ltp - day_open_v) / day_open_v) * 100.0 if day_open_v > 0 else 0.0

    is_opt = _is_option_symbol(symbol)
    allow_bull_anchor = True
    allow_bear_anchor = not (OPTIONS_BUYING_ONLY and is_opt)
    _live_d = S.get('live_delta') if S.get('live_delta') is not None else S.get('ind_close')

    row = {
        'Symbol': symbol, 'LTP': ltp, 'DayChangePct': ((ltp - ref) / ref) * 100.0 if ref else 0.0,
        'OpenChangePct': open_chg_pct,
        'ActiveAnchor': None, 'AnchorDate': '', 'DaysHeld': 0, 'AnchorPrice': 0.0, 'AnchorDir': "NONE", 'MovePct': 0.0,
        'State': "NONE", 'Blocks': 0, 'KilledTime': '-', 'KilledPrice': 0.0, 'RejectReason': "",
        'Kills': [], 'AtrBars': S['atr_bars'], 'AtrTf': S.get('atr_tf', ''), 'BaseAtr': base_atr, 'RiskAtr': S.get('risk_atr', base_atr),
        'DeltaBaseAtr': float(S.get('delta_base_atr') or 0.0), 'AnchorDelta': 0.0,
        'CurrDelta': float(_live_d[last]) if (is_opt and _live_d is not None) else 0.0,
        'KilledDelta': 0.0, 'PeakDelta': 0.0, 'PriorStops': 0, 'BankedR': 0.0,
        'TargetPrice': 0.0, 'SLPrice': 0.0, 'Sector': sector or DEFAULT_UNMAPPED_SECTOR,
        'SectorWeight': float(sector_weight or 0.0), 'IsSectorIndex': bool(is_sector_index),
        'DualBand': None, 'DeltaBB': None, 'InceptionDelta': None,
    }

    iv_kin = S.get('iv_kin')
    row['IV_Kin'] = iv_kin
    anchor, info = evaluate_anchor_tripwire(
        close, S['dt'], kin, base_atr, S['bb_upper'], S['bb_lower'], S['kc_upper'], S['kc_lower'],
        scan_start, S['is_late_session'], end, allow_bull=allow_bull_anchor, allow_bear=allow_bear_anchor,
        iv_kin=iv_kin, is_option=is_opt, dual_gate=S.get('dual_band'), delta_gate=S.get('delta_bb'),
        day_ref=ref, day_open_ref=day_open_v, ind_close=S.get('ind_close'), live_delta=_live_d,
        delta_base_atr=S.get('delta_base_atr'), inception_gate=S.get('inception_delta'),
        is_expiry_day=bool(S.get('is_expiry_day', False)), swing_mode=swing_mode, target_dt=target_dt
    )
    row['Kills'] = info.get('kills', [])
    row['DualBand'] = anchor.get('dual') if anchor else None
    row['DeltaBB'] = anchor.get('delta_bb') if anchor else None
    row['InceptionDelta'] = anchor.get('inception_delta') if anchor else None
    eval_dir = anchor['dir'] if anchor else info.get('dir')
    if not eval_dir or eval_dir == "NONE": eval_dir = "BULL"

    row['ATR_Grid'], row['BB_Grid'] = calculate_grids(kin, last, eval_dir, iv_kin=iv_kin)
    _ic_last = (S['ind_close'][last] if S.get('ind_close') is not None else close[last])
    bb_expanding = (S['bb_upper'][last] > S['kc_upper'][last] or (_ic_last > S['bb_upper'][last] and _ic_last > S['kc_upper'][last])) if eval_dir == "BULL" else (S['bb_lower'][last] < S['kc_lower'][last] or (_ic_last < S['bb_lower'][last] and _ic_last < S['kc_lower'][last]))
    pure_state = "[ BB > KC ]" if bb_expanding else "[ BB < KC ]"

    if anchor:
        if anchor.get('eff_atr'):
            row['BaseAtr'] = float(anchor['eff_atr'])
            row['RiskAtr'] = max(float(row.get('RiskAtr', 0.0) or 0.0), float(anchor['eff_atr']))
        anc_ts = pd.to_datetime(anchor['time'])
        row['AnchorDate'] = anc_ts.strftime("%Y-%m-%d")
        row['ActiveAnchor'] = _format_anchor_label(anc_ts, target_dt if swing_mode else None)
        if swing_mode and S.get('unique_sessions'):
            u_sess = S['unique_sessions']
            anc_d = anc_ts.date()
            cur_d = pd.to_datetime(S['dt'][last]).date()
            row['DaysHeld'] = sum(1 for d_ in u_sess if anc_d <= d_ <= cur_d)
        row['AnchorPrice'] = float(anchor['price'])
        row['AnchorDelta'] = float(anchor.get('delta', 0.0)) if is_opt else 0.0
        row['AnchorDir'] = anchor['dir']
        row['PremiumDir'] = anchor['dir']
        row['MovePct'] = ((ltp - anchor['price']) / anchor['price']) * 100.0 if anchor['price'] else 0.0
        row['State'], row['Blocks'] = pure_state, last - anchor['idx']
        _seg = close[int(anchor['idx']) + 1:last + 1]
        _ap = float(anchor['price'])
        if len(_seg):
            row['PeakPrice'] = max(_ap, float(_seg.max())) if anchor['dir'] == 'BULL' else min(_ap, float(_seg.min()))
        else:
            row['PeakPrice'] = _ap
        row['PeakFor'] = _ap
        if is_opt and _live_d is not None:
            _dseg = _live_d[int(anchor['idx']) + 1:last + 1]
            _ad = float(anchor.get('delta', 0.0))
            if len(_dseg):
                row['PeakDelta'] = max(_ad, float(_dseg.max())) if anchor['dir'] == 'BULL' else min(_ad, float(_dseg.min()))
            else:
                row['PeakDelta'] = _ad
        row['TargetPrice'], row['SLPrice'] = _compute_target_sl(row)
    else:
        row['AnchorDir'] = info.get('dir', 'NONE')
        row['PremiumDir'] = row['AnchorDir']
        row['AnchorDate'] = info.get('anchor_date', '')
        row['ActiveAnchor'] = info.get('anchor_time', '-')
        row['AnchorPrice'] = float(info.get('anchor_price', 0.0))
        row['AnchorDelta'] = float(info.get('anchor_delta', 0.0)) if is_opt else 0.0
        row['KilledTime'] = info.get('killed_time', '-')
        row['KilledPrice'] = float(info.get('killed_price', 0.0))
        row['KilledDelta'] = float(info.get('killed_delta', 0.0)) if is_opt else 0.0
        row['RejectReason'] = info.get('reason', 'Failed Kinetic Alignment')
        if row['AnchorPrice'] > 0 and row['KilledPrice'] > 0:
            row['MovePct'] = ((row['KilledPrice'] - row['AnchorPrice']) / row['AnchorPrice']) * 100.0
        row['State'] = "[STOPPED]"
        row['TargetPrice'], row['SLPrice'] = _compute_target_sl(row)

    classify_setup_popup_type(row)
    row['RR_Val'], row['RR_Str'] = compute_risk_reward_ratio(row)
    return row

# ==============================================================================
# 3. WORKERS, SEGREGATION & MULTI-EPISODE MERGE LOGIC
# ==============================================================================
def _history_worker_daily(args):
    item, start, end, progress = args
    try:
        if item.get("pinned") or item.get("is_sector_index"): return item
        frames = []
        for c_start, c_end in _date_chunks(start, end + timedelta(days=1), span=CHUNK_SPAN_DAILY_DAYS):
            status, df = _candles(_url_range_daily(item['key'], c_start, c_end))
            if status == 200 and df is not None: frames.append(df)
        if not frames: return None
        master = prepare_master(frames)
        master = master[master['Datetime'].dt.date < end]
        if master.empty: return None
        close, vol = master['Close'].iloc[-1], master['Volume'].iloc[-1]
        return item if (MIN_PRICE <= close <= MAX_PRICE and vol >= MIN_DAILY_VOLUME) else None
    except Exception as e:
        ERRORS.add(f"prefilter:{item.get('symbol')}", e)
        return None
    finally:
        progress.tick()

def _load_master_1m(key, target_dt, final_cutoff, history_days, use_intraday):
    """Cached 1-minute candles of ONE instrument up to (excluding) the final cutoff; avoids duplicate fetches for twin legs."""
    cache_key = (key, target_dt, final_cutoff, int(history_days), bool(use_intraday))
    with _MASTER_1M_LOCK:
        if cache_key in _MASTER_1M_CACHE:
            return _MASTER_1M_CACHE[cache_key]
    frames = []
    for c_start, c_end in _date_chunks(target_dt - timedelta(days=history_days), target_dt + timedelta(days=1), span=CHUNK_SPAN_1M_DAYS):
        status, df = _candles(_url_range_1m(key, c_start, c_end))
        if status == 200 and df is not None: frames.append(df)
    if use_intraday:
        today_df = fetch_today(key)
        if today_df is not None: frames.append(today_df)
    res = None
    if frames:
        master = prepare_master(frames)
        res = master[master['Datetime'] < final_cutoff].reset_index(drop=True)
    with _MASTER_1M_LOCK:
        _MASTER_1M_CACHE[cache_key] = res
    return res

def _evaluate_symbol_checkpoints(item, master, twin_master, checkpoints, extra_lookback_cps, mode):
    sym = item['symbol']
    is_sec_idx = bool(item.get('is_sector_index', False))
    sec_name = item.get('sector', DEFAULT_UNMAPPED_SECTOR)
    sec_wt = float(item.get('sector_weight', 0.0))
    swing_mode = _is_swing_mode(mode)
    try:
        final_cutoff = checkpoints[-1]
        target_dt = final_cutoff.date()
        if master is None: return [(cp, "no_data", sym) for cp in checkpoints], []
        if _is_option_mode(mode) and not is_sec_idx:
            day = master[master['Datetime'].dt.date == target_dt]
            if day.empty or day['Close'].iloc[-1] < OPT_MIN_PRICE:
                return [(cp, "illiquid", sym) for cp in checkpoints], []
            # Scale intraday volume threshold proportionally to elapsed session time so morning runs don't drop liquid options
            cutoff_min = final_cutoff.hour * 60 + final_cutoff.minute
            elapsed_frac = max(0.05, min(1.0, (cutoff_min - SESSION_OPEN_MIN) / float(SESSION_CLOSE_MIN - SESSION_OPEN_MIN)))
            eff_min_vol = float(OPT_MIN_VOLUME) * elapsed_frac
            today_vol = float(day['Volume'].sum())
            vol_ok = (today_vol >= eff_min_vol)
            if (not vol_ok) and swing_mode:
                # In Swing mode, also accept contracts whose prior completed session met OPT_MIN_VOLUME
                prior_days = master[master['Datetime'].dt.date < target_dt]
                if not prior_days.empty:
                    last_prev_d = prior_days['Datetime'].dt.date.iloc[-1]
                    prev_vol = float(prior_days[prior_days['Datetime'].dt.date == last_prev_d]['Volume'].sum())
                    vol_ok = (prev_vol >= float(OPT_MIN_VOLUME))
            if not vol_ok:
                return [(cp, "illiquid", sym) for cp in checkpoints], []

        S, why = prepare_series(
            master, target_dt, is_sector_index=is_sec_idx,
            is_option=(_is_option_mode(mode) or _is_option_symbol(sym)) and (not is_sec_idx),
            opt_item=item, twin_master=twin_master,
            dual_required=_dual_gate_required(item, mode, is_sec_idx),
            cutoff_dt=final_cutoff, swing_mode=swing_mode
        )
        if S is None: return [(cp, why, sym) for cp in checkpoints], []

        close_dt_arr = S.get('close_dt') if S.get('close_dt') is not None else S['dt']
        ends = np.searchsorted(close_dt_arr, np.array(checkpoints, dtype='datetime64[ns]'), side='right')
        out, by_cp_row = [], {}
        for cp, end in zip(checkpoints, ends):
            row = row_at(S, int(end), sym, sector=sec_name, sector_weight=sec_wt, is_sector_index=is_sec_idx)
            if row is None:
                out.append((cp, "no_data", sym))
            else:
                row['CheckpointTime'] = cp
                out.append((cp, "ok", row))
                by_cp_row[cp] = row

        extra_out = []
        for l_cp in (extra_lookback_cps or []):
            if l_cp in by_cp_row:
                extra_out.append((l_cp, "ok", by_cp_row[l_cp]))
                continue
            l_end = int(np.searchsorted(close_dt_arr, np.datetime64(l_cp), side='right'))
            l_row = row_at(S, l_end, sym, sector=sec_name, sector_weight=sec_wt, is_sector_index=is_sec_idx)
            if l_row is not None:
                l_row['CheckpointTime'] = l_cp
                extra_out.append((l_cp, "ok", l_row))

        return out, extra_out
    except Exception as e:
        ERRORS.add(f"1m:{item.get('symbol')}", e)
        return [(cp, "error", item.get('symbol')) for cp in checkpoints], []

def _build_work_units(candidates):
    """Group candidates into work units: the CE and PE of one (underlying, expiry, strike) travel together."""
    units, by_pair = [], {}
    for it in candidates:
        if DUAL_BAND_GATE_MANDATORY and _is_true_option_item(it):
            pid = (it.get('underlying'), it.get('expiry'), float(it.get('strike') or 0.0))
            if pid in by_pair:
                by_pair[pid].append(it)
                continue
            by_pair[pid] = [it]
            units.append(by_pair[pid])
        else:
            units.append([it])
    return units

def process_work_unit(args):
    """Evaluate every item of one work unit. Returns one (outcomes, extra_outcomes) tuple per item."""
    items, checkpoints, extra_lookback_cps, use_intraday, history_days, mode, progress = args
    final_cutoff = checkpoints[-1]
    target_dt = final_cutoff.date()

    def _master_for(key):
        if not key: return None
        return _load_master_1m(key, target_dt, final_cutoff, history_days, use_intraday)

    results = []
    for item in items:
        sym = item.get('symbol')
        try:
            master = _master_for(item['key'])
            twin_master = None
            if DUAL_BAND_GATE_MANDATORY and _is_true_option_item(item):
                try: twin_master = _master_for(item.get('twin_key'))
                except Exception as e: ERRORS.add(f"1m-twin:{sym}", e)
            results.append(_evaluate_symbol_checkpoints(item, master, twin_master, checkpoints, extra_lookback_cps, mode))
        except Exception as e:
            ERRORS.add(f"1m:{sym}", e)
            results.append(([(cp, "error", sym) for cp in checkpoints], []))
        finally:
            progress.tick()
    return results

def process_stock_checkpoints(args):
    item, checkpoints, extra_lookback_cps, use_intraday, history_days, mode, progress = args
    return process_work_unit(([item], checkpoints, extra_lookback_cps, use_intraday, history_days, mode, progress))[0]

def _clip(s, w, allow_clip=True):
    if not allow_clip: return str(s)
    return s if len(s) <= w else s[:w - 1] + "~"

def _print_error_summary():
    if STATS.auth_failed:
        print(f"{COLOR_RED_FG}✖ Upstox API authentication failed (HTTP 401) — refresh UPSTOX_ACCESS_TOKEN.{COLOR_RESET}")
    if STATS.failed:
        print(f"{COLOR_YELLOW}⚠ {STATS.failed} API requests failed.{COLOR_RESET}")
        for s in STATS.samples[:3]:
            print(f"   {COLOR_DIM}• {s}{COLOR_RESET}")
    if ERRORS.items:
        print(f"{COLOR_YELLOW}⚠ {len(ERRORS.items)} instruments raised exceptions.{COLOR_RESET}")
        for where, msg, _ in ERRORS.items[:3]:
            print(f"   {COLOR_DIM}• {where}: {msg}{COLOR_RESET}")

def _signal_id(row):
    return (row['AnchorDir'], row.get('AnchorDate', ''), row['ActiveAnchor'], round(float(row['AnchorPrice']), 6))

def _find_kill(kills, d, t):
    return next((k for k in reversed(kills or []) if k.get('dir') == d and k.get('anchor_time') == t), None)

def _exit_price(row):
    return float(row['KilledPrice']) if row.get('State') == "[STOPPED]" and row.get('KilledPrice', 0) > 0 else float(row.get('LTP', 0.0) or 0.0)

def _kill_r_multiple(kill_dict, base_atr):
    ap, kp = float(kill_dict.get('anchor_price', 0.0) or 0.0), float(kill_dict.get('killed_price', 0.0) or 0.0)
    atr = float(kill_dict.get('eff_atr') or base_atr or 0.0)
    if ap <= 0 or kp <= 0 or atr <= 0: return 0.0
    return (kp - ap) / atr if kill_dict.get('dir', 'BULL') == 'BULL' else (ap - kp) / atr

def _r_multiple(row, include_banked=False):
    ap, base_atr, live_r = row.get('AnchorPrice', 0.0), row.get('BaseAtr', 0.0), 0.0
    if ap > 0 and base_atr > 0:
        price, direction = _exit_price(row), row.get('PremiumDir', row.get('AnchorDir'))
        live_r = (price - ap) / base_atr if direction == 'BULL' else (ap - price) / base_atr
    return live_r + float(row.get('BankedR', 0.0) or 0.0) if include_banked else live_r

def _rank_r_score(row):
    return max(_r_multiple(row, False), _r_multiple(row, True))

def _merge_signal(sig_rows, last_row, direction):
    first, latest = sig_rows[0], sig_rows[-1]
    row = dict(last_row)
    row['ActiveAnchor'], row['AnchorPrice'], row['AnchorDir'] = first['ActiveAnchor'], first['AnchorPrice'], direction
    _fcp = first.get('CheckpointTime')
    row['FirstSeenIter'] = _fcp.strftime('%H:%M') if hasattr(_fcp, 'strftime') else str(_fcp or '')
    row['AnchorDate'] = first.get('AnchorDate', '')
    row['DaysHeld'] = int(last_row.get('DaysHeld', first.get('DaysHeld', 0)) or 0)
    row['AnchorDelta'] = float(first.get('AnchorDelta', 0.0) or 0.0)
    row['PremiumDir'] = direction
    row['DualBand'] = first.get('DualBand')
    row['DeltaBB'] = first.get('DeltaBB')
    row['InceptionDelta'] = first.get('InceptionDelta')
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
            row['KilledDelta'] = float(k.get('killed_delta', 0.0) or 0.0)
            row['PeakPrice'] = float(k.get('peak_price', first['AnchorPrice']) or first['AnchorPrice'])
            row['PeakFor'] = float(first['AnchorPrice'])
            if k.get('eff_atr'):
                row['BaseAtr'] = float(k['eff_atr'])
        else:
            row['KilledTime'], row['KilledPrice'], row['RejectReason'] = latest['CheckpointTime'].strftime('%H:%M'), float(latest['LTP']), "Superseded by newer anchor"
            row['PeakPrice'] = float(latest.get('PeakPrice', first['AnchorPrice']) or first['AnchorPrice'])
            row['PeakFor'] = float(first['AnchorPrice'])
    ap = row['AnchorPrice'] or last_row['LTP']
    row['MovePct'] = ((_exit_price(row) - ap) / ap) * 100.0 if ap else 0.0
    row['TargetPrice'], row['SLPrice'] = _compute_target_sl(row)
    classify_setup_popup_type(row)
    row['RR_Val'], row['RR_Str'] = compute_risk_reward_ratio(row)
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
            if merged_ep['State'] != "[STOPPED]": active_merged[direction] = merged_ep
            else: grave_episodes.append(merged_ep)
        for k in [k for k in kills if k.get('dir') == direction and (direction, k.get('anchor_time')) not in seen_anchor_times]:
            seen_anchor_times.add((direction, k.get('anchor_time')))
            g_row = dict(last_row)
            g_row['AnchorDir'], g_row['ActiveAnchor'], g_row['AnchorPrice'] = direction, k['anchor_time'], float(k['anchor_price'])
            g_row['AnchorDate'] = k.get('anchor_date', '')
            g_row['AnchorDelta'] = float(k.get('anchor_delta', 0.0) or 0.0)
            g_row['PremiumDir'] = direction
            g_row['DualBand'] = k.get('dual')
            g_row['KilledTime'], g_row['KilledPrice'], g_row['RejectReason'], g_row['State'] = k['killed_time'], float(k['killed_price']), k['reason'], "[STOPPED]"
            g_row['KilledDelta'] = float(k.get('killed_delta', 0.0) or 0.0)
            g_row['PeakPrice'] = float(k.get('peak_price', k['anchor_price']) or k['anchor_price'])
            g_row['PeakFor'] = float(k['anchor_price'])
            if k.get('eff_atr'):
                g_row['BaseAtr'] = float(k['eff_atr'])
            g_row['MovePct'] = ((g_row['KilledPrice'] - g_row['AnchorPrice']) / g_row['AnchorPrice']) * 100.0 if g_row['AnchorPrice'] > 0 else 0.0
            g_row['TargetPrice'], g_row['SLPrice'] = _compute_target_sl(g_row)
            g_row['RR_Val'], g_row['RR_Str'] = compute_risk_reward_ratio(g_row)
            grave_episodes.append(g_row)
    if not active_merged and not grave_episodes:
        grave = dict(last_row)
        grave['CheckpointTime'] = rows[0]['CheckpointTime']
        grave['TargetPrice'], grave['SLPrice'] = _compute_target_sl(grave)
        grave['RR_Val'], grave['RR_Str'] = compute_risk_reward_ratio(grave)
        grave_episodes.append(grave)
    return active_merged.get("BULL"), active_merged.get("BEAR"), grave_episodes

def _script_group_key(row):
    sym = str(row.get("Symbol") or "").upper().strip()
    sec = str(row.get("Sector") or "").upper().strip()
    if sec in INDEX_CONFIG: return sec
    first_tok = sym.split()[0] if sym else ""
    if first_tok in ALIAS_TO_INDEX: return ALIAS_TO_INDEX[first_tok]
    for idx_name in INDEX_CONFIG:
        if sym.startswith(idx_name): return idx_name
    return first_tok or sym

def _dedupe_best_index_strike(rows_list):
    """Optionally collapse correlated strikes of the same index/direction to the highest-ranked row."""
    if not INDEX_OPTIONS_BEST_STRIKE_ONLY or not rows_list:
        return rows_list
    out, seen_keys = [], set()
    for r in rows_list:
        gk = (_script_group_key(r), str(r.get("AnchorDir", "")))
        if gk not in seen_keys:
            seen_keys.add(gk)
            out.append(r)
    return out

def sort_rows_by_script_count(rows_list):
    if not rows_list: return rows_list
    first_grid_key = GRID_MULTIPLIERS[0] if GRID_MULTIPLIERS else 1
    def state_rank(st): return 0 if "[ BB > KC ]" in str(st) else 1 if "[ BB < KC ]" in str(st) else 2

    counts = {}
    for r in rows_list:
        gk = _script_group_key(r)
        counts[gk] = counts.get(gk, 0) + 1
        if 'RR_Val' not in r or 'RR_Str' not in r:
            rr_val, rr_str = compute_risk_reward_ratio(r)
            r['RR_Val'] = rr_val
            r['RR_Str'] = rr_str

    is_asc = (str(RR_SORT_ORDER).upper() == "ASC")
    tiers = ATTEMPT_SORT_ENABLED
    t_sign = -1 if ATTEMPT_SORT_TIME_DESC else 1
    def _time_sort_val(r):
        if _is_swing_mode():
            ad = str(r.get('AnchorDate') or '0000-00-00')
            try:
                ord_d = datetime.strptime(ad, "%Y-%m-%d").toordinal()
            except Exception:
                ord_d = 0
            return ord_d * 1440 + _hhmm_to_min(r.get('ActiveAnchor'))
        return _hhmm_to_min(r.get('ActiveAnchor'))

    rows_list.sort(key=lambda r: (
        1 if str(r.get('RR_Str', '')) == 'SL hit' else 0,
        _attempt_tier(r) if tiers else 0,
        t_sign * _time_sort_val(r) if tiers else 0,
        r.get('RR_Val', 999.0) if is_asc else -r.get('RR_Val', 0.0),
        -counts.get(_script_group_key(r), 0),
        _script_group_key(r),
        -_rank_r_score(r),
        -_r_multiple(r),
        state_rank(r.get("State", "")),
        -r.get("ATR_Grid", {}).get(first_grid_key, 0),
        -r.get("BB_Grid", {}).get(first_grid_key, 0),
        str(r.get("Symbol", "")),
    ))
    return rows_list

def segregate_and_rank_buckets(by_symbol_map, filter_timestamp_gap=None, filter_popup_strict=None):
    swing = _is_swing_mode()
    if filter_timestamp_gap is None:
        filter_timestamp_gap = False if swing else ENABLE_TIMESTAMP_GAP_CHECK
    if filter_popup_strict is None:
        filter_popup_strict = False if swing else STRICT_FIRST_TIME_POPUP_FILTER

    bulls_live, bears_live, graveyard_candidates = [], [], []

    for symbol, rows in by_symbol_map.items():
        b, r_, grave_list = merge_symbol_rows(rows)
        is_pe = _is_pe_symbol(symbol)
        is_opt = _is_option_symbol(symbol)

        def _apply_dir(item, raw_dir):
            item['PremiumDir'] = raw_dir
            item['AnchorDir'] = "BEAR" if (raw_dir == "BULL" and is_pe) else ("BULL" if (raw_dir == "BEAR" and is_pe) else raw_dir)
            item['TargetPrice'], item['SLPrice'] = _compute_target_sl(item)
            item['RR_Val'], item['RR_Str'] = compute_risk_reward_ratio(item)

        for item, raw_dir in ((b, "BULL"), (r_, "BEAR")):
            if not item: continue
            if OPTIONS_BUYING_ONLY and is_opt and raw_dir != "BULL":
                item['PremiumDir'] = raw_dir
                item['State'] = "[STOPPED]"
                item['RejectReason'] = "Filtered (Option Buying Only — Falling Option Ignored)"
                graveyard_candidates.append(item)
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

    drift_floor = float(ADVERSE_DRIFT_R_FLOOR) if ADVERSE_DRIFT_R_FLOOR is not None else -abs(float(HARD_STOP_ATR_MULT or 1.2) * float(ADVERSE_DRIFT_STOP_RATIO))

    surviving_bulls, surviving_bears = [], []
    for r in bulls_live:
        if not (_r_multiple(r) > drift_floor or _r_multiple(r, include_banked=True) > BANKED_R_PROTECTION_FLOOR):
            r['RejectReason'] = f"Adverse Drift (Losing R-Multiple < {drift_floor:g}R)"
            graveyard_candidates.append(r)
        elif filter_timestamp_gap and MAX_TIMESTAMP_GAP_MINUTES > 0 and _get_timestamp_gap_min(r) > MAX_TIMESTAMP_GAP_MINUTES and not (PURE_REANCHOR_ANY_TIME and str(r.get('PopupType', '')) == 'RE-ANCHOR+'):
            r['RejectReason'] = f"Timestamp Gap Stale ({_get_timestamp_gap_min(r)}m > {MAX_TIMESTAMP_GAP_MINUTES}m)"
            graveyard_candidates.append(r)
        elif filter_popup_strict and _r_multiple(r) < float(POPUP_MIN_R_MULTIPLE) - 1e-6:
            r['RejectReason'] = f"Underwater Setup (Live R-Multiple {_r_multiple(r):+.2f}R < {POPUP_MIN_R_MULTIPLE:g}R)"
            graveyard_candidates.append(r)
        elif filter_popup_strict and POPUP_REQUIRE_NON_ZERO_GRID and (
            sum(int(v or 0) for v in (r.get('ATR_Grid') or {}).values()) <= 0 and sum(int(v or 0) for v in (r.get('BB_Grid') or {}).values()) <= 0
        ):
            r['RejectReason'] = "Kinetic Decay (Both ATR & BB Grids at 0 0 0 0)"
            graveyard_candidates.append(r)
        else:
            surviving_bulls.append(r)

    for r in bears_live:
        if not (_r_multiple(r) > drift_floor or _r_multiple(r, include_banked=True) > BANKED_R_PROTECTION_FLOOR):
            r['RejectReason'] = f"Adverse Drift (Losing R-Multiple < {drift_floor:g}R)"
            graveyard_candidates.append(r)
        elif filter_timestamp_gap and MAX_TIMESTAMP_GAP_MINUTES > 0 and _get_timestamp_gap_min(r) > MAX_TIMESTAMP_GAP_MINUTES and not (PURE_REANCHOR_ANY_TIME and str(r.get('PopupType', '')) == 'RE-ANCHOR+'):
            r['RejectReason'] = f"Timestamp Gap Stale ({_get_timestamp_gap_min(r)}m > {MAX_TIMESTAMP_GAP_MINUTES}m)"
            graveyard_candidates.append(r)
        elif filter_popup_strict and _r_multiple(r) < float(POPUP_MIN_R_MULTIPLE) - 1e-6:
            r['RejectReason'] = f"Underwater Setup (Live R-Multiple {_r_multiple(r):+.2f}R < {POPUP_MIN_R_MULTIPLE:g}R)"
            graveyard_candidates.append(r)
        elif filter_popup_strict and POPUP_REQUIRE_NON_ZERO_GRID and (
            sum(int(v or 0) for v in (r.get('ATR_Grid') or {}).values()) <= 0 and sum(int(v or 0) for v in (r.get('BB_Grid') or {}).values()) <= 0
        ):
            r['RejectReason'] = "Kinetic Decay (Both ATR & BB Grids at 0 0 0 0)"
            graveyard_candidates.append(r)
        else:
            surviving_bears.append(r)

    sort_rows_by_script_count(surviving_bulls)
    sort_rows_by_script_count(surviving_bears)
    if _is_index_option_mode():
        surviving_bulls = _dedupe_best_index_strike(surviving_bulls)
        surviving_bears = _dedupe_best_index_strike(surviving_bears)

    def sort_reason(r):
        return 0 if any(x in str(r.get('RejectReason', '')) for x in ["Hard stop", "Profit Lock", "Kinetic SL", "Delta Trail", "Delta Swing", "Superseded"]) else 1 if "Adverse" in str(r.get('RejectReason', '')) else 2 if any(x in str(r.get('RejectReason', '')) for x in ("Warzone", "Dual-Band Gate", "Delta-BB Gate")) else 3

    dedup_grave, seen_grave = [], set()
    for row in graveyard_candidates:
        gid = (row['Symbol'], row.get('ActiveAnchor', '-'), row.get('KilledTime', '-'))
        if gid not in seen_grave:
            seen_grave.add(gid)
            dedup_grave.append(row)

    grave_counts = {}
    for x in dedup_grave:
        gk = _script_group_key(x)
        grave_counts[gk] = grave_counts.get(gk, 0) + 1
    dedup_grave.sort(key=lambda x: (-grave_counts.get(_script_group_key(x), 0), _script_group_key(x), sort_reason(x), -_r_multiple(x) if _r_multiple(x) != 0 else -abs(x.get('DayChangePct', 0.0)), str(x.get('Symbol', ''))))

    return surviving_bulls, surviving_bears, dedup_grave

# ==============================================================================
# 3B. IN-MEMORY MULTI-ITERATION FIRST-TIME POPUP ENGINE (ZERO FILE STORAGE)
# ==============================================================================
def compute_lookback_iteration_schedule(cutoff_dt, num_iterations=None, interval_min=None):
    cfg_iters, cfg_interval = get_lookback_iterations_config()
    n_iters = cfg_iters if num_iterations is None else max(0, int(num_iterations))
    step_min = cfg_interval if interval_min is None else max(1, int(interval_min))
    try:
        pure_lb_min = int(PURE_ANCHOR_LOOKBACK_MIN or 0)
    except Exception:
        pure_lb_min = 0

    session_open_dt = _open_dt(cutoff_dt.date())
    iter_window_start_dt = max(session_open_dt, cutoff_dt - timedelta(minutes=n_iters * step_min))
    # Honor PURE_ANCHOR_LOOKBACK_MIN strictly for the Pure 1st-Time anchor window (don't silently stretch to n_iters * step_min)
    if pure_lb_min > 0:
        window_start_dt = max(session_open_dt, cutoff_dt - timedelta(minutes=pure_lb_min))
    else:
        window_start_dt = iter_window_start_dt

    prior_iter_dts = []
    for k in range(1, n_iters + 1):
        it_dt = cutoff_dt - timedelta(minutes=k * step_min)
        if it_dt >= session_open_dt and it_dt not in prior_iter_dts:
            prior_iter_dts.append(it_dt)

    return {
        "enabled": bool(ENABLE_ITERATION_LOOKBACK and (n_iters > 0 or pure_lb_min > 0)),
        "num_iterations": n_iters,
        "interval_min": step_min,
        "window_start_dt": window_start_dt,
        "iter_window_start_dt": iter_window_start_dt,
        "window_end_dt": cutoff_dt,
        "prior_iter_dts_desc": prior_iter_dts,
        "prior_iter_dts_chrono": sorted(prior_iter_dts),
    }

def _anchor_within_lookback_window(anchor_str, window_start_dt, window_end_dt):
    if not anchor_str or anchor_str == "-": return False
    clean_t = str(anchor_str).replace("*", "").strip()
    if ' ' in clean_t:
        return False
    try:
        parsed_t = datetime.strptime(clean_t, "%H:%M").time()
        anchor_dt = datetime.combine(window_end_dt.date(), parsed_t)
        return window_start_dt <= anchor_dt <= window_end_dt
    except ValueError:
        return False

def _passes_strict_popup_quality(row, check_gap=True, check_grid=True, min_r=None):
    sym = str(row.get('Symbol') or '')
    if OPTIONS_BUYING_ONLY and _is_option_symbol(sym):
        if str(row.get('PremiumDir') or 'BULL').upper() != 'BULL': return False
        anc_dir = str(row.get('AnchorDir') or '').upper()
        if anc_dir == 'BULL' and not _is_ce_symbol(sym): return False
        if anc_dir == 'BEAR' and not _is_pe_symbol(sym): return False

    if check_gap and ENABLE_TIMESTAMP_GAP_CHECK and MAX_TIMESTAMP_GAP_MINUTES > 0:
        if _get_timestamp_gap_min(row) > MAX_TIMESTAMP_GAP_MINUTES:
            return False

    if not STRICT_FIRST_TIME_POPUP_FILTER: return True
    live_r = _r_multiple(row, include_banked=False)
    if live_r < float(POPUP_MIN_R_MULTIPLE if min_r is None else min_r) - 1e-6: return False
    if (not _is_swing_mode()) and float(POPUP_MAX_R_MULTIPLE or 0.0) > 0 and live_r > float(POPUP_MAX_R_MULTIPLE):
        return False
    if (not _is_swing_mode()) and float(POPUP_MAX_PULLBACK_FROM_PEAK_R or 0.0) > 0:
        _pdir = str(row.get('PremiumDir') or row.get('AnchorDir') or 'BULL').upper()
        _ap = float(row.get('AnchorPrice', 0.0) or 0.0)
        _pk = float(row.get('PeakPrice', 0.0) or 0.0)
        _batr = float(row.get('BaseAtr', 0.0) or 0.0)
        peak_r = (((_ap - _pk) if _pdir == 'BEAR' else (_pk - _ap)) / _batr) if (_ap > 0 and _pk > 0 and _batr > 0) else live_r
        if (peak_r - live_r) > float(POPUP_MAX_PULLBACK_FROM_PEAK_R):
            return False

    if POPUP_REQUIRE_NON_ZERO_GRID and check_grid:
        ag, bg = row.get('ATR_Grid') or {}, row.get('BB_Grid') or {}
        if sum(int(v or 0) for v in ag.values()) <= 0 and sum(int(v or 0) for v in bg.values()) <= 0:
            return False
    return True

def classify_setup_popup_type(row, appearances_map=None, window_start_dt=None, window_end_dt=None, curr_label=None):
    """Single canonical source of truth for PopupType and FirstPopupIter."""
    sym = str(row.get('Symbol', '') or '')
    curr_anc = str(row.get('ActiveAnchor') or '-').replace('*', '').strip()
    h = _attempt_history(row)
    prior_rs = [r for _, _, r in h.get("prior", [])]
    reanchor_pos = _prior_attempts_qualify(prior_rs)
    prior_stops = int(row.get('PriorStops', 0) or 0)
    banked_r = float(row.get('BankedR', 0.0) or 0.0)
    has_star_badge = (prior_stops > 0 and abs(banked_r) > BANKED_R_DISPLAY_EPSILON) or ('*' in str(row.get('ActiveAnchor', ''))) or (len(prior_rs) > 0)

    if _is_swing_mode():
        days_held = int(row.get('DaysHeld', 0) or 0)
        first_it = str(row.get('AnchorDate') or '')[-5:] if row.get('AnchorDate') else (curr_anc if curr_anc != '-' else (curr_label or '-'))
        if 0 < days_held <= int(SWING_FRESH_SESSIONS or 2):
            popup_type = "SWING-NEW" if not has_star_badge else "SWING-RE+"
        else:
            popup_type = f"SWING-{days_held}D" if days_held > 0 else "SWING-HOLD"
        row['PopupType'] = popup_type
        row['FirstPopupIter'] = first_it
        return popup_type, first_it

    same_anc_iters, other_anc_iters = [], []
    if appearances_map and sym in appearances_map:
        app_list = appearances_map.get(sym, [])
        same_anc_iters = [it_lbl for (it_lbl, anc) in app_list if anc == curr_anc and anc != '-']
        if window_start_dt and window_end_dt:
            other_anc_iters = [it_lbl for (it_lbl, anc) in app_list if anc != curr_anc and _anchor_within_lookback_window(anc, window_start_dt, window_end_dt)]

    # FirstPopupIter = earliest detection iteration/checkpoint for this anchor (never overwrite with later snapshot time)
    first_seen_cp = str(row.get('FirstSeenIter') or '').strip()
    if same_anc_iters and first_seen_cp:
        first_it = same_anc_iters[0] if _hhmm_to_min(same_anc_iters[0]) <= _hhmm_to_min(first_seen_cp) else first_seen_cp
    elif same_anc_iters:
        first_it = same_anc_iters[0]
    elif first_seen_cp:
        first_it = first_seen_cp
    elif curr_label:
        first_it = curr_label
    else:
        cp_dt = row.get('CheckpointTime')
        first_it = cp_dt.strftime('%H:%M') if hasattr(cp_dt, 'strftime') else (curr_anc if curr_anc != '-' else '-')

    if not has_star_badge and prior_stops == 0 and len(other_anc_iters) == 0 and not prior_rs:
        popup_type = "1ST-TIME"
    elif reanchor_pos:
        popup_type = "RE-ANCHOR+"
    elif len(other_anc_iters) == 0 and not prior_rs:
        popup_type = "1ST-IN-ITERS*"
    else:
        popup_type = "RE-ANCHOR*"

    row['PopupType'] = popup_type
    row['FirstPopupIter'] = first_it
    return popup_type, first_it

def evaluate_first_time_popups_in_memory(by_symbol_all_cps, current_bulls, current_bears, lookback_meta, pool_bulls=None, pool_bears=None):
    if _is_swing_mode():
        # In Swing mode: Pure 1st-Time = Fresh Swing anchors within SWING_FRESH_SESSIONS; Window = all active swings
        curr_label = lookback_meta["window_end_dt"].strftime("%H:%M") if lookback_meta else ""
        src_b = pool_bulls if pool_bulls is not None else current_bulls
        src_r = pool_bears if pool_bears is not None else current_bears
        new_b, new_r, win_b, win_r = [], [], [], []
        for src, out_new, out_win in ((src_b, new_b, win_b), (src_r, new_r, win_r)):
            for row in (src or []):
                classify_setup_popup_type(row, curr_label=curr_label)
                if not _popup_row_ok(row):
                    continue
                rc = dict(row)
                rc['TargetPrice'], rc['SLPrice'] = _compute_target_sl(rc)
                rc['RR_Val'], rc['RR_Str'] = compute_risk_reward_ratio(rc)
                out_win.append(rc)
                days_held = int(rc.get('DaysHeld', 0) or 0)
                if 0 < days_held <= int(SWING_FRESH_SESSIONS or 2):
                    out_new.append(rc)
        sort_rows_by_script_count(new_b)
        sort_rows_by_script_count(new_r)
        sort_rows_by_script_count(win_b)
        sort_rows_by_script_count(win_r)
        return {"new_bulls": new_b, "new_bears": new_r, "window_first_bulls": win_b, "window_first_bears": win_r, "iter_stats": []}

    if not lookback_meta or not lookback_meta.get("enabled"):
        return {"new_bulls": [], "new_bears": [], "window_first_bulls": [], "window_first_bears": [], "iter_stats": []}

    w_start = lookback_meta["window_start_dt"]
    w_end = lookback_meta["window_end_dt"]
    prior_dts = lookback_meta["prior_iter_dts_chrono"]

    prior_bull_appearances, prior_bear_appearances, iter_stats = {}, {}, []

    for it_dt in prior_dts:
        it_cps = set(generate_checkpoints(it_dt, interval_min=CHECKPOINT_INTERVAL_MIN) if BACKTRACE_CHECKPOINTS else [it_dt])
        it_cps.add(it_dt)

        sub_map = {}
        for sym, rows in by_symbol_all_cps.items():
            f_rows = [r for r in rows if r['CheckpointTime'] <= it_dt and r['CheckpointTime'] in it_cps]
            if not f_rows: f_rows = [r for r in rows if r['CheckpointTime'] <= it_dt]
            if f_rows: sub_map[sym] = f_rows

        it_bulls, it_bears, it_grave = segregate_and_rank_buckets(sub_map, filter_timestamp_gap=False, filter_popup_strict=False) if sub_map else ([], [], [])
        it_bulls = [r for r in it_bulls if _popup_row_ok(r)]
        it_bears = [r for r in it_bears if _popup_row_ok(r)]
        it_label = it_dt.strftime("%H:%M")
        iter_stats.append({"iter_time": it_label, "bull_count": len(it_bulls), "bear_count": len(it_bears), "grave_count": len(it_grave)})

        for r in it_bulls: prior_bull_appearances.setdefault(r['Symbol'], []).append((it_label, str(r.get('ActiveAnchor') or '-').replace('*', '').strip()))
        for r in it_bears: prior_bear_appearances.setdefault(r['Symbol'], []).append((it_label, str(r.get('ActiveAnchor') or '-').replace('*', '').strip()))

    curr_label = w_end.strftime("%H:%M")
    new_bulls, new_bears, window_first_bulls, window_first_bears = [], [], [], []

    for r in (pool_bulls or []):
        classify_setup_popup_type(r, prior_bull_appearances, w_start, w_end, curr_label)
    for r in (pool_bears or []):
        classify_setup_popup_type(r, prior_bear_appearances, w_start, w_end, curr_label)
    for r in (current_bulls or []):
        classify_setup_popup_type(r, prior_bull_appearances, w_start, w_end, curr_label)
    for r in (current_bears or []):
        classify_setup_popup_type(r, prior_bear_appearances, w_start, w_end, curr_label)

    def _classify_rows(pool_list, prior_app_map, out_pure_first, out_all_window):
        for row in pool_list:
            sym = row['Symbol']
            curr_anc = str(row.get('ActiveAnchor') or '-').replace('*', '').strip()
            prior_rs = [r for _, _, r in _attempt_history(row)["prior"]]
            reanchor_pos = _prior_attempts_qualify(prior_rs)
            in_window = _anchor_within_lookback_window(curr_anc, w_start, w_end)
            any_time = bool(PURE_REANCHOR_ANY_TIME and reanchor_pos)
            if not (in_window or any_time) or not _popup_row_ok(row, any_time=any_time):
                continue

            appearances = prior_app_map.get(sym, [])
            same_anc_iters = [it_lbl for (it_lbl, anc) in appearances if anc == curr_anc]
            if PURE_1ST_TIME_CURRENT_ITER_ONLY and len(same_anc_iters) > 0: continue

            row_copy = dict(row)
            row_copy['TargetPrice'], row_copy['SLPrice'] = _compute_target_sl(row_copy)
            row_copy['RR_Val'], row_copy['RR_Str'] = compute_risk_reward_ratio(row_copy)

            p_type = row.get('PopupType', '1ST-TIME')
            row_copy['PopupType'] = p_type
            row_copy['FirstPopupIter'] = row.get('FirstPopupIter', curr_label)

            if p_type == "1ST-TIME":
                if PURE_INCLUDE_PLAIN_FIRST_TIME: out_pure_first.append(row_copy)
                out_all_window.append(row_copy)
            elif p_type == "RE-ANCHOR+":
                out_pure_first.append(row_copy)
                out_all_window.append(row_copy)
            else:
                # Losing RE-ANCHOR* and 1ST-IN-ITERS* stay in All Window only — never leak into Pure 1st-Time
                out_all_window.append(row_copy)

    _classify_rows(pool_bulls if pool_bulls is not None else current_bulls, prior_bull_appearances, new_bulls, window_first_bulls)
    _classify_rows(pool_bears if pool_bears is not None else current_bears, prior_bear_appearances, new_bears, window_first_bears)

    sort_rows_by_script_count(new_bulls)
    sort_rows_by_script_count(new_bears)
    sort_rows_by_script_count(window_first_bulls)
    sort_rows_by_script_count(window_first_bears)

    return {"new_bulls": new_bulls, "new_bears": new_bears, "window_first_bulls": window_first_bulls, "window_first_bears": window_first_bears, "iter_stats": iter_stats}

def _state_display(row):
    return f"[STOP {row.get('KilledTime')}]" if row['State'] == "[STOPPED]" and row.get('KilledTime') and row.get('KilledTime') != '-' else row['State']

def _fav_color(pct, direction):
    fav = pct * (1 if direction == "BULL" else -1)
    return COLOR_GREEN_FG if fav > 0 else COLOR_RED_FG if fav < 0 else COLOR_DIM

def _format_val(v, dir_color):
    return f"{dir_color}{v}{COLOR_RESET}" if v > 0 else f"{COLOR_DIM}{v}{COLOR_RESET}"

# ==============================================================================
# 4. CONSOLE RENDERER & PURE 1ST-TIME EMAIL / LIGHT-THEME HTML ENGINE
# ==============================================================================
def render_console_dashboard(
    mode, cutoff_dt, checkpoints,
    idx_bulls, idx_bears, idx_grave,
    stock_bulls, stock_bears, stock_grave,
    active_sector_filter=None,
    lookback_meta=None,
    idx_popup_res=None,
    stock_popup_res=None
):
    norm_filter = _norm_sym(active_sector_filter) if active_sector_filter else ""
    if norm_filter:
        f_stock_bulls = [r for r in stock_bulls if _norm_sym(r.get('Sector', '')) == norm_filter or norm_filter in _norm_sym(r.get('Sector', ''))]
        f_stock_bears = [r for r in stock_bears if _norm_sym(r.get('Sector', '')) == norm_filter or norm_filter in _norm_sym(r.get('Sector', ''))]
        f_stock_grave = [r for r in stock_grave if _norm_sym(r.get('Sector', '')) == norm_filter or norm_filter in _norm_sym(r.get('Sector', ''))]
        filter_banner = f" [ACTIVE SECTOR FILTER: {active_sector_filter.upper()}]"
    else:
        f_stock_bulls = stock_bulls if TOP_N_BUYERS <= 0 else stock_bulls[:TOP_N_BUYERS]
        f_stock_bears = stock_bears if TOP_N_SELLERS <= 0 else stock_bears[:TOP_N_SELLERS]
        f_stock_grave = stock_grave if TOP_N_GRAVEYARD <= 0 else stock_grave[:TOP_N_GRAVEYARD]
        filter_banner = " [DEFAULT VIEW: SHOWING ALL INDICES & ALL STOCKS]"

    new_idx_bulls = (idx_popup_res or {}).get("new_bulls", [])
    new_idx_bears = (idx_popup_res or {}).get("new_bears", [])
    win_idx_bulls = (idx_popup_res or {}).get("window_first_bulls", [])
    win_idx_bears = (idx_popup_res or {}).get("window_first_bears", [])
    new_stk_bulls = (stock_popup_res or {}).get("new_bulls", [])
    new_stk_bears = (stock_popup_res or {}).get("new_bears", [])
    win_stk_bulls = (stock_popup_res or {}).get("window_first_bulls", [])
    win_stk_bears = (stock_popup_res or {}).get("window_first_bears", [])

    all_shown = (
        idx_bulls + idx_bears + idx_grave + f_stock_bulls + f_stock_bears + f_stock_grave
        + new_idx_bulls + new_idx_bears + win_idx_bulls + win_idx_bears
        + new_stk_bulls + new_stk_bears + win_stk_bulls + win_stk_bears
    )
    sym_w = max(MIN_SYMBOL_WIDTH, min(MAX_SYMBOL_WIDTH, max((len(r.get('Symbol', '')) for r in all_shown), default=MIN_SYMBOL_WIDTH)))
    anc_w = max(12 if _is_swing_mode() else 7, max((len(str(r.get('ActiveAnchor', '-'))) + 1 for r in all_shown), default=7))
    sec_w = max(
        SECTOR_COL_WIDTH,
        max(
            (
                len(f"{r.get('Sector', DEFAULT_UNMAPPED_SECTOR)} ({r.get('SectorWeight', 0.0):.1f}%)")
                for r in all_shown
            ),
            default=SECTOR_COL_WIDTH,
        ),
    )

    print(f"\n{COLOR_BOLD}=== STATEFUL INSTITUTIONAL VOLATILITY TRACKER [{mode}]{filter_banner} ==={COLOR_RESET}")
    print(
        f"Target Snapshot: {cutoff_dt.strftime('%Y-%m-%d %H:%M')} | Checkpoints: {len(checkpoints)} | "
        f"Sectoral Indices (Bull: {len(idx_bulls)}, Bear: {len(idx_bears)}, Grave: {len(idx_grave)}) | "
        f"Stocks Shown (Bull: {len(f_stock_bulls)}, Bear: {len(f_stock_bears)}, Grave: {len(f_stock_grave)})\n"
        f"{COLOR_DIM}{_report_legend_text()}{COLOR_RESET}\n"
    )

    def print_active_table(title, icon, data_list, show_sector_col=True, show_first_iter_col=False, show_target_sl=True):
        if not data_list: return
        print(f"\n{COLOR_BOLD}{icon} {title}{COLOR_RESET}")
        grid_hdr = " ".join(f"{m}X" for m in GRID_MULTIPLIERS)
        h_atr = f"ATR: {grid_hdr}"
        h_bb = f"BB: {grid_hdr}"
        tsl_hdr = f"{'Target':>12} {'SL':>9} | " if show_target_sl else ""
        sec_hdr = f"{'Sector (MaxWt%)':<{sec_w}} | " if show_sector_col else ""
        iter_hdr = f"{'1stIter':^7} {'PopupType':^13} | " if show_first_iter_col else ""
        header_str = (
            f" {COLOR_CYAN}{'Script':<{sym_w}} {'R:R':^7} {'LTP/Exit':>9} | "
            f"{tsl_hdr}{sec_hdr}{iter_hdr}"
            f"{'Seen':^5} {'Anchor':^{anc_w}} {'Anch Val':>9} {'R-Mult (Bank)':>15} | "
            f"{h_atr:^16} | {h_bb:^15} | {'State':^13}{COLOR_RESET}"
        )
        print(header_str)
        print("-" * len(ANSI_RE.sub("", header_str)))
        basket_dir = "BULL" if "BULL" in title else "BEAR"
        default_g = {m: 0 for m in GRID_MULTIPLIERS}

        for row in data_list:
            live_r, banked_r = _r_multiple(row, include_banked=False), float(row.get('BankedR', 0.0) or 0.0)
            rank_r = _rank_r_score(row)
            r_color = COLOR_GREEN_FG if rank_r > 0 else COLOR_RED_FG if rank_r < 0 else COLOR_DIM
            has_bank = row.get('PriorStops', 0) > 0 and abs(banked_r) > BANKED_R_DISPLAY_EPSILON
            r_txt = f"{live_r:>+5.2f}R ({banked_r:>+4.1f}R)" if has_bank else f"{live_r:>+5.2f}R"
            r_mult_str = f"{r_txt:>15}"
            anchor_disp = f"{row.get('ActiveAnchor', '-')}{'*' if has_bank else ''}"

            dir_c = COLOR_GREEN_FG if basket_dir == "BULL" else COLOR_RED_FG
            ag, bg = row.get('ATR_Grid', default_g), row.get('BB_Grid', default_g)
            a_str = "  ".join(_format_val(ag.get(k, 0), dir_c) for k in GRID_MULTIPLIERS)
            b_str = "  ".join(_format_val(bg.get(k, 0), dir_c) for k in GRID_MULTIPLIERS)
            a_cell = f"   {a_str}   "
            b_cell = f"  {b_str}   "

            shown_state = _state_display(row)
            if "[ BB > KC ]" in row['State']:
                dir_tag = "▲" if basket_dir == "BULL" else "▼"
                state_text = f"{COLOR_GREEN_FG if dir_tag == '▲' else COLOR_RED_FG}{f'[ BB > KC ] {dir_tag}'.strip():^13}{COLOR_RESET}"
            else:
                state_text = f"{COLOR_YELLOW if '[ BB < KC ]' in row['State'] else COLOR_DIM}{shown_state:^13}{COLOR_RESET}"

            ltp_val = f"{float(row['KilledPrice']):>8.2f}x" if ("[STOP" in shown_state and float(row.get('KilledPrice', 0.0) or 0.0) > 0) else f"{float(row.get('LTP', 0.0) or 0.0):>9.2f}"
            _tg = _target_cell_text(row)
            sl_val = f"{float(row.get('SLPrice', 0.0) or 0.0):>9.2f}"
            tsl_cell = f"{COLOR_YELLOW}{_tg[:3]}{COLOR_RESET}{COLOR_GREEN_FG}{_tg[3:]}{COLOR_RESET} {COLOR_RED_FG}{sl_val}{COLOR_RESET} | " if show_target_sl else ""

            sec_str = f"{row.get('Sector', DEFAULT_UNMAPPED_SECTOR)} ({row.get('SectorWeight', 0.0):.1f}%)"
            sec_cell = f"{_clip(sec_str, sec_w, False):<{sec_w}} | " if show_sector_col else ""

            cp_val = row.get('CheckpointTime')
            seen_val = cp_val.strftime('%H:%M') if hasattr(cp_val, 'strftime') else str(cp_val or '-')
            first_iter_val = str(row.get('FirstPopupIter', seen_val))
            popup_type_val = str(row.get('PopupType', '1ST-TIME'))
            iter_cell = f"{first_iter_val:^7} {popup_type_val:^13} | " if show_first_iter_col else ""

            anch_val_str = f"{float(row.get('AnchorPrice', 0.0) or 0.0):>9.2f}"
            rr_str = row.get('RR_Str') or compute_risk_reward_ratio(row)[1]

            print(
                f" {COLOR_BOLD}{_clip(row['Symbol'], sym_w):<{sym_w}}{COLOR_RESET} {COLOR_CYAN}{rr_str:^7}{COLOR_RESET} {ltp_val} | "
                f"{tsl_cell}{sec_cell}{iter_cell}"
                f"{seen_val:^5} {anchor_disp:^{anc_w}} {anch_val_str} "
                f"{r_color}{r_mult_str}{COLOR_RESET} | {a_cell} | {b_cell} | {state_text}"
            )
            _pa = _prior_attempts_text(row)
            if _pa:
                print(f"   {COLOR_DIM}↳ {_pa}{COLOR_RESET}")

    def print_grave_table(title, icon, data_list, show_sector_col=True):
        if not data_list: return
        print(f"\n{COLOR_BOLD}{icon} {title}{COLOR_RESET}")
        sec_hdr = f"{'Sector':<{sec_w}} | " if show_sector_col else ""
        header_str = (
            f" {COLOR_CYAN}{'Script':<{sym_w}} | {sec_hdr}{'Signal':^6} {'Anchor':^{anc_w}} {'Anch Val':>9} "
            f"{'Killed':^{anc_w}} {'Kill Val':>9} {'LTP':>9} {'R-Mult':>7} {'Move%':>7} | "
            f"{'Forensic Rejection / Exit Reason'}{COLOR_RESET}"
        )
        print(header_str)
        print("-" * len(ANSI_RE.sub("", header_str)))
        for row in data_list:
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
            reason_color = COLOR_YELLOW if any(x in reason for x in ["Kinetic SL", "Hard stop", "Adverse", "Profit Lock", "Delta Trail", "Delta Swing", "Superseded", "Timestamp"]) else COLOR_RED_FG if any(x in reason for x in ["Chop", "Warzone", "Dual-Band Gate", "Delta-BB Gate"]) else COLOR_DIM
            sec_cell = f"{_clip(row.get('Sector', DEFAULT_UNMAPPED_SECTOR), sec_w):<{sec_w}} | " if show_sector_col else ""
            print(
                f" {COLOR_BOLD}{_clip(row['Symbol'], sym_w):<{sym_w}}{COLOR_RESET} | {sec_cell}{sig_colored} "
                f"{a_time:^{anc_w}} {a_price} {k_time:^{anc_w}} {k_price} {row['LTP']:>9.2f} "
                f"{g_r_color}{g_r_str}{COLOR_RESET} {move_color}{move_pct_str}{COLOR_RESET} | {reason_color}{reason}{COLOR_RESET}"
            )

    if idx_bulls or idx_bears or idx_grave:
        print(f"{COLOR_CYAN}{'=' * 95}{COLOR_RESET}")
        print(f"{COLOR_BOLD}🏛️  STAGE 1: SECTORAL INDICES (Evaluated with 3-Lock Kinetic Engine){COLOR_RESET}")
        print(f"{COLOR_CYAN}{'=' * 95}{COLOR_RESET}")
        print_active_table("SECTORAL INDICES — TOP BULL SETUPS", "🏛️🔥", idx_bulls, show_sector_col=False)
        print_active_table("SECTORAL INDICES — TOP BEAR SETUPS", "🏛️🩸", idx_bears, show_sector_col=False)
        print_grave_table("SECTORAL INDICES — THE GRAVEYARD", "🏛️🚫", idx_grave, show_sector_col=False)

    print(f"\n{COLOR_CYAN}{'=' * 95}{COLOR_RESET}")
    stage2_label = f"STAGE 2: STOCKS FOR SECTOR [{active_sector_filter.upper()}]" if norm_filter else "STAGE 2: ALL ACTIVE STOCKS (Mapped by Highest Sectoral Index Weightage)"
    print(f"{COLOR_BOLD}📈  {stage2_label}{COLOR_RESET}")
    print(f"{COLOR_CYAN}{'=' * 95}{COLOR_RESET}")
    print_active_table(f"STOCKS — TOP BULL SETUPS — BULL {_format_count_with_ce_pe(f_stock_bulls)}", "🔥", f_stock_bulls, show_sector_col=True, show_first_iter_col=True, show_target_sl=True)
    print_active_table(f"STOCKS — TOP BEAR SETUPS — BEAR {_format_count_with_ce_pe(f_stock_bears)}", "🩸", f_stock_bears, show_sector_col=True, show_first_iter_col=True, show_target_sl=True)
    if not f_stock_bulls and not f_stock_bears:
        print(f"{COLOR_YELLOW}No surviving stock anchors for this view.{COLOR_RESET}")
    print_grave_table("STOCKS — THE GRAVEYARD", "🚫", f_stock_grave, show_sector_col=True)

    if ((lookback_meta and lookback_meta.get("enabled")) or _is_swing_mode()) and stock_popup_res is not None:
        w_start_s = lookback_meta["window_start_dt"].strftime("%H:%M") if lookback_meta else "-"
        w_end_s = lookback_meta["window_end_dt"].strftime("%H:%M") if lookback_meta else cutoff_dt.strftime("%H:%M")

        if norm_filter:
            new_stk_bulls = [r for r in new_stk_bulls if _norm_sym(r.get('Sector', '')) == norm_filter or norm_filter in _norm_sym(r.get('Sector', ''))]
            new_stk_bears = [r for r in new_stk_bears if _norm_sym(r.get('Sector', '')) == norm_filter or norm_filter in _norm_sym(r.get('Sector', ''))]
            win_stk_bulls = [r for r in win_stk_bulls if _norm_sym(r.get('Sector', '')) == norm_filter or norm_filter in _norm_sym(r.get('Sector', ''))]
            win_stk_bears = [r for r in win_stk_bears if _norm_sym(r.get('Sector', '')) == norm_filter or norm_filter in _norm_sym(r.get('Sector', ''))]

        stage3_title = f"STAGE 3: FRESH SWING POPUPS (Last {int(SWING_FRESH_SESSIONS)} Sessions)" if _is_swing_mode() else f"STAGE 3: FIRST-TIME POPUPS (Anchor Window: {w_start_s}..{w_end_s})"
        print(f"\n{COLOR_CYAN}{'=' * 95}{COLOR_RESET}")
        print(f"{COLOR_BOLD}🆕  {stage3_title}{COLOR_RESET}")
        print(f"{COLOR_DIM}{_pure_note_text()}{COLOR_RESET}")
        print(f"{COLOR_CYAN}{'=' * 95}{COLOR_RESET}")

        if new_idx_bulls or new_idx_bears:
            print_active_table(f"PURE FIRST-TIME SECTORAL INDEX POPUPS — BULL ({len(new_idx_bulls)})", "🏛️🆕🔥", new_idx_bulls, show_sector_col=False, show_first_iter_col=True, show_target_sl=True)
            print_active_table(f"PURE FIRST-TIME SECTORAL INDEX POPUPS — BEAR ({len(new_idx_bears)})", "🏛️🆕🩸", new_idx_bears, show_sector_col=False, show_first_iter_col=True, show_target_sl=True)

        print_active_table(f"PURE FIRST-TIME STOCK POPUPS — BULL {_format_count_with_ce_pe(new_stk_bulls)}", "🆕🔥", new_stk_bulls, show_sector_col=True, show_first_iter_col=True, show_target_sl=True)
        print_active_table(f"PURE FIRST-TIME STOCK POPUPS — BEAR {_format_count_with_ce_pe(new_stk_bears)}", "🆕🩸", new_stk_bears, show_sector_col=True, show_first_iter_col=True, show_target_sl=True)

        if win_idx_bulls or win_idx_bears:
            print_active_table(f"ALL WINDOW SECTORAL INDEX POPUPS — BULL ({len(win_idx_bulls)})", "🏛️⏱️🔥", win_idx_bulls, show_sector_col=False, show_first_iter_col=True, show_target_sl=True)
            print_active_table(f"ALL WINDOW SECTORAL INDEX POPUPS — BEAR ({len(win_idx_bears)})", "🏛️⏱️🩸", win_idx_bears, show_sector_col=False, show_first_iter_col=True, show_target_sl=True)

        print_active_table(f"ALL WINDOW STOCK POPUPS — BULL {_format_count_with_ce_pe(win_stk_bulls)}", "⏱️🔥", win_stk_bulls, show_sector_col=True, show_first_iter_col=True, show_target_sl=True)
        print_active_table(f"ALL WINDOW STOCK POPUPS — BEAR {_format_count_with_ce_pe(win_stk_bears)}", "⏱️🩸", win_stk_bears, show_sector_col=True, show_first_iter_col=True, show_target_sl=True)

def build_full_untruncated_plain_text(
    mode, cutoff_dt, checkpoints, idx_bulls, idx_bears, idx_grave,
    stock_bulls, stock_bears, stock_grave, lookback_meta=None,
    idx_popup_res=None, stock_popup_res=None
):
    new_idx_b = (idx_popup_res or {}).get("new_bulls", [])
    new_idx_r = (idx_popup_res or {}).get("new_bears", [])
    win_idx_b = (idx_popup_res or {}).get("window_first_bulls", [])
    win_idx_r = (idx_popup_res or {}).get("window_first_bears", [])
    new_stk_b = (stock_popup_res or {}).get("new_bulls", [])
    new_stk_r = (stock_popup_res or {}).get("new_bears", [])
    win_stk_b = (stock_popup_res or {}).get("window_first_bulls", [])
    win_stk_r = (stock_popup_res or {}).get("window_first_bears", [])

    pure_rows = new_idx_b + new_idx_r + new_stk_b + new_stk_r
    popup_rows = pure_rows + win_idx_b + win_idx_r + win_stk_b + win_stk_r
    e_bulls = stock_bulls if EMAIL_MAX_BUYERS <= 0 else stock_bulls[:EMAIL_MAX_BUYERS]
    e_bears = stock_bears if EMAIL_MAX_SELLERS <= 0 else stock_bears[:EMAIL_MAX_SELLERS]
    e_grave = stock_grave if EMAIL_MAX_GRAVEYARD <= 0 else stock_grave[:EMAIL_MAX_GRAVEYARD]

    all_eval_rows = (
        pure_rows + popup_rows + idx_bulls + idx_bears + idx_grave
        + stock_bulls + stock_bears + stock_grave + e_bulls + e_bears + e_grave
    )
    sym_w = max(MIN_SYMBOL_WIDTH, max((len(r.get('Symbol', '')) for r in all_eval_rows), default=MIN_SYMBOL_WIDTH))
    anc_w = max(12 if _is_swing_mode() else 7, max((len(str(r.get('ActiveAnchor', '-'))) + 1 for r in all_eval_rows), default=7))
    sec_w = max(
        SECTOR_COL_WIDTH,
        max(
            (
                len(f"{r.get('Sector', DEFAULT_UNMAPPED_SECTOR)} ({r.get('SectorWeight', 0.0):.1f}%)")
                for r in all_eval_rows
            ),
            default=SECTOR_COL_WIDTH,
        ),
    )
    lines = []

    def _fmt_active_block(title, data_list, show_sector=True, show_first_iter=False, include_ce_pe_counts=True):
        count_suffix = _format_count_with_ce_pe(data_list) if include_ce_pe_counts else f"({len(data_list)})"
        if not data_list:
            lines.append(f"--- {title} {count_suffix} ---")
            lines.append("None\n")
            return
        lines.append(f"--- {title} {count_suffix} ---")
        grid_hdr = " ".join(f"{m}X" for m in GRID_MULTIPLIERS)
        h_atr = f"ATR: {grid_hdr}"
        h_bb = f"BB: {grid_hdr}"
        sec_hdr = f"{'Sector (MaxWt%)':<{sec_w}} | " if show_sector else ""
        iter_hdr = f"{'1stIter':^7} {'PopupType':^13} | " if show_first_iter else ""
        hdr = (
            f"{'Script':<{sym_w}} {'R:R':^7} {'LTP/Exit':>9} {'Target':>12} {'SL':>9} | "
            f"{sec_hdr}{iter_hdr}{'Seen':^5} {'Anchor':^{anc_w}} {'Anch Val':>9} {'R-Mult (Bank)':>15} | "
            f"{h_atr:^16} | {h_bb:^15} | {'State':^13}"
        )
        lines.append(hdr)
        lines.append("-" * len(hdr))
        basket_dir = "BULL" if "BULL" in title else "BEAR"
        default_g = {m: 0 for m in GRID_MULTIPLIERS}
        for r in data_list:
            live_r, banked_r = _r_multiple(r, include_banked=False), float(r.get('BankedR', 0.0) or 0.0)
            has_bank = r.get('PriorStops', 0) > 0 and abs(banked_r) > BANKED_R_DISPLAY_EPSILON
            r_txt = f"{live_r:>+5.2f}R ({banked_r:>+4.1f}R)" if has_bank else f"{live_r:>+5.2f}R"
            r_mult_str = f"{r_txt:>15}"
            anchor_disp = f"{r.get('ActiveAnchor', '-')}{'*' if has_bank else ''}"
            ag, bg = r.get('ATR_Grid', default_g), r.get('BB_Grid', default_g)
            a_str = "  ".join(str(ag.get(k, 0)) for k in GRID_MULTIPLIERS)
            b_str = "  ".join(str(bg.get(k, 0)) for k in GRID_MULTIPLIERS)
            shown_state = _state_display(r)
            if "[ BB > KC ]" in r['State']: shown_state = f"[ BB > KC ] {'▲' if basket_dir == 'BULL' else '▼'}"
            ltp_val = f"{float(r['KilledPrice']):>8.2f}x" if ("[STOP" in shown_state and float(r.get('KilledPrice', 0.0) or 0.0) > 0) else f"{float(r.get('LTP', 0.0) or 0.0):>9.2f}"
            tgt_cell = _target_cell_text(r)
            sl_val = f"{float(r.get('SLPrice', 0.0) or 0.0):>9.2f}"
            rr_str = r.get('RR_Str') or compute_risk_reward_ratio(r)[1]
            sec_str = f"{r.get('Sector', DEFAULT_UNMAPPED_SECTOR)} ({r.get('SectorWeight', 0.0):.1f}%)"
            sec_cell = f"{_clip(sec_str, sec_w, EMAIL_CLIP_SYMBOLS):<{sec_w}} | " if show_sector else ""

            cp_val = r.get('CheckpointTime')
            seen_val = cp_val.strftime('%H:%M') if hasattr(cp_val, 'strftime') else str(cp_val or '-')
            first_iter_val = str(r.get('FirstPopupIter', seen_val))
            popup_type_val = str(r.get('PopupType', '1ST-TIME'))
            iter_cell = f"{first_iter_val:^7} {popup_type_val:^13} | " if show_first_iter else ""
            anch_val_str = f"{float(r.get('AnchorPrice', 0.0) or 0.0):>9.2f}"

            lines.append(
                f"{_clip(r['Symbol'], sym_w, EMAIL_CLIP_SYMBOLS):<{sym_w}} {rr_str:^7} {ltp_val} {tgt_cell} {sl_val} | "
                f"{sec_cell}{iter_cell}{seen_val:^5} {anchor_disp:^{anc_w}} {anch_val_str} "
                f"{r_mult_str} | {a_str:^16} | {b_str:^15} | {shown_state:^13}"
            )
            _pa = _prior_attempts_text(r)
            if _pa:
                lines.append(f"   ↳ {_pa}")
        lines.append("")

    if ((lookback_meta and lookback_meta.get("enabled")) or _is_swing_mode()) and stock_popup_res is not None:
        w_start_s = lookback_meta["window_start_dt"].strftime("%H:%M") if lookback_meta else "-"
        w_end_s = lookback_meta["window_end_dt"].strftime("%H:%M") if lookback_meta else cutoff_dt.strftime("%H:%M")
        use_pure = bool(EMAIL_ONLY_PURE_1ST_TIME_CONTENT)
        sel_idx_b = new_idx_b if use_pure else win_idx_b
        sel_idx_r = new_idx_r if use_pure else win_idx_r
        sel_stk_b = new_stk_b if use_pure else win_stk_b
        sel_stk_r = new_stk_r if use_pure else win_stk_r
        scope_tag = _pure_label().upper() if use_pure else "ALL WINDOW"
        nb_ce, nb_pe = _ce_pe_counts(sel_stk_b)
        nr_ce, nr_pe = _ce_pe_counts(sel_stk_r)

        lines.append(f"=== SYSTEM 3.0 [{mode}] — {scope_tag} POPUPS REPORT ({w_start_s}..{w_end_s}) ===")
        lines.append(f"Snapshot: {cutoff_dt.strftime('%Y-%m-%d %H:%M')} | Anchor Window: {w_start_s} to {w_end_s}")
        lines.append(_report_legend_text())
        lines.append(_pure_note_text())
        lines.append(
            f"Sectoral Indices {scope_tag}: ▲ {len(sel_idx_b)} Bull / ▼ {len(sel_idx_r)} Bear | "
            f"Stocks {scope_tag}: ▲ {len(sel_stk_b)} Bull [CE({nb_ce}) PE({nb_pe})] / ▼ {len(sel_stk_r)} Bear [CE({nr_ce}) PE({nr_pe})]\n"
        )
        if sel_idx_b or sel_idx_r:
            _fmt_active_block(f"{scope_tag} SECTORAL INDEX POPUPS — BULL", sel_idx_b, show_sector=False, show_first_iter=True, include_ce_pe_counts=False)
            _fmt_active_block(f"{scope_tag} SECTORAL INDEX POPUPS — BEAR", sel_idx_r, show_sector=False, show_first_iter=True, include_ce_pe_counts=False)
        _fmt_active_block(f"{scope_tag} STOCK POPUPS — BULL", sel_stk_b, show_sector=True, show_first_iter=True, include_ce_pe_counts=True)
        _fmt_active_block(f"{scope_tag} STOCK POPUPS — BEAR", sel_stk_r, show_sector=True, show_first_iter=True, include_ce_pe_counts=True)
        return "\n".join(lines)

    lines.append(f"=== SYSTEM 3.0 INSTITUTIONAL VOLATILITY TRACKER [{mode}] ===")
    lines.append(f"Snapshot: {cutoff_dt.strftime('%Y-%m-%d %H:%M')}")
    lines.append(_report_legend_text() + "\n")
    if idx_bulls or idx_bears:
        _fmt_active_block("SECTORAL INDICES — TOP BULL SETUPS", idx_bulls, show_sector=False, include_ce_pe_counts=False)
        _fmt_active_block("SECTORAL INDICES — TOP BEAR SETUPS", idx_bears, show_sector=False, include_ce_pe_counts=False)
    _fmt_active_block("ALL STOCKS — TOP BULL SETUPS", e_bulls, show_sector=True, include_ce_pe_counts=True)
    _fmt_active_block("ALL STOCKS — TOP BEAR SETUPS", e_bears, show_sector=True, include_ce_pe_counts=True)
    return "\n".join(lines)

def build_email_and_attachment_html(
    mode, cutoff_dt, checkpoints, idx_bulls, idx_bears, idx_grave,
    stock_bulls, stock_bears, stock_grave, lookback_meta=None,
    idx_popup_res=None, stock_popup_res=None
):
    new_idx_b = (idx_popup_res or {}).get("new_bulls", [])
    new_idx_r = (idx_popup_res or {}).get("new_bears", [])
    win_idx_b = (idx_popup_res or {}).get("window_first_bulls", [])
    win_idx_r = (idx_popup_res or {}).get("window_first_bears", [])
    new_stk_b = (stock_popup_res or {}).get("new_bulls", [])
    new_stk_r = (stock_popup_res or {}).get("new_bears", [])
    win_stk_b = (stock_popup_res or {}).get("window_first_bulls", [])
    win_stk_r = (stock_popup_res or {}).get("window_first_bears", [])

    w_start_s = lookback_meta["window_start_dt"].strftime("%H:%M") if (lookback_meta and lookback_meta.get("enabled")) else "-"
    w_end_s = lookback_meta["window_end_dt"].strftime("%H:%M") if (lookback_meta and lookback_meta.get("enabled")) else cutoff_dt.strftime("%H:%M")

    def _ser_active(r, basket_dir):
        if float(r.get('TargetPrice', 0.0) or 0.0) <= 0 or float(r.get('SLPrice', 0.0) or 0.0) <= 0:
            r['TargetPrice'], r['SLPrice'] = _compute_target_sl(r)
        live_r = _r_multiple(r, include_banked=False)
        banked_r = float(r.get('BankedR', 0.0) or 0.0)
        has_bank = r.get('PriorStops', 0) > 0 and abs(banked_r) > BANKED_R_DISPLAY_EPSILON
        default_g = {m: 0 for m in GRID_MULTIPLIERS}
        ag, bg = r.get('ATR_Grid', default_g), r.get('BB_Grid', default_g)
        st = _state_display(r)
        if "[ BB > KC ]" in r['State']: st = f"[ BB > KC ] {'▲' if basket_dir == 'BULL' else '▼'}"

        cp_val = r.get('CheckpointTime')
        seen_s = cp_val.strftime('%H:%M') if hasattr(cp_val, 'strftime') else str(cp_val or '-')
        first_iter_s = str(r.get('FirstPopupIter') or seen_s)
        shown_ltp = f"{float(r['KilledPrice']):.2f}x" if ("[STOP" in st and float(r.get('KilledPrice', 0.0) or 0.0) > 0) else f"{float(r.get('LTP', 0.0) or 0.0):.2f}"

        return {
            "symbol": r['Symbol'],
            "rr": r.get('RR_Str') or compute_risk_reward_ratio(r)[1],
            "sector": r.get('Sector', DEFAULT_UNMAPPED_SECTOR),
            "weight": f"{float(r.get('SectorWeight', 0.0)):.1f}%",
            "first_iter": first_iter_s,
            "popup_type": r.get('PopupType', '1ST-TIME'),
            "ltp": shown_ltp,
            "tgt_tag": _target_tag(r),
            "prior": _prior_attempts_text(r),
            "target": f"{float(r.get('TargetPrice', 0.0) or 0.0):.2f}" if float(r.get('TargetPrice', 0.0) or 0.0) > 0 else "-",
            "sl": f"{float(r.get('SLPrice', 0.0) or 0.0):.2f}" if float(r.get('SLPrice', 0.0) or 0.0) > 0 else "-",
            "seen": seen_s,
            "anchor": f"{r.get('ActiveAnchor', '-')}{'*' if has_bank else ''}",
            "anchor_val": f"{float(r.get('AnchorPrice', 0.0) or 0.0):.2f}",
            "r_str": f"{live_r:+.2f}R ({banked_r:+.1f}R)" if has_bank else f"{live_r:+.2f}R",
            "rank_r_pos": _rank_r_score(r) >= 0,
            "atr_grid": "  ".join(str(ag.get(k, 0)) for k in GRID_MULTIPLIERS),
            "bb_grid": "  ".join(str(bg.get(k, 0)) for k in GRID_MULTIPLIERS),
            "state": st,
            "is_expanding": "[ BB > KC ]" in r['State'],
        }

    def _table_active_html(rows_list, basket_dir, show_sec=True, js_clickable_index=False, show_first_iter=True):
        if not rows_list:
            return "<div style='background-color:#ffffff;border:1px solid #cbd5e1;border-radius:6px;padding:10px 14px;color:#64748b;font-style:italic;margin-bottom:14px;'>No setups in this category for the current lookback window.</div>"
        th_style = "background-color:#1e293b;color:#ffffff;padding:8px 10px;border:1px solid #334155;font-weight:700;font-size:12px;"
        sec_th = f"<th style='{th_style}' align='left'>Sector (MaxWt%)</th>" if show_sec else ""
        iter_th = f"<th style='{th_style}' align='center'>1stIter</th><th style='{th_style}' align='center'>PopupType</th>" if show_first_iter else ""

        tr_list = []
        for idx_i, r in enumerate(rows_list):
            d = _ser_active(r, basket_dir)
            tgt_badge = f"<span style='font-size:10px;background-color:#fef3c7;color:#92400e;border-radius:3px;padding:1px 4px;margin-right:3px;'>{d['tgt_tag']}</span> " if d["tgt_tag"] else ""
            row_bg = "#ffffff" if idx_i % 2 == 0 else "#f8fafc"
            base_td = f"padding:7px 10px;border:1px solid #cbd5e1;background-color:{row_bg};color:#0f172a;font-size:12px;"
            raw_sec = d["symbol"] if not show_sec else d["sector"]
            sec_attr = html.escape(raw_sec, quote=True)
            click_attr = f"onclick=\"filterBySector({_js_onclick_arg(raw_sec)})\" class=\"idx-row\"" if js_clickable_index else ""
            sec_td = f"<td style='{base_td}color:#1e40af;font-weight:600;'>{html.escape(d['sector'])} <span style='color:#475569;font-weight:normal;'>({d['weight']})</span></td>" if show_sec else ""
            is_re = str(d['popup_type']).startswith('RE-ANCHOR') or str(d['popup_type']).startswith('SWING-RE')
            iter_td = f"<td align='center' style='{base_td}background-color:#e0f2fe;color:#075985;font-weight:700;'>{html.escape(d['first_iter'])}</td><td align='center' style='{base_td}background-color:{'#fef3c7' if is_re else '#dcfce7'};color:{'#92400e' if is_re else '#166534'};font-weight:700;'>{html.escape(d['popup_type'])}</td>" if show_first_iter else ""
            r_bg = "#dcfce7" if d["rank_r_pos"] else "#fee2e2"
            r_fg = "#166534" if d["rank_r_pos"] else "#991b1b"
            st_bg = ("#dcfce7" if basket_dir == "BULL" else "#fee2e2") if d["is_expanding"] else "#fef3c7"
            st_fg = ("#166534" if basket_dir == "BULL" else "#991b1b") if d["is_expanding"] else "#92400e"

            tr_list.append(
                f"<tr {click_attr} data-sector='{sec_attr}' style='background-color:{row_bg};'>"
                f"<td style='{base_td}color:#0f172a;font-weight:700;'>{html.escape(d['symbol'])}</td>"
                f"<td align='center' style='{base_td}background-color:#f0f9ff;color:#0284c7;font-weight:700;'>{html.escape(d['rr'])}</td>"
                f"<td align='right' style='{base_td}color:#0f172a;font-weight:700;'>{d['ltp']}</td>"
                f"<td align='right' style='{base_td}background-color:#f0fdf4;color:#15803d;font-weight:700;'>{tgt_badge}{html.escape(d['target'])}</td>"
                f"<td align='right' style='{base_td}background-color:#fef2f2;color:#b91c1c;font-weight:700;'>{html.escape(d['sl'])}</td>"
                f"{sec_td}{iter_td}"
                f"<td align='center' style='{base_td}color:#334155;'>{d['seen']}</td>"
                f"<td align='center' style='{base_td}color:#0f172a;font-weight:700;'>{html.escape(d['anchor'])}</td>"
                f"<td align='right' style='{base_td}color:#1e293b;font-weight:600;'>{d['anchor_val']}</td>"
                f"<td align='right' style='{base_td}background-color:{r_bg};color:{r_fg};font-weight:700;'>{html.escape(d['r_str'])}</td>"
                f"<td align='center' style='{base_td}color:#1e293b;font-weight:600;'>{d['atr_grid']}</td>"
                f"<td align='center' style='{base_td}color:#1e293b;font-weight:600;'>{d['bb_grid']}</td>"
                f"<td align='center' style='{base_td}background-color:{st_bg};color:{st_fg};font-weight:700;'>{html.escape(d['state'])}</td>"
                f"</tr>"
            )
            if d["prior"]:
                tr_list.append(
                    f"<tr data-sector='{sec_attr}' style='background-color:{row_bg};'>"
                    f"<td colspan='{12 + (1 if show_sec else 0) + (2 if show_first_iter else 0)}' style='padding:3px 10px 6px 24px;border:1px solid #cbd5e1;background-color:{row_bg};color:#64748b;font-size:11px;font-style:italic;'>↳ {html.escape(d['prior'])}</td></tr>"
                )
        return (
            f"<table style='width:100%;border-collapse:collapse;margin-bottom:16px;background-color:#ffffff;border:1px solid #cbd5e1;font-family:Consolas,monospace;font-size:12px;'>"
            f"<thead><tr>"
            f"<th style='{th_style}' align='left'>Script</th><th style='{th_style}' align='center'>R:R</th><th style='{th_style}' align='right'>LTP/Exit</th>"
            f"<th style='{th_style}' align='right'>Target</th><th style='{th_style}' align='right'>SL</th>"
            f"{sec_th}{iter_th}"
            f"<th style='{th_style}' align='center'>Seen</th><th style='{th_style}' align='center'>Anchor</th><th style='{th_style}' align='right'>Anch Val</th>"
            f"<th style='{th_style}' align='right'>R-Mult (Bank)</th><th style='{th_style}' align='center'>ATR: 1X 2X 3X 5X</th>"
            f"<th style='{th_style}' align='center'>BB: 1X 2X 3X 5X</th><th style='{th_style}' align='center'>State</th>"
            f"</tr></thead><tbody>{''.join(tr_list)}</tbody></table>"
        )

    def _sec_banner(title, is_bull=True):
        bg, border, left_b, fg = ("#ecfdf5", "#a7f3d0", "#16a34a", "#065f46") if is_bull else ("#fef2f2", "#fecaca", "#dc2626", "#991b1b")
        return f"<div style='background-color:{bg};border:1px solid {border};border-left:5px solid {left_b};color:{fg};padding:8px 12px;font-weight:700;font-size:13px;border-radius:4px;margin:14px 0 8px 0;'>{html.escape(title)}</div>"

    use_pure = bool(EMAIL_ONLY_PURE_1ST_TIME_CONTENT)
    email_stk_b = new_stk_b if use_pure else win_stk_b
    email_stk_r = new_stk_r if use_pure else win_stk_r
    email_idx_b = new_idx_b if use_pure else win_idx_b
    email_idx_r = new_idx_r if use_pure else win_idx_r
    scope_lbl = _pure_label().upper() if use_pure else "ALL WINDOW"
    sec_tag = "PURE FIRST-TIME" if use_pure else "ALL WINDOW"

    light_css = "body{background-color:#f1f5f9;color:#0f172a;font-family:'Segoe UI',Roboto,Helvetica,Arial,sans-serif;font-size:13px;margin:0;padding:16px;}.pill{background-color:#ffffff;color:#1e293b;border:1px solid #94a3b8;padding:5px 10px;margin:2px;border-radius:4px;cursor:pointer;font-weight:600;}.pill:hover,.pill.active{background-color:#2563eb;color:#ffffff;border-color:#1d4ed8;}.clr{background-color:#dc2626;color:#ffffff;border:none;padding:6px 14px;border-radius:4px;font-weight:700;cursor:pointer;}"
    nb_ce, nb_pe = _ce_pe_counts(email_stk_b)
    nr_ce, nr_pe = _ce_pe_counts(email_stk_r)

    header_card_html = (
        f"<div style='background-color:#ffffff;border:1px solid #cbd5e1;border-left:6px solid #2563eb;border-radius:6px;padding:14px 18px;margin-bottom:16px;'>"
        f"<div style='font-size:17px;font-weight:800;color:#0f172a;margin-bottom:6px;'>📡 SYSTEM 3.0 [{html.escape(mode)}] — {html.escape(scope_lbl)} POPUPS ({w_start_s}..{w_end_s})</div>"
        f"<div style='color:#334155;font-size:13px;margin-bottom:4px;'>"
        f"<b>Target Snapshot:</b> {cutoff_dt.strftime('%Y-%m-%d %H:%M')} &nbsp;|&nbsp; <b>Anchor Window:</b> <span style='background-color:#e0f2fe;color:#0369a1;padding:2px 6px;border-radius:4px;font-weight:700;'>{('Last ' + str(int(SWING_FRESH_SESSIONS)) + ' Sessions') if _is_swing_mode() else (w_start_s + ' to ' + w_end_s)}</span> &nbsp;|&nbsp; <b>Max Gap:</b> {('off' if (_is_swing_mode() or PURE_ANCHOR_IGNORE_GAP or not ENABLE_TIMESTAMP_GAP_CHECK or MAX_TIMESTAMP_GAP_MINUTES <= 0) else str(MAX_TIMESTAMP_GAP_MINUTES) + 'm')}</div>"
        f"<div style='color:#64748b;font-size:11px;margin-bottom:4px;'>{html.escape(_report_legend_text())}</div>"
        f"<div style='color:#0369a1;font-size:11px;margin-bottom:4px;'>{html.escape(_pure_note_text())}</div>"
        f"<div style='color:#334155;font-size:12px;margin-top:4px;'>"
        f"<b>Sectoral Indices:</b> <span style='color:#15803d;font-weight:700;'>▲ {len(email_idx_b)} Bull</span> &nbsp;/&nbsp; <span style='color:#b91c1c;font-weight:700;'>▼ {len(email_idx_r)} Bear</span> &nbsp;|&nbsp; "
        f"<b>Stocks:</b> <span style='color:#15803d;font-weight:700;'>▲ {len(email_stk_b)} Bull [CE({nb_ce}) PE({nb_pe})]</span> &nbsp;/&nbsp; <span style='color:#b91c1c;font-weight:700;'>▼ {len(email_stk_r)} Bear [CE({nr_ce}) PE({nr_pe})]</span>"
        f"</div></div>"
    )

    idx_section_html = ""
    if email_idx_b or email_idx_r:
        idx_section_html = (
            f"<div style='font-size:15px;font-weight:800;color:#0f172a;margin:18px 0 6px 0;'>🏛️ SECTORAL INDICES — {sec_tag} POPUPS</div>"
            + (f"{_sec_banner(f'🏛️🆕🔥 {sec_tag} SECTORAL INDEX POPUPS — BULL ({len(email_idx_b)})', True)}"
               f"{_table_active_html(email_idx_b, 'BULL', show_sec=False, show_first_iter=True)}" if email_idx_b else "")
            + (f"{_sec_banner(f'🏛️🆕🩸 {sec_tag} SECTORAL INDEX POPUPS — BEAR ({len(email_idx_r)})', False)}"
               f"{_table_active_html(email_idx_r, 'BEAR', show_sec=False, show_first_iter=True)}" if email_idx_r else "")
        )

    stk_section_html = ""
    if email_stk_b or email_stk_r:
        stk_section_html = (
            f"<div style='font-size:15px;font-weight:800;color:#0f172a;margin:18px 0 6px 0;'>📈 STOCKS — {sec_tag} POPUPS</div>"
            + (f"{_sec_banner(f'🆕🔥 {sec_tag} STOCK POPUPS — BULL {_format_count_with_ce_pe(email_stk_b)}', True)}"
               f"{_table_active_html(email_stk_b, 'BULL', show_sec=True, show_first_iter=True)}" if email_stk_b else "")
            + (f"{_sec_banner(f'🆕🩸 {sec_tag} STOCK POPUPS — BEAR {_format_count_with_ce_pe(email_stk_r)}', False)}"
               f"{_table_active_html(email_stk_r, 'BEAR', show_sec=True, show_first_iter=True)}" if email_stk_r else "")
        )

    main_tables_html = f"{idx_section_html}{stk_section_html}"

    email_inline_html = (
        f"<!DOCTYPE html><html><head><meta charset='utf-8'><style>{light_css}</style></head>"
        f"<body style='background-color:#f1f5f9;color:#0f172a;font-family:Segoe UI,sans-serif;font-size:13px;margin:0;padding:16px;'>"
        f"<div style='max-width:1280px;margin:0 auto;'>"
        f"{header_card_html}{main_tables_html}"
        + f"</div></body></html>"
    )

    active_popup_sectors = sorted({r.get('Sector', DEFAULT_UNMAPPED_SECTOR) for r in (email_stk_b + email_stk_r)})
    pills_js_html = "".join(f"<button class='pill' data-sec='{html.escape(s, quote=True)}' onclick=\"filterBySector({_js_onclick_arg(s)})\">{html.escape(s)}</button>" for s in active_popup_sectors)

    attachment_js_html = (
        f"<!DOCTYPE html><html><head><meta charset='utf-8'><title>System 3.0 Popups</title><style>{light_css}</style></head>"
        f"<body style='background-color:#f1f5f9;color:#0f172a;font-family:Segoe UI,sans-serif;font-size:13px;margin:0;padding:16px;'>"
        f"<div style='max-width:1280px;margin:0 auto;'>"
        f"<div style='position:sticky;top:0;background-color:#ffffff;padding:12px 16px;border:1px solid #cbd5e1;border-radius:6px;margin-bottom:14px;box-shadow:0 2px 4px rgba(0,0,0,0.05);'>"
        f"<div style='display:flex;justify-content:space-between;align-items:center;margin-bottom:8px;'>"
        f"<b style='font-size:15px;color:#0f172a;'>📡 SYSTEM 3.0 [{html.escape(mode)}] @ {cutoff_dt.strftime('%Y-%m-%d %H:%M')}</b>"
        f"<div><span id='statusBadge' style='color:#2563eb;margin-right:10px;font-weight:700;'>SHOWING ALL</span><button class='clr' onclick='clearFilter()'>SHOW ALL</button></div></div>"
        f"<div>{pills_js_html}</div></div>"
        f"{header_card_html}<div id='popupContainer'>{main_tables_html}</div>"
        f"<script>"
        f"function filterBySector(sec){{document.querySelectorAll('#popupContainer tbody tr').forEach(r=>{{r.style.display=(r.getAttribute('data-sector')===sec)?'':'none';}});document.querySelectorAll('.pill').forEach(p=>p.classList.toggle('active',p.getAttribute('data-sec')===sec));document.getElementById('statusBadge').textContent='FILTERED BY: '+sec;}}"
        f"function clearFilter(){{document.querySelectorAll('#popupContainer tbody tr').forEach(r=>{{r.style.display='';}});document.querySelectorAll('.pill').forEach(p=>p.classList.remove('active'));document.getElementById('statusBadge').textContent='SHOWING ALL';}}"
        f"</script></div></body></html>"
    )

    return email_inline_html, attachment_js_html

# ==============================================================================
# 4B. DEDICATED STILL-HOLDING GOOD EMAIL & HTML ATTACHMENT BUILDER
# ==============================================================================
def build_still_holding_plain_text(
    mode, cutoff_dt, checkpoints, idx_hold_bulls, idx_hold_bears,
    stock_hold_bulls, stock_hold_bears
):
    hb_ce, hb_pe = _ce_pe_counts(stock_hold_bulls)
    hr_ce, hr_pe = _ce_pe_counts(stock_hold_bears)

    all_shown = idx_hold_bulls + idx_hold_bears + stock_hold_bulls + stock_hold_bears
    sym_w = max(MIN_SYMBOL_WIDTH, max((len(r.get('Symbol', '')) for r in all_shown), default=MIN_SYMBOL_WIDTH))
    anc_w = max(12 if _is_swing_mode() else 7, max((len(str(r.get('ActiveAnchor', '-'))) + 1 for r in all_shown), default=7))
    sec_w = max(
        SECTOR_COL_WIDTH,
        max(
            (
                len(f"{r.get('Sector', DEFAULT_UNMAPPED_SECTOR)} ({r.get('SectorWeight', 0.0):.1f}%)")
                for r in all_shown
            ),
            default=SECTOR_COL_WIDTH,
        ),
    )

    lines = []
    lines.append(f"=== SYSTEM 3.0 [{mode}] — SETUPS STILL HOLDING GOOD REPORT ===")
    lines.append(f"Snapshot: {cutoff_dt.strftime('%Y-%m-%d %H:%M')} | Active Holding Setups (Alive & Not Stopped Out)")
    lines.append(_report_legend_text())
    lines.append(
        f"Sectoral Indices Still Holding: ▲ {len(idx_hold_bulls)} Bull / ▼ {len(idx_hold_bears)} Bear | "
        f"Stocks Still Holding: ▲ {len(stock_hold_bulls)} Bull [CE({hb_ce}) PE({hb_pe})] / ▼ {len(stock_hold_bears)} Bear [CE({hr_ce}) PE({hr_pe})]\n"
    )

    def _fmt_holding_block(title, data_list, show_sector=True, include_ce_pe_counts=True):
        count_suffix = _format_count_with_ce_pe(data_list) if include_ce_pe_counts else f"({len(data_list)})"
        if not data_list:
            lines.append(f"--- {title} {count_suffix} ---")
            lines.append("None\n")
            return
        lines.append(f"--- {title} {count_suffix} ---")
        grid_hdr = " ".join(f"{m}X" for m in GRID_MULTIPLIERS)
        h_atr = f"ATR: {grid_hdr}"
        h_bb = f"BB: {grid_hdr}"
        sec_hdr = f"{'Sector (MaxWt%)':<{sec_w}} | " if show_sector else ""
        iter_hdr = f"{'1stIter':^7} {'PopupType':^13} | "
        hdr = (
            f"{'Script':<{sym_w}} {'R:R':^7} {'LTP/Exit':>9} {'Target':>12} {'SL':>9} | "
            f"{sec_hdr}{iter_hdr}{'Seen':^5} {'Anchor':^{anc_w}} {'Anch Val':>9} {'R-Mult (Bank)':>15} | "
            f"{h_atr:^16} | {h_bb:^15} | {'State':^13}"
        )
        lines.append(hdr)
        lines.append("-" * len(hdr))
        basket_dir = "BULL" if "BULL" in title else "BEAR"
        default_g = {m: 0 for m in GRID_MULTIPLIERS}
        for r in data_list:
            live_r, banked_r = _r_multiple(r, include_banked=False), float(r.get('BankedR', 0.0) or 0.0)
            has_bank = r.get('PriorStops', 0) > 0 and abs(banked_r) > BANKED_R_DISPLAY_EPSILON
            r_txt = f"{live_r:>+5.2f}R ({banked_r:>+4.1f}R)" if has_bank else f"{live_r:>+5.2f}R"
            r_mult_str = f"{r_txt:>15}"
            anchor_disp = f"{r.get('ActiveAnchor', '-')}{'*' if has_bank else ''}"
            ag, bg = r.get('ATR_Grid', default_g), r.get('BB_Grid', default_g)
            a_str = "  ".join(str(ag.get(k, 0)) for k in GRID_MULTIPLIERS)
            b_str = "  ".join(str(bg.get(k, 0)) for k in GRID_MULTIPLIERS)
            shown_state = _state_display(r)
            if "[ BB > KC ]" in r['State']: shown_state = f"[ BB > KC ] {'▲' if basket_dir == 'BULL' else '▼'}"
            ltp_val = f"{float(r['KilledPrice']):>8.2f}x" if ("[STOP" in shown_state and float(r.get('KilledPrice', 0.0) or 0.0) > 0) else f"{float(r.get('LTP', 0.0) or 0.0):>9.2f}"
            tgt_cell = _target_cell_text(r)
            sl_val = f"{float(r.get('SLPrice', 0.0) or 0.0):>9.2f}"
            rr_str = r.get('RR_Str') or compute_risk_reward_ratio(r)[1]
            sec_str = f"{r.get('Sector', DEFAULT_UNMAPPED_SECTOR)} ({r.get('SectorWeight', 0.0):.1f}%)"
            sec_cell = f"{_clip(sec_str, sec_w, EMAIL_CLIP_SYMBOLS):<{sec_w}} | " if show_sector else ""

            cp_val = r.get('CheckpointTime')
            seen_val = cp_val.strftime('%H:%M') if hasattr(cp_val, 'strftime') else str(cp_val or '-')
            anch_val_str = f"{float(r.get('AnchorPrice', 0.0) or 0.0):>9.2f}"

            first_iter_val = str(r.get('FirstPopupIter', seen_val))
            popup_type_val = str(r.get('PopupType', '1ST-TIME'))
            iter_cell = f"{first_iter_val:^7} {popup_type_val:^13} | "
            lines.append(
                f"{_clip(r['Symbol'], sym_w, EMAIL_CLIP_SYMBOLS):<{sym_w}} {rr_str:^7} {ltp_val} {tgt_cell} {sl_val} | "
                f"{sec_cell}{iter_cell}{seen_val:^5} {anchor_disp:^{anc_w}} {anch_val_str} "
                f"{r_mult_str} | {a_str:^16} | {b_str:^15} | {shown_state:^13}"
            )
            _pa = _prior_attempts_text(r)
            if _pa:
                lines.append(f"   ↳ {_pa}")
        lines.append("")

    if idx_hold_bulls or idx_hold_bears:
        _fmt_holding_block("SECTORAL INDICES — STILL HOLDING BULL", idx_hold_bulls, show_sector=False, include_ce_pe_counts=False)
        _fmt_holding_block("SECTORAL INDICES — STILL HOLDING BEAR", idx_hold_bears, show_sector=False, include_ce_pe_counts=False)

    _fmt_holding_block("STOCKS — STILL HOLDING BULL", stock_hold_bulls, show_sector=True, include_ce_pe_counts=True)
    _fmt_holding_block("STOCKS — STILL HOLDING BEAR", stock_hold_bears, show_sector=True, include_ce_pe_counts=True)
    return "\n".join(lines)

def build_still_holding_email_and_attachment_html(
    mode, cutoff_dt, checkpoints, idx_hold_bulls, idx_hold_bears,
    stock_hold_bulls, stock_hold_bears
):
    hb_ce, hb_pe = _ce_pe_counts(stock_hold_bulls)
    hr_ce, hr_pe = _ce_pe_counts(stock_hold_bears)

    def _ser_holding(r, basket_dir):
        if float(r.get('TargetPrice', 0.0) or 0.0) <= 0 or float(r.get('SLPrice', 0.0) or 0.0) <= 0:
            r['TargetPrice'], r['SLPrice'] = _compute_target_sl(r)
        live_r = _r_multiple(r, include_banked=False)
        banked_r = float(r.get('BankedR', 0.0) or 0.0)
        has_bank = r.get('PriorStops', 0) > 0 and abs(banked_r) > BANKED_R_DISPLAY_EPSILON
        default_g = {m: 0 for m in GRID_MULTIPLIERS}
        ag, bg = r.get('ATR_Grid', default_g), r.get('BB_Grid', default_g)
        st = _state_display(r)
        if "[ BB > KC ]" in r['State']: st = f"[ BB > KC ] {'▲' if basket_dir == 'BULL' else '▼'}"

        cp_val = r.get('CheckpointTime')
        seen_s = cp_val.strftime('%H:%M') if hasattr(cp_val, 'strftime') else str(cp_val or '-')
        shown_ltp = f"{float(r['KilledPrice']):.2f}x" if ("[STOP" in st and float(r.get('KilledPrice', 0.0) or 0.0) > 0) else f"{float(r.get('LTP', 0.0) or 0.0):.2f}"

        return {
            "symbol": r['Symbol'],
            "rr": r.get('RR_Str') or compute_risk_reward_ratio(r)[1],
            "sector": r.get('Sector', DEFAULT_UNMAPPED_SECTOR),
            "weight": f"{float(r.get('SectorWeight', 0.0)):.1f}%",
            "first_iter": str(r.get('FirstPopupIter') or seen_s),
            "popup_type": r.get('PopupType', '1ST-TIME'),
            "ltp": shown_ltp,
            "tgt_tag": _target_tag(r),
            "prior": _prior_attempts_text(r),
            "target": f"{float(r.get('TargetPrice', 0.0) or 0.0):.2f}" if float(r.get('TargetPrice', 0.0) or 0.0) > 0 else "-",
            "sl": f"{float(r.get('SLPrice', 0.0) or 0.0):.2f}" if float(r.get('SLPrice', 0.0) or 0.0) > 0 else "-",
            "seen": seen_s,
            "anchor": f"{r.get('ActiveAnchor', '-')}{'*' if has_bank else ''}",
            "anchor_val": f"{float(r.get('AnchorPrice', 0.0) or 0.0):.2f}",
            "r_str": f"{live_r:+.2f}R ({banked_r:+.1f}R)" if has_bank else f"{live_r:+.2f}R",
            "rank_r_pos": _rank_r_score(r) >= 0,
            "atr_grid": "  ".join(str(ag.get(k, 0)) for k in GRID_MULTIPLIERS),
            "bb_grid": "  ".join(str(bg.get(k, 0)) for k in GRID_MULTIPLIERS),
            "state": st,
            "is_expanding": "[ BB > KC ]" in r['State'],
        }

    def _table_holding_html(rows_list, basket_dir, show_sec=True, js_clickable_index=False):
        if not rows_list:
            return "<div style='background:#fff;border:1px solid #cbd5e1;border-radius:6px;padding:10px 14px;color:#64748b;font-style:italic;margin-bottom:14px;'>No setups currently holding in this category.</div>"
        th_s = "background:#1e293b;color:#fff;padding:6px 7px;border:1px solid #334155;font-size:11px;white-space:nowrap;"
        td_b = "padding:5px 6px;border:1px solid #cbd5e1;font-size:12px;"
        sec_th = f"<th align='left' style='{th_s}'>Sector (MaxWt%)</th>" if show_sec else ""
        iter_th = f"<th style='{th_s}'>1stIter</th><th style='{th_s}'>PopupType</th>"

        tr_list = []
        for idx_i, r in enumerate(rows_list):
            d = _ser_holding(r, basket_dir)
            tgt_badge = f"<span style='font-size:10px;background:#fef3c7;color:#92400e;border-radius:3px;padding:1px 3px;margin-right:2px;'>{d['tgt_tag']}</span> " if d["tgt_tag"] else ""
            row_bg = "background:#f8fafc;" if idx_i % 2 else "background:#fff;"
            raw_sec = d["symbol"] if not show_sec else d["sector"]
            sec_attr = html.escape(raw_sec, quote=True)
            click_attr = f" onclick=\"filterBySector({_js_onclick_arg(raw_sec)})\"" if js_clickable_index else ""
            sec_td = f"<td style='{td_b}color:#1e40af;font-weight:600;'>{html.escape(d['sector'])} ({d['weight']})</td>" if show_sec else ""
            _pt_s = str(d.get('popup_type', '1ST-TIME'))
            is_re = _pt_s.startswith('RE-ANCHOR') or _pt_s.startswith('SWING-RE')
            pt_bg, pt_fg = ("#fef3c7", "#92400e") if is_re else (("#dbeafe", "#1e40af") if _pt_s.startswith("HOLD-") else ("#dcfce7", "#166534"))
            r_bg, r_fg = ("#dcfce7", "#166534") if d["rank_r_pos"] else ("#fee2e2", "#991b1b")
            st_bg, st_fg = (("#dcfce7", "#166534") if basket_dir == "BULL" else ("#fee2e2", "#991b1b")) if d["is_expanding"] else ("#fef3c7", "#92400e")

            tr_list.append(
                f"<tr style='{row_bg}'{click_attr} data-sector='{sec_attr}'>"
                f"<td style='{td_b}font-weight:700;color:#0f172a;'>{html.escape(d['symbol'])}</td>"
                f"<td style='{td_b}background:#f0f9ff;color:#0284c7;font-weight:700;text-align:center;white-space:nowrap;'>{html.escape(d['rr'])}</td>"
                f"<td style='{td_b}font-weight:700;text-align:right;white-space:nowrap;'>{d['ltp']}</td>"
                f"<td style='{td_b}background:#f0fdf4;color:#15803d;font-weight:700;text-align:right;white-space:nowrap;'>{tgt_badge}{html.escape(d['target'])}</td>"
                f"<td style='{td_b}background:#fef2f2;color:#b91c1c;font-weight:700;text-align:right;white-space:nowrap;'>{html.escape(d['sl'])}</td>"
                f"{sec_td}"
                f"<td style='{td_b}background:#e0f2fe;color:#075985;font-weight:700;text-align:center;white-space:nowrap;'>{html.escape(d.get('first_iter', d['seen']))}</td>"
                f"<td style='{td_b}background:{pt_bg};color:{pt_fg};font-weight:700;text-align:center;white-space:nowrap;'>{html.escape(d.get('popup_type', '1ST-TIME'))}</td>"
                f"<td style='{td_b}text-align:center;white-space:nowrap;'>{d['seen']}</td>"
                f"<td style='{td_b}font-weight:700;text-align:center;white-space:nowrap;'>{html.escape(d['anchor'])}</td>"
                f"<td style='{td_b}text-align:right;white-space:nowrap;'>{d['anchor_val']}</td>"
                f"<td style='{td_b}background:{r_bg};color:{r_fg};font-weight:700;text-align:right;white-space:nowrap;'>{html.escape(d['r_str'])}</td>"
                f"<td style='{td_b}text-align:center;white-space:nowrap;'>{d['atr_grid']}</td>"
                f"<td style='{td_b}text-align:center;white-space:nowrap;'>{d['bb_grid']}</td>"
                f"<td style='{td_b}background:{st_bg};color:{st_fg};font-weight:700;text-align:center;white-space:nowrap;'>{html.escape(d['state'])}</td>"
                f"</tr>"
            )
            if d["prior"]:
                tr_list.append(
                    f"<tr style='{row_bg}' data-sector='{sec_attr}'><td colspan='{14 + (1 if show_sec else 0)}' style='padding:3px 8px 5px 18px;border:1px solid #cbd5e1;color:#64748b;font-size:11px;font-style:italic;'>↳ {html.escape(d['prior'])}</td></tr>"
                )
        return (
            f"<table width='100%' cellpadding='0' cellspacing='0' style='width:100%;border-collapse:collapse;margin-bottom:14px;background:#fff;border:1px solid #cbd5e1;font-family:Consolas,monospace;font-size:12px;'><thead><tr>"
            f"<th align='left' style='{th_s}'>Script</th><th style='{th_s}'>R:R</th><th align='right' style='{th_s}'>LTP/Exit</th>"
            f"<th align='right' style='{th_s}'>Target</th><th align='right' style='{th_s}'>SL</th>"
            f"{sec_th}{iter_th}"
            f"<th style='{th_s}'>Seen</th><th style='{th_s}'>Anchor</th><th align='right' style='{th_s}'>Anch Val</th>"
            f"<th align='right' style='{th_s}'>R-Mult (Bank)</th><th style='{th_s}'>ATR: 1X 2X 3X 5X</th>"
            f"<th style='{th_s}'>BB: 1X 2X 3X 5X</th><th style='{th_s}'>State</th>"
            f"</tr></thead><tbody>{''.join(tr_list)}</tbody></table>"
        )

    def _sec_banner(title, is_bull=True):
        bg, border, left_b, fg = ("#ecfdf5", "#a7f3d0", "#16a34a", "#065f46") if is_bull else ("#fef2f2", "#fecaca", "#dc2626", "#991b1b")
        return f"<div style='background-color:{bg};border:1px solid {border};border-left:5px solid {left_b};color:{fg};padding:8px 12px;font-weight:700;font-size:13px;border-radius:4px;margin:14px 0 8px 0;'>{html.escape(title)}</div>"

    light_css = "body{background:#f1f5f9;color:#0f172a;font-family:'Segoe UI',Roboto,Helvetica,Arial,sans-serif;font-size:13px;margin:0;padding:14px;}.pill{background:#fff;color:#1e293b;border:1px solid #94a3b8;padding:5px 10px;margin:2px;border-radius:4px;cursor:pointer;font-weight:600;}.pill:hover,.pill.active{background:#2563eb;color:#fff;border-color:#1d4ed8;}.clr{background:#dc2626;color:#fff;border:none;padding:6px 14px;border-radius:4px;font-weight:700;cursor:pointer;}"

    header_card_html = (
        f"<div style='background-color:#ffffff;border:1px solid #cbd5e1;border-left:6px solid #16a34a;border-radius:6px;padding:14px 18px;margin-bottom:16px;'>"
        f"<div style='font-size:17px;font-weight:800;color:#0f172a;margin-bottom:6px;'>🛡️ SYSTEM 3.0 [{html.escape(mode)}] — SETUPS STILL HOLDING GOOD</div>"
        f"<div style='color:#334155;font-size:13px;margin-bottom:4px;'>"
        f"<b>Target Snapshot:</b> {cutoff_dt.strftime('%Y-%m-%d %H:%M')} &nbsp;|&nbsp; <b>Status:</b> <span style='background-color:#dcfce7;color:#166534;padding:2px 6px;border-radius:4px;font-weight:700;'>Active Survived Setups</span></div>"
        f"<div style='color:#64748b;font-size:11px;margin-bottom:4px;'>{html.escape(_report_legend_text())}</div>"
        f"<div style='color:#334155;font-size:12px;margin-top:4px;'>"
        f"<b>Sectoral Indices:</b> <span style='color:#15803d;font-weight:700;'>▲ {len(idx_hold_bulls)} Bull</span> &nbsp;/&nbsp; <span style='color:#b91c1c;font-weight:700;'>▼ {len(idx_hold_bears)} Bear</span> &nbsp;|&nbsp; "
        f"<b>Stocks:</b> <span style='color:#15803d;font-weight:700;'>▲ {len(stock_hold_bulls)} Bull [CE({hb_ce}) PE({hb_pe})]</span> &nbsp;/&nbsp; <span style='color:#b91c1c;font-weight:700;'>▼ {len(stock_hold_bears)} Bear [CE({hr_ce}) PE({hr_pe})]</span>"
        f"</div></div>"
    )

    idx_section_html = ""
    if idx_hold_bulls or idx_hold_bears:
        idx_section_html = (
            f"<div style='font-size:15px;font-weight:800;color:#0f172a;margin:18px 0 6px 0;'>🏛️ SECTORAL INDICES — STILL HOLDING</div>"
            + (f"{_sec_banner(f'🏛️🔥 SECTORAL INDICES — STILL HOLDING BULL ({len(idx_hold_bulls)})', True)}"
               f"{_table_holding_html(idx_hold_bulls, 'BULL', show_sec=False)}" if idx_hold_bulls else "")
            + (f"{_sec_banner(f'🏛️🩸 SECTORAL INDICES — STILL HOLDING BEAR ({len(idx_hold_bears)})', False)}"
               f"{_table_holding_html(idx_hold_bears, 'BEAR', show_sec=False)}" if idx_hold_bears else "")
        )

    stk_section_html = ""
    if stock_hold_bulls or stock_hold_bears:
        stk_section_html = (
            f"<div style='font-size:15px;font-weight:800;color:#0f172a;margin:18px 0 6px 0;'>📈 STOCKS — STILL HOLDING</div>"
            + (f"{_sec_banner(f'🔥 STOCKS — STILL HOLDING BULL {_format_count_with_ce_pe(stock_hold_bulls)}', True)}"
               f"{_table_holding_html(stock_hold_bulls, 'BULL', show_sec=True)}" if stock_hold_bulls else "")
            + (f"{_sec_banner(f'🩸 STOCKS — STILL HOLDING BEAR {_format_count_with_ce_pe(stock_hold_bears)}', False)}"
               f"{_table_holding_html(stock_hold_bears, 'BEAR', show_sec=True)}" if stock_hold_bears else "")
        )

    main_tables_html = f"{idx_section_html}{stk_section_html}"

    email_inline_html = (
        f"<!DOCTYPE html><html><head><meta charset='utf-8'><style>{light_css}</style></head>"
        f"<body style='background-color:#f1f5f9;color:#0f172a;font-family:Segoe UI,sans-serif;font-size:13px;margin:0;padding:16px;'>"
        f"<div style='max-width:1280px;margin:0 auto;'>"
        f"{header_card_html}{main_tables_html}"
        + f"</div></body></html>"
    )

    active_holding_sectors = sorted({r.get('Sector', DEFAULT_UNMAPPED_SECTOR) for r in (stock_hold_bulls + stock_hold_bears)})
    pills_js_html = "".join(f"<button class='pill' data-sec='{html.escape(s, quote=True)}' onclick=\"filterBySector({_js_onclick_arg(s)})\">{html.escape(s)}</button>" for s in active_holding_sectors)

    attachment_js_html = (
        f"<!DOCTYPE html><html><head><meta charset='utf-8'><title>System 3.0 Still Holding Setups</title><style>{light_css}</style></head>"
        f"<body style='background-color:#f1f5f9;color:#0f172a;font-family:Segoe UI,sans-serif;font-size:13px;margin:0;padding:16px;'>"
        f"<div style='max-width:1280px;margin:0 auto;'>"
        f"<div style='position:sticky;top:0;background-color:#ffffff;padding:12px 16px;border:1px solid #cbd5e1;border-radius:6px;margin-bottom:14px;box-shadow:0 2px 4px rgba(0,0,0,0.05);'>"
        f"<div style='display:flex;justify-content:space-between;align-items:center;margin-bottom:8px;'>"
        f"<b style='font-size:15px;color:#0f172a;'>🛡️ SYSTEM 3.0 STILL HOLDING [{html.escape(mode)}] @ {cutoff_dt.strftime('%Y-%m-%d %H:%M')}</b>"
        f"<div><span id='statusBadge' style='color:#16a34a;margin-right:10px;font-weight:700;'>SHOWING ALL</span><button class='clr' onclick='clearFilter()'>SHOW ALL</button></div></div>"
        f"<div>{pills_js_html}</div></div>"
        f"{header_card_html}<div id='holdingContainer'>{main_tables_html}</div>"
        f"<script>"
        f"function filterBySector(sec){{document.querySelectorAll('#holdingContainer tbody tr').forEach(r=>{{r.style.display=(r.getAttribute('data-sector')===sec)?'':'none';}});document.querySelectorAll('.pill').forEach(p=>p.classList.toggle('active',p.getAttribute('data-sec')===sec));document.getElementById('statusBadge').textContent='FILTERED BY: '+sec;}}"
        f"function clearFilter(){{document.querySelectorAll('#holdingContainer tbody tr').forEach(r=>{{r.style.display='';}});document.querySelectorAll('.pill').forEach(p=>p.classList.remove('active'));document.getElementById('statusBadge').textContent='SHOWING ALL';}}"
        f"</script></div></body></html>"
    )

    return email_inline_html, attachment_js_html

# ------------------------------------------------------------------------------
# EMAIL TRANSMISSION ENGINE (USES GITHUB REPOSITORY SECRETS)
# ------------------------------------------------------------------------------
_EMAIL_SEND_LOCK = threading.Lock()
_EMAIL_SENT_TAGS = set()

def _send_email_message(subject, plain_body, html_body=None, attachment_html=None, tag="email", attachment_filename=None):
    with _EMAIL_SEND_LOCK:
        if tag in _EMAIL_SENT_TAGS: return False
        _EMAIL_SENT_TAGS.add(tag)
    sender = os.environ.get(ENV_SENDER_EMAIL)
    password = os.environ.get(ENV_SENDER_PASSWORD)
    recipient = os.environ.get(ENV_RECIPIENT_EMAIL)
    smtp_server = os.environ.get(ENV_SMTP_SERVER) or DEFAULT_SMTP_SERVER
    smtp_port = os.environ.get(ENV_SMTP_PORT) or DEFAULT_SMTP_PORT

    if not sender or not password or not recipient: return False
    try: port = int(smtp_port)
    except ValueError: return False

    msg = EmailMessage()
    msg['Subject'] = subject
    msg['From'] = sender
    msg['To'] = recipient
    msg.set_content(plain_body)
    if html_body: msg.add_alternative(html_body, subtype='html')
    att_name = attachment_filename or EMAIL_ATTACHMENT_FILENAME
    if attachment_html:
        msg.add_attachment(attachment_html.encode('utf-8'), maintype='text', subtype='html', filename=att_name)

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
        print(f"   {COLOR_GREEN_FG}📧 [{tag}] Email sent to {recipient} (Subject: {subject}){COLOR_RESET}")
        return True
    except Exception as e:
        print(f"   {COLOR_YELLOW}[{tag}] Failed to send email ({e}){COLOR_RESET}")
        return False

def print_and_alert_tracked_history(tracked_symbol_raw, by_symbol, live_run=True):
    if not tracked_symbol_raw or tracked_symbol_raw == "***": return
    n = _norm_sym(tracked_symbol_raw)
    matched_sym = next((sym for sym in by_symbol.keys() if n in _norm_sym(sym)), None)
    if not matched_sym: return
    rows = sorted(by_symbol[matched_sym], key=lambda r: r['CheckpointTime'])
    history, is_pe = [], _is_pe_symbol(matched_sym)
    for i in range(1, len(rows) + 1):
        subset = rows[:i]
        b, r_, g_list = merge_symbol_rows(subset)
        selected_row = b if (b and b['State'] != "[STOPPED]") else r_ if (r_ and r_['State'] != "[STOPPED]") else (g_list[-1] if g_list else None)
        if selected_row:
            row_copy = dict(selected_row)
            raw_d = row_copy.get('AnchorDir', 'BULL')
            if raw_d in ("BULL", "BEAR"):
                row_copy['PremiumDir'] = raw_d
                if is_pe: row_copy['AnchorDir'] = "BEAR" if raw_d == "BULL" else "BULL"
                row_copy['TargetPrice'], row_copy['SLPrice'] = _compute_target_sl(row_copy)
                row_copy['RR_Val'], row_copy['RR_Str'] = compute_risk_reward_ratio(row_copy)
            row_copy['_EvalTime'] = subset[-1]['CheckpointTime']
            history.append(row_copy)
    if not history: return

    sec_name, sec_wt, _ = resolve_primary_sector(matched_sym)
    print(f"\n{COLOR_BOLD}🔍 INTRADAY HISTORY FOR TRACKED SYMBOL: {matched_sym} [{sec_name} ({sec_wt:.2f}%)]{COLOR_RESET}")
    sym_w = max(MIN_SYMBOL_WIDTH, len(matched_sym))
    anc_w = 12 if _is_swing_mode() else 7
    grid_hdr = " ".join(f"{m}X" for m in GRID_MULTIPLIERS)
    h_atr = f"ATR: {grid_hdr}"
    h_bb = f"BB: {grid_hdr}"
    header_str = (
        f" {COLOR_CYAN}{'Script':<{sym_w}} {'R:R':^7} {'LTP/Exit':>9} {'Target':>12} {'SL':>9} | "
        f"{'Eval':^5} {'Anchor':^{anc_w}} {'Anch Val':>9} {'R-Mult (Bank)':>15} | "
        f"{h_atr:^16} | {h_bb:^15} | {'State':^13}{COLOR_RESET}"
    )
    print(header_str)
    print("-" * len(ANSI_RE.sub("", header_str)))
    for row in history:
        live_r, banked_r = _r_multiple(row, include_banked=False), float(row.get('BankedR', 0.0) or 0.0)
        rank_r = _rank_r_score(row)
        r_color = COLOR_GREEN_FG if rank_r > 0 else COLOR_RED_FG if rank_r < 0 else COLOR_DIM
        has_bank = row.get('PriorStops', 0) > 0 and abs(banked_r) > BANKED_R_DISPLAY_EPSILON
        r_txt = f"{live_r:>+5.2f}R ({banked_r:>+4.1f}R)" if has_bank else f"{live_r:>+5.2f}R"
        r_mult_str = f"{r_txt:>15}"
        a_time = f"{row.get('ActiveAnchor', '-')}{'*' if has_bank else ''}"
        direction = row.get('AnchorDir', 'BULL')
        dir_c = COLOR_GREEN_FG if direction == "BULL" else COLOR_RED_FG
        default_g = {m: 0 for m in GRID_MULTIPLIERS}
        ag, bg = row.get('ATR_Grid', default_g), row.get('BB_Grid', default_g)
        a_str = "  ".join(_format_val(ag.get(k, 0), dir_c) for k in GRID_MULTIPLIERS)
        b_str = "  ".join(_format_val(bg.get(k, 0), dir_c) for k in GRID_MULTIPLIERS)
        a_cell = f"   {a_str}   "
        b_cell = f"  {b_str}   "

        shown_state = _state_display(row)
        if "[ BB > KC ]" in row['State']:
            dir_tag = "▲" if direction == "BULL" else "▼" if direction == "BEAR" else ""
            state_text = f"{COLOR_GREEN_FG if dir_tag == '▲' else COLOR_RED_FG}{f'[ BB > KC ] {dir_tag}'.strip():^13}{COLOR_RESET}"
        else:
            state_text = f"{COLOR_YELLOW if '[ BB < KC ]' in row['State'] else COLOR_DIM}{shown_state:^13}{COLOR_RESET}"

        ltp_val = f"{float(row['KilledPrice']):>8.2f}x" if ("[STOP" in shown_state and float(row.get('KilledPrice', 0.0) or 0.0) > 0) else f"{float(row.get('LTP', 0.0) or 0.0):>9.2f}"
        _tg = f"{COLOR_GREEN_FG}{_target_cell_text(row)}{COLOR_RESET}"
        sl_val = f"{COLOR_RED_FG}{float(row.get('SLPrice', 0.0) or 0.0):>9.2f}{COLOR_RESET}"
        rr_str = row.get('RR_Str') or compute_risk_reward_ratio(row)[1]

        eval_dt = row.get('_EvalTime')
        eval_s = eval_dt.strftime('%H:%M') if hasattr(eval_dt, 'strftime') else str(eval_dt or '-')
        anch_val_str = f"{float(row.get('AnchorPrice', 0.0) or 0.0):>9.2f}"

        print(
            f" {COLOR_BOLD}{_clip(row['Symbol'], sym_w):<{sym_w}}{COLOR_RESET} {COLOR_CYAN}{rr_str:^7}{COLOR_RESET} {ltp_val} {_tg} {sl_val} | "
            f"{eval_s:^5} {a_time:^{anc_w}} {anch_val_str} {r_color}{r_mult_str}{COLOR_RESET} | {a_cell} | {b_cell} | {state_text}"
        )

# ==============================================================================
# MAIN RUNNER
# ==============================================================================
def run_screener(
    mode=TRADING_MODE, days=BACKTRACE_DAYS, target_date_str=None, target_time_str=None,
    track_symbol="", sector_filter="", html_out=HTML_DASHBOARD_PATH, interactive_cli=False,
    lookback_iterations=None, lookback_interval_min=None
):
    global _RUN_MODE
    _RUN_MODE = mode
    _apply_mode_profile(mode)
    swing_mode = _is_swing_mode(mode)
    with _EMAIL_SEND_LOCK:
        _EMAIL_SENT_TAGS.clear()
    t_start = time.time()
    cutoff_dt, explicit = _parse_snapshot_datetime(target_date_str, target_time_str)
    has_explicit_time = bool(target_time_str and str(target_time_str).strip() and str(target_time_str).strip().lower() not in ("live", "now"))
    cutoff_dt, notes = resolve_session(cutoff_dt.replace(second=0, microsecond=0), explicit_time=has_explicit_time)
    if STATS.auth_failed: return 1
    target_dt, today_session = cutoff_dt.date(), (cutoff_dt.date() == now_ist().date())
    cutoff_dt = min(cutoff_dt, _close_dt(target_dt))

    if not track_symbol: track_symbol = os.environ.get(ENV_TRACK_SYMBOL, "").strip()

    lookback_meta = compute_lookback_iteration_schedule(
        cutoff_dt, num_iterations=lookback_iterations, interval_min=lookback_interval_min
    )

    print(f"\n{COLOR_CYAN}📡 Initializing Tracker [{mode}] | Time Machine: {cutoff_dt:%Y-%m-%d %H:%M:%S} (Live: {not explicit}){COLOR_RESET}")
    if swing_mode:
        print(
            f"   {COLOR_DIM}» Option-Delta Swing Mode: {int(CANDLE_TF_MIN)}m gap-stitched candles (overnight gaps ignored) | "
            f"Carryover: {int(SWING_LOOKBACK_SESSIONS)} sessions | Fresh Swing <= {int(SWING_FRESH_SESSIONS)} sessions | "
            f"Live |Δ| zone: {SWING_DELTA_MIN_ABS:.2f}-{SWING_DELTA_MAX_ABS:.2f} | |Δ| Trail: {SWING_DELTA_TRAIL_ATR_MULT:g}x Δ-ATR | Expiry: {EXPIRY_SELECTION}{COLOR_RESET}"
        )
    elif lookback_meta["enabled"]:
        prior_str = ", ".join(d.strftime("%H:%M") for d in lookback_meta["prior_iter_dts_desc"]) or "None"
        print(f"   {COLOR_DIM}» Lookback Engine: NO_OF_ITERATIONS={lookback_meta['num_iterations']} | INTERVAL={lookback_meta['interval_min']}m | Prior Runs: [{prior_str}] | Pure Anchor Window: {lookback_meta['window_start_dt']:%H:%M}..{lookback_meta['window_end_dt']:%H:%M} | Max Gap: {MAX_TIMESTAMP_GAP_MINUTES}m{COLOR_RESET}")
    for n in notes: print(f"   {COLOR_YELLOW}» {n}{COLOR_RESET}")
    _fb = _atr_fallback_minutes()
    _trail_txt = ("trailing " + (f"{float(SL_TRAIL_ATR_MULT):g}" if SL_TRAIL_ATR_MULT not in (None, "") else f"{float(SL_ATR_MULT):g}") + " ATR behind best price") if SL_TRAILING_ENABLED else f"fixed {float(SL_ATR_MULT):g} ATR from anchor"
    _risk_txt = f" | SL/Target ATR: {_tf_to_minutes(RISK_ATR_TF)}min" if (RISK_ATR_TF and str(RISK_ATR_TF).strip().lower() not in ("none", "off", "same", "0", "false")) else ""
    print(f"   {COLOR_DIM}» Candle TF: {int(CANDLE_TF_MIN or 1)}min | ATR basis: {_tf_to_minutes(ATR_BASIS_TF)}min x{ATR_BASIS_PERIOD} ({'gap-free' if ATR_IGNORE_GAPS else 'with-gaps'}, min {ATR_MIN_BARS} bars, fallback {str(_fb) + 'min' if _fb else 'off'}){_risk_txt} | ATR cap % of day-open: stocks {MAX_ATR_PCT:g} / index {INDEX_MAX_ATR_PCT:g} / options {OPT_MAX_ATR_PCT:g} (0=off) | SL: {_trail_txt}{COLOR_RESET}")

    if _dual_gate_required(None, mode):
        print(
            f"   {COLOR_DIM}» CE/PE Dual-Band gate: MANDATORY (premium-based IV-trap guard) | "
            f"{int(DUAL_BAND_PERIOD)}-bar {'session-isolated' if ISOLATE_SESSION_BANDS else 'gap-stitched'} BB ±{_dual_eff_std_mult():g}σ on active leg, "
            f"twin normalised by {str(DUAL_BAND_ALIGN_MODE).upper()} | twin max stale {_dual_eff_max_stale():g}m | "
            f"reversal exit {'ON' if DUAL_BAND_REVERSAL_EXIT else 'OFF'} | missing twin = no anchor | "
            f"Indicators/BB-KC/Δ-BB: Black-Scholes |Δ| (MANDATORY {int(DELTA_BB_PERIOD)}-bar {'session-isolated' if ISOLATE_SESSION_BANDS else 'gap-stitched'} |Δ| > Upper BB +{float(DELTA_BB_STD_MULT):g}σ){COLOR_RESET}"
        )

    needles = [track_symbol] if (track_symbol and track_symbol != "***") else []
    _recent_session = (now_ist().date() - target_dt).days <= 3
    universe_raw = get_dynamic_universe(mode, cutoff_dt, (today_session or _recent_session), needles=needles)
    if STATS.auth_failed or not universe_raw: return 1
    _exp_summary = ""
    if _is_option_mode(mode):
        _exps = sorted({str(it.get("expiry")) for it in universe_raw if it.get("expiry")})
        if _exps:
            _exp_summary = f" | Expiry ({EXPIRY_SELECTION}): {', '.join(_exps[:6])}"
    print(f"   {COLOR_DIM}» Universe: {len(universe_raw)} instruments{_exp_summary}{COLOR_RESET}")

    if _is_option_mode(mode):
        stock_candidates = universe_raw
        if _is_stock_option_mode(mode) and (today_session or _recent_session) and universe_raw:
            q_opt = fetch_quotes(universe_raw)
            if q_opt and any((v.get('vol') or 0) > 0 for v in q_opt.values()):
                cutoff_min = cutoff_dt.hour * 60 + cutoff_dt.minute
                elapsed_frac = max(0.05, min(1.0, (cutoff_min - SESSION_OPEN_MIN) / float(SESSION_CLOSE_MIN - SESSION_OPEN_MIN)))
                eff_quote_min_vol = float(OPT_MIN_VOLUME) * elapsed_frac
                by_pair = {}
                for it in universe_raw:
                    pid = (it.get("underlying"), it.get("expiry"), float(it.get("strike") or 0.0))
                    by_pair.setdefault(pid, []).append(it)
                scored_pairs, pinned_pairs = [], []
                for pid, legs in by_pair.items():
                    is_pin = any(it.get("pinned") for it in legs)
                    leg_stats = [(q_opt.get(it["key"], {}).get("ltp") or 0.0, q_opt.get(it["key"], {}).get("vol") or 0) for it in legs]
                    has_liquid_leg = any(ltp >= OPT_MIN_PRICE and vol >= eff_quote_min_vol for ltp, vol in leg_stats)
                    if is_pin:
                        pinned_pairs.append((pid, legs))
                    elif has_liquid_leg:
                        turnover = sum(ltp * vol for ltp, vol in leg_stats)
                        scored_pairs.append((turnover, pid, legs))
                scored_pairs.sort(key=lambda x: -x[0])
                max_pairs = int(STOCK_FNO_MAX_STRIKES or 0)
                if max_pairs > 0 and len(scored_pairs) > max_pairs:
                    scored_pairs = scored_pairs[:max_pairs]
                liquid = [it for _, legs in pinned_pairs for it in legs] + [it for _, _, legs in scored_pairs for it in legs]
                if liquid:
                    print(f"   {COLOR_DIM}» Pre-quote liquidity filter: {len(universe_raw)} -> {len(liquid)} liquid options ({len(pinned_pairs) + len(scored_pairs)} strike pairs, {MIN1_HISTORY_DAYS}d history){COLOR_RESET}")
                    stock_candidates = liquid
        if _is_index_option_mode(mode):
            for _u_idx in INDEX_CONFIG:
                _get_cached_spot_1m(_u_idx, target_dt, MIN1_HISTORY_DAYS, today_session)
        elif _is_stock_option_mode(mode) and stock_candidates:
            _u_pairs = sorted({(it.get("underlying"), it.get("underlying_key")) for it in stock_candidates if it.get("underlying") and it.get("underlying_key")})
            if _u_pairs:
                with concurrent.futures.ThreadPoolExecutor(max_workers=WORKERS) as _ex_sp:
                    list(_ex_sp.map(lambda p_: _get_cached_spot_1m(p_[0], target_dt, MIN1_HISTORY_DAYS, today_session, spot_key=p_[1]), _u_pairs))
                print(f"   {COLOR_DIM}» Underlying 1m spot cache pre-warmed: {len(_u_pairs)} equities (for BS IV & mandatory Δ-BB gate){COLOR_RESET}")
    else:
        prog = Progress("prefilter", len(universe_raw))
        lookback_cal_days = days * PREFILTER_LOOKBACK_MULT + PREFILTER_LOOKBACK_PAD_DAYS
        with concurrent.futures.ThreadPoolExecutor(max_workers=WORKERS) as ex:
            stock_candidates = [r for r in ex.map(_history_worker_daily, [(it, target_dt - timedelta(days=lookback_cal_days), target_dt, prog) for it in universe_raw]) if r is not None]
        prog.done()
        if STATS.auth_failed or not stock_candidates: return 1
        print(f"   {COLOR_DIM}» Prefilter: {len(universe_raw)} -> {len(stock_candidates)} instruments{COLOR_RESET}")

    sector_index_candidates = _sectoral_indices_universe() if (ENABLE_SECTORAL_INDICES and not _is_index_option_mode(mode)) else []
    all_candidates = sector_index_candidates + stock_candidates
    cp_interval = max(int(CANDLE_TF_MIN), int(CHECKPOINT_INTERVAL_MIN)) if swing_mode else CHECKPOINT_INTERVAL_MIN
    checkpoints = generate_checkpoints(cutoff_dt, interval_min=cp_interval) if BACKTRACE_CHECKPOINTS else [cutoff_dt]
    print(f"   {COLOR_DIM}» Checkpoints ({len(checkpoints)}): {', '.join(c.strftime('%H:%M') for c in checkpoints)}{COLOR_RESET}")

    extra_lookback_cps = []
    if (not swing_mode) and lookback_meta["enabled"]:
        cp_set = set(checkpoints)
        extra_set = set()
        for it_dt in lookback_meta["prior_iter_dts_chrono"]:
            it_cps = generate_checkpoints(it_dt) if BACKTRACE_CHECKPOINTS else [it_dt]
            for c in it_cps + [it_dt]:
                if c not in cp_set: extra_set.add(c)
        extra_lookback_cps = sorted(extra_set)

    prog = Progress("checkpoints", len(all_candidates))
    work_units = _build_work_units(all_candidates)
    with concurrent.futures.ThreadPoolExecutor(max_workers=WORKERS) as ex:
        unit_results = list(ex.map(
            process_work_unit,
            [(unit, checkpoints, extra_lookback_cps, today_session, MIN1_HISTORY_DAYS, mode, prog) for unit in work_units]
        ))
    per_symbol = [res for unit_res in unit_results for res in unit_res]
    prog.done()
    if STATS.auth_failed:
        _print_error_summary()
        return 1

    outcomes = [o for sym_outcomes, _ in per_symbol for o in sym_outcomes]
    results = [p for _, s, p in outcomes if s == "ok"]
    extra_results = [p for _, extra_outcomes in per_symbol for _, s, p in extra_outcomes if s == "ok"]
    if not results:
        status_counts = {}
        for _, s, _ in outcomes:
            status_counts[s] = status_counts.get(s, 0) + 1
        print(f"{COLOR_RED_FG}✖ No instruments produced valid checkpoint data (status breakdown: {status_counts}).{COLOR_RESET}")
        _print_error_summary()
        return 1

    by_index, by_stock = {}, {}
    for row in results:
        (by_index if row.get('IsSectorIndex') else by_stock).setdefault(row['Symbol'], []).append(row)

    # Keep Stage 2 & Still-Holding aligned: active surviving setups stay active (freshness/gap filters apply in Stage 3 Pure 1st-Time)
    idx_hold_bulls, idx_hold_bears, idx_grave = segregate_and_rank_buckets(by_index, filter_timestamp_gap=False, filter_popup_strict=False) if by_index else ([], [], [])
    stock_hold_bulls, stock_hold_bears, stock_grave = segregate_and_rank_buckets(by_stock, filter_timestamp_gap=False, filter_popup_strict=False)
    idx_bulls, idx_bears = list(idx_hold_bulls), list(idx_hold_bears)
    stock_bulls, stock_bears = list(stock_hold_bulls), list(stock_hold_bears)

    if STOCK_FNO_BLOCK_AGAINST_SECTOR and _is_stock_option_mode(mode) and (idx_hold_bulls or idx_hold_bears):
        _bull_secs = {str(r.get('Symbol', '')).upper() for r in idx_hold_bulls}
        _bear_secs = {str(r.get('Symbol', '')).upper() for r in idx_hold_bears}
        def _sector_aligned(rows_in, want_dir):
            opp_secs = _bear_secs if want_dir == "BULL" else _bull_secs
            kept = []
            for r_ in rows_in:
                sec_u = str(r_.get('Sector', '')).upper()
                if sec_u in opp_secs:
                    r_['RejectReason'] = f"Counter-Sector Setup ({sec_u} is {'BEAR' if want_dir == 'BULL' else 'BULL'})"
                    stock_grave.append(r_)
                else:
                    kept.append(r_)
            return kept
        stock_bulls = _sector_aligned(stock_bulls, "BULL")
        stock_bears = _sector_aligned(stock_bears, "BEAR")
        stock_hold_bulls = list(stock_bulls)
        stock_hold_bears = list(stock_bears)

    by_index_all_cps = {k: list(v) for k, v in by_index.items()}
    by_stock_all_cps = {k: list(v) for k, v in by_stock.items()}
    for row in extra_results:
        (by_index_all_cps if row.get('IsSectorIndex') else by_stock_all_cps).setdefault(row['Symbol'], []).append(row)

    idx_popup_res = evaluate_first_time_popups_in_memory(by_index_all_cps, idx_bulls, idx_bears, lookback_meta, pool_bulls=idx_hold_bulls, pool_bears=idx_hold_bears) if by_index_all_cps else {
        "new_bulls": [], "new_bears": [], "window_first_bulls": [], "window_first_bears": [], "iter_stats": []
    }
    stock_popup_res = evaluate_first_time_popups_in_memory(by_stock_all_cps, stock_bulls, stock_bears, lookback_meta, pool_bulls=stock_hold_bulls, pool_bears=stock_hold_bears)

    if track_symbol and track_symbol != "***":
        combined_map = dict(by_index, **by_stock)
        print_and_alert_tracked_history(track_symbol, combined_map, live_run=not explicit)

    render_console_dashboard(
        mode, cutoff_dt, checkpoints, idx_bulls, idx_bears, idx_grave,
        stock_bulls, stock_bears, stock_grave, active_sector_filter=sector_filter,
        lookback_meta=lookback_meta, idx_popup_res=idx_popup_res, stock_popup_res=stock_popup_res
    )

    if SAVE_LOCAL_HTML_FILE and html_out:
        try:
            _, _local_html = build_email_and_attachment_html(
                mode, cutoff_dt, checkpoints, idx_bulls, idx_bears, idx_grave,
                stock_bulls, stock_bears, stock_grave, lookback_meta=lookback_meta,
                idx_popup_res=idx_popup_res, stock_popup_res=stock_popup_res
            )
            with open(html_out, "w", encoding="utf-8") as f: f.write(_local_html)
            print(f"   {COLOR_GREEN_FG}» Local HTML Dashboard saved to: {html_out}{COLOR_RESET}")
        except Exception as e:
            print(f"   {COLOR_YELLOW}» Could not write local HTML file ({e}){COLOR_RESET}")

    _k_b = "new_bulls" if EMAIL_ONLY_PURE_1ST_TIME_CONTENT else "window_first_bulls"
    _k_r = "new_bears" if EMAIL_ONLY_PURE_1ST_TIME_CONTENT else "window_first_bears"
    new_idx_b = (idx_popup_res or {}).get(_k_b, [])
    new_idx_r = (idx_popup_res or {}).get(_k_r, [])
    new_stk_b = (stock_popup_res or {}).get(_k_b, [])
    new_stk_r = (stock_popup_res or {}).get(_k_r, [])
    if sector_filter:
        _nf = _norm_sym(sector_filter)
        new_stk_b = [r for r in new_stk_b if _norm_sym(r.get('Sector', '')) == _nf or _nf in _norm_sym(r.get('Sector', ''))]
        new_stk_r = [r for r in new_stk_r if _norm_sym(r.get('Sector', '')) == _nf or _nf in _norm_sym(r.get('Sector', ''))]

    if SEND_DASHBOARD_EMAIL:
        new_b_cnt, new_r_cnt = len(new_stk_b), len(new_stk_r)
        new_idx_b_cnt, new_idx_r_cnt = len(new_idx_b), len(new_idx_r)
        pure_total = new_b_cnt + new_r_cnt + new_idx_b_cnt + new_idx_r_cnt

        nb_ce, nb_pe = _ce_pe_counts(new_stk_b)
        nr_ce, nr_pe = _ce_pe_counts(new_stk_r)

        if EMAIL_SEND_ONLY_WHEN_PURE_1ST_TIME_GT_0 and pure_total <= 0:
            print(f"   {COLOR_YELLOW}📧 [dashboard-email] Skipped sending: No popup rows to send (Index: ▲{new_idx_b_cnt} ▼{new_idx_r_cnt} | Stocks: ▲{new_b_cnt} ▼{new_r_cnt}).{COLOR_RESET}")
        else:
            _email_stock_popup_res = dict(stock_popup_res or {}, **{_k_b: new_stk_b, _k_r: new_stk_r})
            plain_text_report = build_full_untruncated_plain_text(
                mode, cutoff_dt, checkpoints, idx_bulls, idx_bears, idx_grave,
                stock_bulls, stock_bears, stock_grave, lookback_meta=lookback_meta,
                idx_popup_res=idx_popup_res, stock_popup_res=_email_stock_popup_res
            )
            email_inline_html, attachment_js_html = build_email_and_attachment_html(
                mode, cutoff_dt, checkpoints, idx_bulls, idx_bears, idx_grave,
                stock_bulls, stock_bears, stock_grave, lookback_meta=lookback_meta,
                idx_popup_res=idx_popup_res, stock_popup_res=_email_stock_popup_res
            )

            w_start_s = lookback_meta["window_start_dt"].strftime("%H:%M") if (lookback_meta and lookback_meta.get("enabled")) else ""
            w_end_s = lookback_meta["window_end_dt"].strftime("%H:%M") if (lookback_meta and lookback_meta.get("enabled")) else cutoff_dt.strftime("%H:%M")
            win_tag = f"Last {int(SWING_FRESH_SESSIONS)}d" if swing_mode else f"{w_start_s}..{w_end_s}"
            lbl = _pure_label() if EMAIL_ONLY_PURE_1ST_TIME_CONTENT else "All Window Popups"
            email_subject = f"[System3 {mode}] {lbl} ({win_tag}): Index(▲{new_idx_b_cnt} ▼{new_idx_r_cnt}) | Stocks: {new_b_cnt}▲ (CE:{nb_ce} PE:{nb_pe}) | {new_r_cnt}▼ (CE:{nr_ce} PE:{nr_pe})"
            _send_email_message(
                subject=email_subject,
                plain_body=plain_text_report,
                html_body=email_inline_html,
                attachment_html=attachment_js_html if EMAIL_ATTACH_INTERACTIVE_HTML else None,
                tag="dashboard-email",
                attachment_filename=EMAIL_ATTACHMENT_FILENAME
            )

    if SEND_STILL_HOLDING_EMAIL:
        _pure_idx_keys = {(r.get('Symbol'), r.get('ActiveAnchor')) for r in (new_idx_b + new_idx_r)}
        _pure_stk_keys = {(r.get('Symbol'), r.get('ActiveAnchor')) for r in (new_stk_b + new_stk_r)}
        def _prep_holding(rows_in, pure_keys):
            out = []
            for r_ in rows_in:
                key_ = (r_.get('Symbol'), r_.get('ActiveAnchor'))
                if STILL_HOLDING_EXCLUDE_PURE_1ST_TIME and SEND_DASHBOARD_EMAIL and key_ in pure_keys:
                    continue
                if STILL_HOLDING_MIN_R is not None and _r_multiple(r_, include_banked=False) < float(STILL_HOLDING_MIN_R) - 1e-6:
                    continue
                rc = dict(r_)
                if (not swing_mode) and str(rc.get('PopupType', '')) == '1ST-TIME' and key_ not in pure_keys:
                    rc['PopupType'] = 'HOLD-1ST'
                out.append(rc)
            return out
        _eh_idx_b = _prep_holding(idx_hold_bulls, _pure_idx_keys)
        _eh_idx_r = _prep_holding(idx_hold_bears, _pure_idx_keys)
        _eh_stk_b = _prep_holding(stock_hold_bulls, _pure_stk_keys)
        _eh_stk_r = _prep_holding(stock_hold_bears, _pure_stk_keys)
        if sector_filter:
            _nf = _norm_sym(sector_filter)
            _eh_stk_b = [r for r in _eh_stk_b if _norm_sym(r.get('Sector', '')) == _nf or _nf in _norm_sym(r.get('Sector', ''))]
            _eh_stk_r = [r for r in _eh_stk_r if _norm_sym(r.get('Sector', '')) == _nf or _nf in _norm_sym(r.get('Sector', ''))]
        holding_total = len(_eh_stk_b) + len(_eh_stk_r) + len(_eh_idx_b) + len(_eh_idx_r)
        hb_ce, hb_pe = _ce_pe_counts(_eh_stk_b)
        hr_ce, hr_pe = _ce_pe_counts(_eh_stk_r)

        if EMAIL_SEND_ONLY_WHEN_STILL_HOLDING_GT_0 and holding_total <= 0:
            print(f"   {COLOR_YELLOW}📧 [still-holding-email] Skipped sending: No still-holding rows (Index: ▲{len(_eh_idx_b)} ▼{len(_eh_idx_r)} | Stocks: ▲0 ▼0).{COLOR_RESET}")
        else:
            holding_plain_text = build_still_holding_plain_text(
                mode, cutoff_dt, checkpoints, _eh_idx_b, _eh_idx_r,
                _eh_stk_b, _eh_stk_r
            )
            holding_inline_html, holding_attachment_html = build_still_holding_email_and_attachment_html(
                mode, cutoff_dt, checkpoints, _eh_idx_b, _eh_idx_r,
                _eh_stk_b, _eh_stk_r
            )
            holding_subject = (
                f"[System3 {mode}] Still Holding Good: "
                f"Index(▲{len(_eh_idx_b)} ▼{len(_eh_idx_r)}) | "
                f"Stocks: {len(_eh_stk_b)}▲ (CE:{hb_ce} PE:{hb_pe}) | "
                f"{len(_eh_stk_r)}▼ (CE:{hr_ce} PE:{hr_pe})"
            )
            _send_email_message(
                subject=holding_subject,
                plain_body=holding_plain_text,
                html_body=holding_inline_html,
                attachment_html=holding_attachment_html if EMAIL_ATTACH_STILL_HOLDING_HTML else None,
                tag="still-holding-email",
                attachment_filename=EMAIL_STILL_HOLDING_ATTACHMENT_FILENAME
            )

    _print_error_summary()
    print(f"\n⏱️ Tracker sync completed in {(time.time() - t_start):.2f} seconds ({sum(l.total_calls for l in LIMITERS.values())} API calls).\n")
    return 0

def parse_args():
    p = argparse.ArgumentParser(description="Strict institutional volatility tracker (Upstox) - System 3.0")
    p.add_argument("--mode", choices=["STOCK_FNO", "CASH_EQUITY", "INDEX_OPTIONS", "SWING_FNO", "SWING_INDEX"], default=TRADING_MODE)
    p.add_argument("--days", type=int, default=BACKTRACE_DAYS)
    p.add_argument("--date", type=str, default=None)
    p.add_argument("--time", type=str, default=None)
    p.add_argument("--history-days", type=int, default=None)
    p.add_argument("--opt-min-price", type=float, default=None)
    p.add_argument("--opt-min-volume", type=float, default=None)
    p.add_argument("--expiry", type=str, default=None, help="Expiry choice: CURRENT (0), NEXT (1), NEXT2 (2), MONTHLY, NEXT_MONTHLY, or exact date (YYYY-MM-DD / DD-MMM-YYYY).")
    p.add_argument("--auto-roll-index-expiry", dest="auto_roll_index_expiry", action="store_true", default=None)
    p.add_argument("--no-auto-roll-index-expiry", dest="auto_roll_index_expiry", action="store_false")
    p.add_argument("--auto-roll-stock-expiry", dest="auto_roll_stock_expiry", action="store_true", default=None)
    p.add_argument("--no-auto-roll-stock-expiry", dest="auto_roll_stock_expiry", action="store_false")
    p.add_argument("--strikes-up", type=int, default=None)
    p.add_argument("--strikes-down", type=int, default=None)
    p.add_argument("--best-strike-only", dest="best_strike_only", action="store_true", default=None)
    p.add_argument("--all-strikes", dest="best_strike_only", action="store_false")
    p.add_argument("--candle-tf", type=int, default=None, help="Engine candle timeframe in minutes, e.g. 3, 15, 25 (default: mode profile).")
    p.add_argument("--checkpoint-min", type=int, default=CHECKPOINT_INTERVAL_MIN)
    p.add_argument("--single-snapshot", action="store_true")
    p.add_argument("--isolate-session-bands", dest="isolate_session_bands", action="store_true", default=None)
    p.add_argument("--stitch-session-bands", dest="isolate_session_bands", action="store_false")
    p.add_argument("--no-otm-strikes", dest="no_otm_strikes", action="store_true", default=NO_OTM_STRIKES)
    p.add_argument("--allow-otm-strikes", dest="no_otm_strikes", action="store_false")
    p.add_argument("--options-buying-only", dest="options_buying_only", action="store_true", default=OPTIONS_BUYING_ONLY)
    p.add_argument("--allow-option-selling", dest="options_buying_only", action="store_false")

    # Option-Delta Swing tuning
    p.add_argument("--swing-lookback-sessions", type=int, default=None, help="Multi-day swing anchor carryover window in trading sessions (default: 10).")
    p.add_argument("--swing-fresh-sessions", type=int, default=None, help="Fresh Swing popup window in trading sessions (default: 2).")
    p.add_argument("--swing-delta-min", type=float, default=None, help="Minimum live |Delta| for swing entry (default: 0.45).")
    p.add_argument("--swing-delta-max", type=float, default=None, help="Maximum live |Delta| for swing entry (default: 0.80).")
    p.add_argument("--swing-delta-trail-atr", type=float, default=None, help="Delta trailing stop distance in Delta-ATR units (default: 2.0).")

    p.add_argument("--no-of-iterations", "--iterations", dest="no_of_iterations", type=int, default=None)
    p.add_argument("--lookback-interval", "--time-of-interval-to-lookback", dest="lookback_interval", type=int, default=None)
    p.add_argument("--no-iteration-lookback", action="store_true")
    p.add_argument("--only-current-iter-popups", dest="only_current_iter_popups", action="store_true", default=PURE_1ST_TIME_CURRENT_ITER_ONLY)

    p.add_argument("--max-timestamp-gap", type=int, default=None, help="Max gap in minutes between Seen Time and Anchor Time (default: MAX_TIMESTAMP_GAP_MINUTES).")
    p.add_argument("--enable-timestamp-gap-check", dest="enable_timestamp_gap_check", action="store_true", default=None)
    p.add_argument("--no-timestamp-gap-check", dest="enable_timestamp_gap_check", action="store_false")
    p.add_argument("--rr-type", choices=["RISK_TO_REWARD", "REWARD_TO_RISK"], default=RR_RATIO_TYPE)
    p.add_argument("--rr-sort", choices=["ASC", "DESC"], default=RR_SORT_ORDER)
    p.add_argument("--target-ladder", dest="target_ladder", action="store_true", default=None)
    p.add_argument("--no-target-ladder", dest="target_ladder", action="store_false")
    p.add_argument("--ladder-step", type=float, default=None)
    p.add_argument("--sl-trail-lock", dest="sl_trail_lock", action="store_true", default=None)
    p.add_argument("--no-sl-trail-lock", dest="sl_trail_lock", action="store_false")
    p.add_argument("--attempt-sort", dest="attempt_sort", action="store_true", default=None)
    p.add_argument("--no-attempt-sort", dest="attempt_sort", action="store_false")
    p.add_argument("--attempts-line-all-rows", dest="attempts_line_all_rows", action="store_true", default=None)
    p.add_argument("--no-attempts-line-all-rows", dest="attempts_line_all_rows", action="store_false")

    p.add_argument("--pure-reanchor", dest="pure_reanchor", action="store_true", default=None)
    p.add_argument("--no-pure-reanchor", dest="pure_reanchor", action="store_false")
    p.add_argument("--pure-reanchor-mode", dest="pure_reanchor_mode", choices=["ANY", "ALL", "NET", "NONE"], default=None)
    p.add_argument("--pure-anchor-minutes", "--pure-reanchor-minutes", dest="pure_anchor_minutes", type=int, default=None)
    p.add_argument("--pure-anchor-ignore-gap", "--pure-reanchor-ignore-gap", dest="pure_anchor_ignore_gap", action="store_true", default=None)
    p.add_argument("--no-pure-anchor-ignore-gap", "--no-pure-reanchor-ignore-gap", dest="pure_anchor_ignore_gap", action="store_false")
    p.add_argument("--pure-anchor-require-grid", "--pure-reanchor-require-grid", dest="pure_anchor_require_grid", action="store_true", default=None)
    p.add_argument("--no-pure-anchor-require-grid", "--no-pure-reanchor-require-grid", dest="pure_anchor_require_grid", action="store_false")
    p.add_argument("--pure-anchor-min-r", "--pure-reanchor-min-r", dest="pure_anchor_min_r", type=float, default=None)
    p.add_argument("--pure-include-plain", dest="pure_include_plain", action="store_true", default=None)
    p.add_argument("--no-pure-include-plain", dest="pure_include_plain", action="store_false")
    p.add_argument("--pure-reanchor-any-time", dest="pure_reanchor_any_time", action="store_true", default=None)
    p.add_argument("--no-pure-reanchor-any-time", dest="pure_reanchor_any_time", action="store_false")
    p.add_argument("--prior-min-r", dest="prior_min_r", type=float, default=None)

    p.add_argument("--sl-trailing", dest="sl_trailing", action="store_true", default=None)
    p.add_argument("--no-sl-trailing", dest="sl_trailing", action="store_false")
    p.add_argument("--sl-trail-atr", dest="sl_trail_atr", type=float, default=None)
    p.add_argument("--atr-tf", dest="atr_tf", default=None)
    p.add_argument("--risk-atr-tf", dest="risk_atr_tf", default=None)
    p.add_argument("--atr-period", dest="atr_period", type=int, default=None)
    p.add_argument("--atr-fallback-tf", dest="atr_fallback_tf", default=None)
    p.add_argument("--atr-ignore-gaps", dest="atr_ignore_gaps", action="store_true", default=None)
    p.add_argument("--atr-include-gaps", dest="atr_ignore_gaps", action="store_false")
    p.add_argument("--max-atr-pct", dest="max_atr_pct", type=float, default=None)
    p.add_argument("--index-max-atr-pct", dest="index_max_atr_pct", type=float, default=None)
    p.add_argument("--opt-max-atr-pct", dest="opt_max_atr_pct", type=float, default=None)

    p.add_argument("--send-email", dest="send_email", action="store_true", default=SEND_DASHBOARD_EMAIL)
    p.add_argument("--no-send-email", dest="send_email", action="store_false")
    p.add_argument("--email-only-when-pure-1st-time", dest="email_only_when_pure_1st_time", action="store_true", default=_parse_bool_env(ENV_EMAIL_SEND_ONLY_WHEN_PURE_1ST_TIME, EMAIL_SEND_ONLY_WHEN_PURE_1ST_TIME_GT_0))
    p.add_argument("--email-always-send", dest="email_only_when_pure_1st_time", action="store_false")
    p.add_argument("--email-only-pure-1st-time-content", dest="email_only_pure_1st_time_content", action="store_true", default=_parse_bool_env(ENV_EMAIL_ONLY_PURE_1ST_TIME_CONTENT, EMAIL_ONLY_PURE_1ST_TIME_CONTENT))
    p.add_argument("--email-all-window-popups-content", dest="email_only_pure_1st_time_content", action="store_false")

    p.add_argument("--send-still-holding-email", dest="send_still_holding_email", action="store_true", default=_parse_bool_env(ENV_SEND_STILL_HOLDING_EMAIL, SEND_STILL_HOLDING_EMAIL))
    p.add_argument("--no-send-still-holding-email", dest="send_still_holding_email", action="store_false")
    p.add_argument("--email-only-when-still-holding-gt-0", dest="email_only_when_still_holding_gt_0", action="store_true", default=_parse_bool_env(ENV_EMAIL_SEND_ONLY_WHEN_STILL_HOLDING_GT_0, EMAIL_SEND_ONLY_WHEN_STILL_HOLDING_GT_0))
    p.add_argument("--email-always-send-still-holding", dest="email_only_when_still_holding_gt_0", action="store_false")
    p.add_argument("--still-holding-exclude-pure", dest="still_holding_exclude_pure", action="store_true", default=None)
    p.add_argument("--still-holding-include-pure", dest="still_holding_exclude_pure", action="store_false")
    p.add_argument("--still-holding-min-r", dest="still_holding_min_r", type=float, default=None, help="Min live R-multiple for Still-Holding email (default: 0.0; set e.g. -0.5 to allow slightly red).")

    p.add_argument("--save-html", dest="save_html", action="store_true", default=SAVE_LOCAL_HTML_FILE)
    p.add_argument("--no-sector-indices", action="store_true")
    p.add_argument("--sector", type=str, default="")
    p.add_argument("--html-out", type=str, default=HTML_DASHBOARD_PATH)
    p.add_argument("--interactive", action="store_true")

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

    p.add_argument("--shock-mult", type=float, default=KINETIC_SHOCK_ATR_MULT)
    p.add_argument("--ind-bb-std", type=float, default=IND_BB_PIERCE_STD)
    p.add_argument("--struct-quorum", type=int, default=STRUCT_QUORUM)
    p.add_argument("--shock-quorum", type=int, default=SHOCK_QUORUM)
    p.add_argument("--shock-window", type=int, default=SHOCK_WINDOW_BARS)

    p.add_argument("--min-atr-pct", type=float, default=MIN_ATR_PCT)
    p.add_argument("--hard-stop-atr", type=float, default=HARD_STOP_ATR_MULT)
    p.add_argument("--soft-stop-atr", type=float, default=SOFT_STOP_ATR_MULT)
    p.add_argument("--target-atr", type=float, default=TARGET_ATR_MULT)
    p.add_argument("--sl-atr", type=float, default=SL_ATR_MULT)
    p.add_argument("--soft-stop-score", type=int, default=SOFT_STOP_OPPOSING_SCORE)
    p.add_argument("--soft-disable-r", type=float, default=SOFT_STOP_DISABLE_R)
    p.add_argument("--profit-lock-trig-r", type=float, default=PROFIT_LOCK_TRIGGER_R)
    p.add_argument("--profit-lock-floor-r", type=float, default=PROFIT_LOCK_FLOOR_R)
    p.add_argument("--adverse-drift-r", type=float, default=ADVERSE_DRIFT_R_FLOOR)
    p.add_argument("--day-pct-basis", choices=["PREV_CLOSE", "OPEN"], default=DAY_PCT_BASIS)
    p.add_argument("--graveyard-max", type=int, default=TOP_N_GRAVEYARD)
    p.add_argument("--iv-engine", dest="iv_engine", action="store_true", default=None)
    p.add_argument("--no-iv-engine", dest="iv_engine", action="store_false")
    p.add_argument("--iv-crush-pct", type=float, default=None)
    p.add_argument("--iv-crush-vel-mult", type=float, default=None)
    p.add_argument("--dual-band-period", dest="dual_band_period", type=int, default=None)
    p.add_argument("--dual-band-std", dest="dual_band_std", type=float, default=None)
    p.add_argument("--dual-band-align", dest="dual_band_align", choices=["ZSCORE", "OFFSET"], default=None)
    p.add_argument("--dual-band-min-sigma", dest="dual_band_min_sigma", type=float, default=None)
    p.add_argument("--dual-band-max-stale", dest="dual_band_max_stale", type=float, default=None)
    p.add_argument("--dual-band-reversal-exit", dest="dual_band_reversal_exit", action="store_true", default=None)
    p.add_argument("--no-dual-band-reversal-exit", dest="dual_band_reversal_exit", action="store_false")
    p.add_argument("--track", type=str, default="")
    return p.parse_args()

if __name__ == "__main__":
    if not get_upstox_token():
        print(f"{COLOR_RED_FG}[!] Missing {ENV_UPSTOX_TOKEN} / {ENV_FALLBACK_TOKEN} environment variable.{COLOR_RESET}")
        sys.exit(1)
    args = parse_args()

    SEND_DASHBOARD_EMAIL = args.send_email
    EMAIL_SEND_ONLY_WHEN_PURE_1ST_TIME_GT_0 = args.email_only_when_pure_1st_time
    EMAIL_ONLY_PURE_1ST_TIME_CONTENT = args.email_only_pure_1st_time_content
    SEND_STILL_HOLDING_EMAIL = args.send_still_holding_email
    EMAIL_SEND_ONLY_WHEN_STILL_HOLDING_GT_0 = args.email_only_when_still_holding_gt_0
    STILL_HOLDING_EXCLUDE_PURE_1ST_TIME = (
        args.still_holding_exclude_pure
        if args.still_holding_exclude_pure is not None
        else _parse_bool_env(ENV_STILL_HOLDING_EXCLUDE_PURE_1ST_TIME, STILL_HOLDING_EXCLUDE_PURE_1ST_TIME)
    )
    INDEX_OPTIONS_BEST_STRIKE_ONLY = (
        args.best_strike_only
        if args.best_strike_only is not None
        else _parse_bool_env(ENV_INDEX_OPTIONS_BEST_STRIKE_ONLY, INDEX_OPTIONS_BEST_STRIKE_ONLY)
    )

    SAVE_LOCAL_HTML_FILE = args.save_html
    if args.history_days is not None:
        MIN1_HISTORY_DAYS = args.history_days
        _CLI_OVERRIDES.add("history_days")
    if args.opt_min_price is not None:
        OPT_MIN_PRICE = args.opt_min_price
        _CLI_OVERRIDES.add("opt_min_price")
    if args.opt_min_volume is not None:
        OPT_MIN_VOLUME = args.opt_min_volume
        _CLI_OVERRIDES.add("opt_min_volume")
    _exp_raw = args.expiry if args.expiry is not None else os.environ.get(ENV_EXPIRY_SELECTION)
    if _exp_raw is not None and str(_exp_raw).strip() != "":
        EXPIRY_SELECTION = str(_exp_raw).strip().upper()
    if args.auto_roll_index_expiry is not None:
        INDEX_AUTO_ROLL_ON_EXPIRY_DAY = args.auto_roll_index_expiry
    else:
        INDEX_AUTO_ROLL_ON_EXPIRY_DAY = _parse_bool_env(ENV_INDEX_AUTO_ROLL_ON_EXPIRY_DAY, INDEX_AUTO_ROLL_ON_EXPIRY_DAY)
    if args.auto_roll_stock_expiry is not None:
        STOCK_FNO_AUTO_ROLL_ON_EXPIRY_DAY = args.auto_roll_stock_expiry
    else:
        STOCK_FNO_AUTO_ROLL_ON_EXPIRY_DAY = _parse_bool_env(ENV_STOCK_FNO_AUTO_ROLL_ON_EXPIRY_DAY, STOCK_FNO_AUTO_ROLL_ON_EXPIRY_DAY)
    _sh_min_r = args.still_holding_min_r if args.still_holding_min_r is not None else os.environ.get(ENV_STILL_HOLDING_MIN_R)
    if _sh_min_r is not None and str(_sh_min_r).strip() != "":
        if str(_sh_min_r).strip().lower() in ("none", "off"):
            STILL_HOLDING_MIN_R = None
        else:
            try: STILL_HOLDING_MIN_R = float(_sh_min_r)
            except ValueError: pass
    if args.strikes_up is not None:
        STRIKES_ABOVE_ATM = args.strikes_up
        STOCK_OPT_STRIKES_ABOVE_ATM = args.strikes_up
        _CLI_OVERRIDES.add("strikes_up")
    if args.strikes_down is not None:
        STRIKES_BELOW_ATM = args.strikes_down
        STOCK_OPT_STRIKES_BELOW_ATM = args.strikes_down
        _CLI_OVERRIDES.add("strikes_down")
    _ctf = args.candle_tf if args.candle_tf is not None else os.environ.get(ENV_CANDLE_TF_MIN)
    if _ctf is not None and str(_ctf).strip() != "":
        try:
            CANDLE_TF_MIN = max(1, int(_ctf))
            _CLI_OVERRIDES.add("candle_tf")
        except ValueError: pass
    CHECKPOINT_INTERVAL_MIN = args.checkpoint_min
    if args.isolate_session_bands is not None:
        ISOLATE_SESSION_BANDS = args.isolate_session_bands
        _CLI_OVERRIDES.add("isolate_session_bands")
    if args.atr_ignore_gaps is not None:
        ATR_IGNORE_GAPS = args.atr_ignore_gaps
    NO_OTM_STRIKES = args.no_otm_strikes
    OPTIONS_BUYING_ONLY = args.options_buying_only
    PURE_1ST_TIME_CURRENT_ITER_ONLY = args.only_current_iter_popups
    ENABLE_SECTORAL_INDICES = not args.no_sector_indices
    ENABLE_ITERATION_LOOKBACK = not args.no_iteration_lookback

    # Swing overrides
    for _nm, _arg, _env, _cast in (
        ("SWING_LOOKBACK_SESSIONS", args.swing_lookback_sessions, ENV_SWING_LOOKBACK_SESSIONS, int),
        ("SWING_FRESH_SESSIONS", args.swing_fresh_sessions, ENV_SWING_FRESH_SESSIONS, int),
        ("SWING_DELTA_MIN_ABS", args.swing_delta_min, ENV_SWING_DELTA_MIN_ABS, float),
        ("SWING_DELTA_MAX_ABS", args.swing_delta_max, ENV_SWING_DELTA_MAX_ABS, float),
        ("SWING_DELTA_TRAIL_ATR_MULT", args.swing_delta_trail_atr, ENV_SWING_DELTA_TRAIL_ATR_MULT, float),
    ):
        _raw = _arg if _arg is not None else os.environ.get(_env)
        if _raw is not None and str(_raw).strip() != "":
            try: globals()[_nm] = _cast(_raw)
            except ValueError: pass

    if args.no_of_iterations is not None:
        NO_OF_ITERATIONS = args.no_of_iterations
        LOOKBACK_ITERATIONS = NO_OF_ITERATIONS
    if args.lookback_interval is not None:
        TIME_OF_INTERVAL_TO_LOOKBACK = args.lookback_interval
        LOOKBACK_INTERVAL_MIN = TIME_OF_INTERVAL_TO_LOOKBACK

    if args.max_timestamp_gap is not None:
        MAX_TIMESTAMP_GAP_MINUTES = args.max_timestamp_gap
    else:
        env_gap = os.environ.get(ENV_MAX_TIMESTAMP_GAP_MINUTES)
        if env_gap and env_gap.strip() != "":
            try: MAX_TIMESTAMP_GAP_MINUTES = int(env_gap)
            except ValueError: pass

    if args.enable_timestamp_gap_check is not None:
        ENABLE_TIMESTAMP_GAP_CHECK = args.enable_timestamp_gap_check
    else:
        ENABLE_TIMESTAMP_GAP_CHECK = _parse_bool_env(ENV_ENABLE_TIMESTAMP_GAP_CHECK, ENABLE_TIMESTAMP_GAP_CHECK)

    RR_RATIO_TYPE = args.rr_type or os.environ.get(ENV_RR_RATIO_TYPE, RR_RATIO_TYPE)
    RR_SORT_ORDER = args.rr_sort or os.environ.get(ENV_RR_SORT_ORDER, RR_SORT_ORDER)

    TARGET_LADDER_ENABLED = args.target_ladder if args.target_ladder is not None else _parse_bool_env(ENV_TARGET_LADDER_ENABLED, TARGET_LADDER_ENABLED)
    SL_TRAIL_TO_LOCK_FLOOR = args.sl_trail_lock if args.sl_trail_lock is not None else _parse_bool_env(ENV_SL_TRAIL_TO_LOCK_FLOOR, SL_TRAIL_TO_LOCK_FLOOR)
    ATTEMPT_SORT_ENABLED = args.attempt_sort if args.attempt_sort is not None else _parse_bool_env(ENV_ATTEMPT_SORT_ENABLED, ATTEMPT_SORT_ENABLED)
    SHOW_ATTEMPTS_LINE_FOR_ALL_ROWS = args.attempts_line_all_rows if args.attempts_line_all_rows is not None else _parse_bool_env(ENV_SHOW_ATTEMPTS_LINE_FOR_ALL_ROWS, SHOW_ATTEMPTS_LINE_FOR_ALL_ROWS)
    PURE_ALLOW_REANCHOR_WITH_POSITIVE_PRIOR = args.pure_reanchor if args.pure_reanchor is not None else _parse_bool_env(ENV_PURE_ALLOW_REANCHOR, PURE_ALLOW_REANCHOR_WITH_POSITIVE_PRIOR)
    PURE_ANCHOR_IGNORE_GAP = args.pure_anchor_ignore_gap if args.pure_anchor_ignore_gap is not None else _parse_bool_env(ENV_PURE_ANCHOR_IGNORE_GAP, _parse_bool_env("PURE_REANCHOR_IGNORE_GAP", PURE_ANCHOR_IGNORE_GAP))
    PURE_INCLUDE_PLAIN_FIRST_TIME = args.pure_include_plain if args.pure_include_plain is not None else _parse_bool_env(ENV_PURE_INCLUDE_PLAIN_FIRST_TIME, PURE_INCLUDE_PLAIN_FIRST_TIME)
    PURE_ANCHOR_REQUIRE_GRID = args.pure_anchor_require_grid if args.pure_anchor_require_grid is not None else _parse_bool_env(ENV_PURE_ANCHOR_REQUIRE_GRID, _parse_bool_env("PURE_REANCHOR_REQUIRE_GRID", PURE_ANCHOR_REQUIRE_GRID))
    _pmr = args.pure_anchor_min_r if args.pure_anchor_min_r is not None else (os.environ.get(ENV_PURE_ANCHOR_MIN_R) or os.environ.get("PURE_REANCHOR_MIN_R"))
    if _pmr is not None and str(_pmr).strip() != "":
        try: PURE_ANCHOR_MIN_R = float(_pmr)
        except ValueError: pass
    _pm = str(args.pure_reanchor_mode or os.environ.get(ENV_PURE_REANCHOR_PRIOR_MODE) or PURE_REANCHOR_PRIOR_MODE).strip().upper()
    PURE_REANCHOR_PRIOR_MODE = _pm if _pm in ("ANY", "ALL", "NET", "NONE") else "ANY"
    _pml = args.pure_anchor_minutes if args.pure_anchor_minutes is not None else (os.environ.get(ENV_PURE_ANCHOR_LOOKBACK_MIN) or os.environ.get("PURE_REANCHOR_LOOKBACK_MIN"))
    if _pml is not None and str(_pml).strip() != "":
        try: PURE_ANCHOR_LOOKBACK_MIN = int(_pml)
        except ValueError: pass
    PURE_REANCHOR_ANY_TIME = args.pure_reanchor_any_time if args.pure_reanchor_any_time is not None else _parse_bool_env(ENV_PURE_REANCHOR_ANY_TIME, PURE_REANCHOR_ANY_TIME)
    _pmp = args.prior_min_r if args.prior_min_r is not None else os.environ.get(ENV_PRIOR_ATTEMPT_MIN_POSITIVE_R)
    if _pmp is not None and str(_pmp).strip() != "":
        try: PRIOR_ATTEMPT_MIN_POSITIVE_R = float(_pmp)
        except ValueError: pass

    SL_TRAILING_ENABLED = args.sl_trailing if args.sl_trailing is not None else _parse_bool_env(ENV_SL_TRAILING_ENABLED, SL_TRAILING_ENABLED)
    _env_trail = os.environ.get(ENV_SL_TRAIL_ATR_MULT)
    if args.sl_trail_atr is not None:
        SL_TRAIL_ATR_MULT = args.sl_trail_atr
    elif _env_trail and _env_trail.strip() != "":
        try: SL_TRAIL_ATR_MULT = float(_env_trail)
        except ValueError: pass

    _atf = args.atr_tf or os.environ.get(ENV_ATR_BASIS_TF)
    if _atf is not None and str(_atf).strip() != "":
        ATR_BASIS_TF = str(_atf).strip()
        _CLI_OVERRIDES.add("atr_tf")
    ATR_FALLBACK_TF = str(args.atr_fallback_tf or os.environ.get(ENV_ATR_FALLBACK_TF) or ATR_FALLBACK_TF).strip()
    _rtf = args.risk_atr_tf or os.environ.get(ENV_RISK_ATR_TF)
    if _rtf is not None and str(_rtf).strip() != "":
        RISK_ATR_TF = str(_rtf).strip()
        _CLI_OVERRIDES.add("risk_atr_tf")
    try: ATR_BASIS_PERIOD = int(args.atr_period or os.environ.get(ENV_ATR_BASIS_PERIOD) or ATR_BASIS_PERIOD)
    except ValueError: pass
    for _nm, _arg, _env in (("MAX_ATR_PCT", args.max_atr_pct, ENV_MAX_ATR_PCT), ("INDEX_MAX_ATR_PCT", args.index_max_atr_pct, ENV_INDEX_MAX_ATR_PCT), ("OPT_MAX_ATR_PCT", args.opt_max_atr_pct, ENV_OPT_MAX_ATR_PCT)):
        _raw = _arg if _arg is not None else os.environ.get(_env)
        if _raw is not None and str(_raw).strip() != "":
            try: globals()[_nm] = float(_raw)
            except ValueError: pass
    if args.ladder_step is not None:
        TARGET_LADDER_STEP = args.ladder_step
    else:
        env_step = os.environ.get(ENV_TARGET_LADDER_STEP)
        if env_step and env_step.strip() != "":
            try: TARGET_LADDER_STEP = float(env_step)
            except ValueError: pass

    HTML_DASHBOARD_PATH = args.html_out
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
    TARGET_ATR_MULT = args.target_atr
    SL_ATR_MULT = args.sl_atr
    SOFT_STOP_OPPOSING_SCORE = args.soft_stop_score
    SOFT_STOP_DISABLE_R = args.soft_disable_r
    PROFIT_LOCK_TRIGGER_R = args.profit_lock_trig_r
    PROFIT_LOCK_FLOOR_R = args.profit_lock_floor_r
    ADVERSE_DRIFT_R_FLOOR = args.adverse_drift_r

    DAY_PCT_BASIS = args.day_pct_basis
    TOP_N_GRAVEYARD = args.graveyard_max
    if args.single_snapshot: BACKTRACE_CHECKPOINTS = False
    if args.iv_engine is not None:
        ENABLE_OPTION_IV_ENGINE = args.iv_engine
    if args.iv_crush_pct is not None:
        IV_CRUSH_DROP_PCT = args.iv_crush_pct
    if args.iv_crush_vel_mult is not None:
        IV_CRUSH_VEL_MULT = args.iv_crush_vel_mult

    for _nm, _arg, _env, _cast in (
        ("DUAL_BAND_PERIOD", args.dual_band_period, ENV_DUAL_BAND_PERIOD, int),
        ("DUAL_BAND_STD_MULT", args.dual_band_std, ENV_DUAL_BAND_STD_MULT, float),
        ("DUAL_BAND_MIN_SIGMA", args.dual_band_min_sigma, ENV_DUAL_BAND_MIN_SIGMA, float),
        ("DUAL_BAND_TWIN_MAX_STALE_MIN", args.dual_band_max_stale, ENV_DUAL_BAND_TWIN_MAX_STALE_MIN, float),
    ):
        _raw = _arg if _arg is not None else os.environ.get(_env)
        if _raw is not None and str(_raw).strip() != "":
            try: globals()[_nm] = _cast(_raw)
            except ValueError: pass
    if DUAL_BAND_PERIOD < 2: DUAL_BAND_PERIOD = 20
    if DUAL_BAND_STD_MULT <= 0: DUAL_BAND_STD_MULT = 2.0
    if args.dual_band_std is not None or os.environ.get(ENV_DUAL_BAND_STD_MULT):
        STOCK_OPT_DUAL_BAND_STD_MULT = DUAL_BAND_STD_MULT
        _CLI_OVERRIDES.add("dual_band_std")
    if args.dual_band_max_stale is not None or os.environ.get(ENV_DUAL_BAND_TWIN_MAX_STALE_MIN):
        STOCK_OPT_DUAL_BAND_TWIN_MAX_STALE_MIN = DUAL_BAND_TWIN_MAX_STALE_MIN
        _CLI_OVERRIDES.add("dual_band_max_stale")
    _dba = str(args.dual_band_align or os.environ.get(ENV_DUAL_BAND_ALIGN_MODE) or DUAL_BAND_ALIGN_MODE).strip().upper()
    DUAL_BAND_ALIGN_MODE = _dba if _dba in ("ZSCORE", "OFFSET") else "ZSCORE"
    DUAL_BAND_REVERSAL_EXIT = args.dual_band_reversal_exit if args.dual_band_reversal_exit is not None else _parse_bool_env(ENV_DUAL_BAND_REVERSAL_EXIT, DUAL_BAND_REVERSAL_EXIT)

    try:
        code = run_screener(
            args.mode, args.days, args.date, args.time,
            track_symbol=args.track, sector_filter=args.sector,
            html_out=args.html_out, interactive_cli=args.interactive,
            lookback_iterations=args.no_of_iterations, lookback_interval_min=args.lookback_interval
        )
    except Exception:
        print(f"\n{COLOR_RED_FG}💥 FATAL: unhandled exception -- full traceback follows.{COLOR_RESET}")
        traceback.print_exc()
        code = 1
    sys.exit(code)