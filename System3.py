#!/usr/bin/env python3
"""
Strict Institutional Volatility Tracker (Upstox) - System 3.0 (Email Sectoral Accordion & Zero-Truncation Edition)
+ Zero Hardcoded Logic: All indicator periods, multipliers, stop ratchets, email toggles, and sector weights live in Section 0.
+ Multi-Index Highest-Weightage Resolver: Assigns dual-listed stocks to the sectoral index where they carry the highest %.
+ Sectoral Index 3-Lock Engine: Evaluates Sectoral Indices first (Top Bulls, Top Bears, Graveyard) before stocks.
+ In-Memory Multi-Iteration First-Time Popup Engine (Zero File/JSON Storage):
  - Evaluates NO_OF_ITERATIONS prior snapshots spaced by TIME_OF_INTERVAL_TO_LOOKBACK minutes purely in memory.
  - Compares current snapshot (e.g. 09:30 or 11:04) against prior iterations (e.g. 09:25, 09:20, 09:15 or 10:49, 10:34, 10:19).
  - Segregates Bull & Bear setups popping up for the first time with Anchor within the lookback window into a new resultset without touching existing resultsets.
+ Full Email Delivery (Zero Truncation):
  - Reads SENDER_EMAIL, SENDER_PASSWORD, RECIPIENT_EMAIL, SMTP_SERVER, SMTP_PORT, TRACK_SYMBOL, UPSTOX_ACCESS_TOKEN.
  - Sends complete, untruncated Plain-Text + Native HTML5 Expand/Collapse Sectoral Accordions + Attached Interactive HTML.
"""
import os
import sys
import re
import html
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

# --- 0.1 EMAIL & GITHUB SECRET CONFIGURATION ---
SEND_DASHBOARD_EMAIL = True             # Master toggle: Send full dashboard email after every run
SEND_TRACK_ALERT_EMAIL = True           # Master toggle: Send state-change alert email when tracking a symbol
EMAIL_ATTACH_INTERACTIVE_HTML = True    # Attach full JS click-to-filter HTML file to the email as well
EMAIL_ATTACHMENT_FILENAME = "System3_Sector_Dashboard.html"
SAVE_LOCAL_HTML_FILE = False            # False = Do not save local HTML file to disk (send via email instead)
HTML_DASHBOARD_PATH = "system3_dashboard.html"

# Zero-Truncation Email Controls (0 = Unlimited / Show Every Single Row)
EMAIL_MAX_BUYERS = 0                    # 0 = Show ALL surviving Bull setups in email without truncation
EMAIL_MAX_SELLERS = 0                   # 0 = Show ALL surviving Bear setups in email without truncation
EMAIL_MAX_GRAVEYARD = 0                 # 0 = Show ALL Graveyard rows in email without truncation
EMAIL_CLIP_SYMBOLS = False              # False = Never truncate/clip long symbol names in email

# Environment Variable Names (Matches GitHub Repository Secrets)
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

DEFAULT_SMTP_SERVER = "smtp.gmail.com"
DEFAULT_SMTP_PORT = "465"
SMTP_SSL_PORT = 465
SMTP_TIMEOUT_SEC = 30

# --- 0.2 EXECUTION & UNIVERSE FILTERING ---
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

# --- 0.3 SESSION & TIME BOUNDARIES ---
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

# --- 0.3B IN-MEMORY MULTI-ITERATION FIRST-TIME POPUP LOOKBACK (ZERO FILE STORAGE) ---
ENABLE_ITERATION_LOOKBACK = True        # Master toggle: Compare current run against N prior iterations in memory
NO_OF_ITERATIONS = 3                    # Number of prior lookback iterations (e.g., 3 -> 09:25, 09:20, 09:15 for a 09:30 run)
TIME_OF_INTERVAL_TO_LOOKBACK = 5        # Lookback interval in minutes between iterations (e.g., 5 mins)
LOOKBACK_ITERATIONS = NO_OF_ITERATIONS  # Alias for convenience
LOOKBACK_INTERVAL_MIN = TIME_OF_INTERVAL_TO_LOOKBACK

# --- 0.4 LOCK 1: PRICE VOLATILITY BANDS (BOLLINGER vs. KELTNER) ---
PRICE_BB_PERIOD = 20                    # Price Bollinger Band rolling window (bars)
PRICE_BB_STD_MULT = 2.0                 # Price Bollinger Band standard deviation multiplier
PRICE_KC_PERIOD = 20                    # Price Keltner Channel ATR rolling window (bars)
PRICE_KC_ATR_MULT = 1.5                 # Price Keltner Channel ATR multiplier
ROLLING_MIN_PERIODS = 1                 # Minimum bars required for rolling stats
ISOLATE_SESSION_BANDS = False           # False = Gap-Stitched Continuous Baseline; True = hard reset at 09:15

# --- 0.5 KINETIC INDICATOR PERIODS ---
RSI_PERIOD = 14                         # 1-minute RSI Wilder smoothing period
MACD_FAST_PERIOD = 12                   # 1-minute MACD fast EMA period (alpha = 2 / (N + 1))
MACD_SLOW_PERIOD = 26                   # 1-minute MACD slow EMA period (alpha = 2 / (N + 1))
MACD_SIGNAL_PERIOD = 9                  # 1-minute MACD signal EMA period (alpha = 2 / (N + 1))
ADX_PERIOD = 14                         # 1-minute DMI / ADX Wilder smoothing period
IND_BB_PERIOD = 20                      # Rolling window (bars) for indicator Bollinger Bands
VEL_ATR_PERIOD = 14                     # Wilder smoothing period for 1-minute indicator Velocity ATR
INDEX_SYNTHETIC_VOL_FALLBACK = True     # Synthesize range-velocity volume when spot index candles have 0 volume
EPSILON = 1e-8                          # Zero-division guard for indicator ratios

# --- 0.6 LOCK 2 & LOCK 3: STRUCTURE & KINETIC SHOCK TRIPWIRES ---
IND_BB_PIERCE_STD = 1.0                 # Lock 2: Indicator StdDev multiplier required for structure breakout
STRUCT_QUORUM = 3                       # Lock 2: Min indicators (out of 4: MACD, OBV, DI, ADX) required
KINETIC_SHOCK_ATR_MULT = 1.5            # Lock 3: Velocity must exceed this multiple of its own Velocity ATR
SHOCK_QUORUM = 3                        # Lock 3: Min indicators (out of 4: MACD, OBV, DI, ADX) required
SHOCK_WINDOW_BARS = 3                   # Lock 3: Rolling 1-minute bars to align sequential indicator shocks
SHOCK_FLOW_MIN_THRESHOLD = 0.0          # Lock 3: Current bar directional velocity floor (vh or vo > threshold)
GRID_MULTIPLIERS = [1, 2, 3, 5]         # Multipliers displayed in the ATR & BB dashboard grids

# --- 0.7 BASE ATR, TRAILING STOPS, PROFIT RATCHET & ADVERSE DRIFT ---
ATR_BASIS_PERIOD = 14                   # Number of historical timeframe bars averaged for Base ATR
ATR_BASIS_TF = "15min"                  # Timeframe string for Base ATR calculation
ATR_MIN_BARS = 5                        # Minimum completed timeframe bars required to validate Base ATR
MIN_ATR_PCT = 0.005                     # Dynamic minimum Base ATR floor as a fraction of stock price (0.5%)
INDEX_MIN_ATR_PCT = 0.0025              # Dynamic minimum Base ATR floor as a fraction of index level (0.25%)
ABSOLUTE_MIN_ATR = 0.01                 # Absolute numerical floor for Base ATR

HARD_STOP_ATR_MULT = 2.5                # Unconditional hard trailing stop distance (in Base ATRs from peak)
SOFT_STOP_ATR_MULT = 1.0                # Soft kinetic trailing stop distance (in Base ATRs from peak)
SOFT_STOP_OPPOSING_SCORE = 4            # Opposing indicator count (0-4) required to fire Soft Stop
OPPOSING_VEL_ATR_MULT = 1.0             # Opposing velocity ATR multiple required to add +1 to opposing score
OPPOSING_BB_STD_MULT = 1.0              # Opposing indicator BB StdDev multiple required to add +1 to score
WARZONE_OPPOSING_SCORE = 2              # Opposing score threshold that blocks anchor creation as Warzone Chop

SOFT_STOP_DISABLE_R = 1.5               # R-Multiple peak above which Soft Stop disables so Hard Stop rides the trend
PROFIT_LOCK_TRIGGER_R = 1.8             # R-Multiple peak required to activate the Breakeven+ Profit Ratchet
PROFIT_LOCK_FLOOR_R = 0.3               # Minimum R-Multiple floor locked in once PROFIT_LOCK_TRIGGER_R is hit

ADVERSE_DRIFT_R_FLOOR = -0.5            # Explicit R-Multiple floor for live purge at snapshot time
ADVERSE_DRIFT_STOP_RATIO = 1.0          # Ratio of HARD_STOP_ATR_MULT used when ADVERSE_DRIFT_R_FLOOR is None
BANKED_R_PROTECTION_FLOOR = 0.5         # Cumulative R threshold that protects re-entries from adverse drift purge

# --- 0.8 SECTORAL INDICES & MULTI-INDEX WEIGHTAGE MATRIX ---
ENABLE_SECTORAL_INDICES = True          # Evaluate & display Sectoral Indices before Stock tables
DEFAULT_UNMAPPED_SECTOR = "NIFTY 500"   # Fallback sector label if a symbol is not in SECTOR_CONSTITUENT_WEIGHTS
DEFAULT_UNMAPPED_WEIGHT = 0.10          # Fallback weight percentage for unmapped symbols

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

# Constituent weightage (%) per Sectoral Index.
# When a stock belongs to 2 or more indices, `resolve_primary_sector()` assigns it to the index where it has the HIGHEST %.
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
        "UNITDSPR": 3.40, "RADICO": 2.20, "PATANJALI": 2.10, "GODFRYPHLP": 1.80, "UBL": 1.70,
        "JYOTHYLAB": 1.30,
    },
    "NIFTY OIL & GAS": {
        "RELIANCE": 33.20, "ONGC": 16.50, "IOC": 11.40, "BPCL": 10.80, "GAIL": 8.90,
        "HINDPETRO": 5.60, "OIL": 4.80, "PETRONET": 4.20, "IGL": 2.90, "MGL": 2.10,
        "GUJGASLTD": 1.95, "ATGL": 1.50,
    },
    "NIFTY ENERGY": {
        "RELIANCE": 27.40, "NTPC": 15.80, "POWERGRID": 11.90, "ONGC": 10.50, "COALINDIA": 9.40,
        "TATAPOWER": 6.80, "IOC": 5.20, "BPCL": 4.90, "ADANIGREEN": 4.60, "ADANIPOWER": 4.40,
        "JSWENERGY": 3.90, "SIEMENS": 4.80, "ABB": 4.20, "CGPOWER": 3.85, "BHEL": 3.40,
        "ADANIENSOL": 3.10, "NHPC": 2.90, "SUZLON": 3.20, "WAAREEENER": 2.80, "PREMIERENE": 2.60,
        "INOXWIND": 2.10, "TORNTPOWER": 2.70, "SJVN": 1.80, "CESC": 1.60, "GVT&D": 2.20,
        "ENRIN": 2.15,
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
        "AMBER": 2.90, "BERGEPAINT": 2.50, "BATAINDIA": 2.10, "KALYANKJIL": 3.30,
        "UPL": 3.70, "PIIND": 4.00, "SRF": 4.20, "AARTIIND": 2.60, "DEEPAKNTR": 2.70,
        "TATACHEM": 2.50, "CHAMBLFERT": 2.30, "COROMANDEL": 2.80,
    },
    "NIFTY PSE": {
        "HAL": 16.40, "BEL": 15.20, "NTPC": 11.00, "POWERGRID": 9.50, "ONGC": 9.00,
        "COALINDIA": 8.20, "BDL": 5.80, "MAZDOCK": 5.60, "COCHINSHIP": 4.90, "IRCTC": 4.80,
        "IRFC": 4.50, "NBCC": 3.20, "HUDCO": 2.90, "NLCINDIA": 2.40,
    },
}

# --- 0.9 INDEX OPTIONS CONFIGURATION ---
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

# --- 0.10 DASHBOARD DISPLAY & FORMATTING ---
TOP_N_BUYERS = 50                       # Max rows in Console Top Bull Setups (0 = unlimited)
TOP_N_SELLERS = 50                      # Max rows in Console Top Bear Setups (0 = unlimited)
TOP_N_GRAVEYARD = 100                   # Max rows in Console Graveyard (0 = unlimited)
MIN_SYMBOL_WIDTH = 12                   # Minimum width of the Script column
MAX_SYMBOL_WIDTH = 30                   # Maximum width of the Script column
SECTOR_COL_WIDTH = 18                   # Width of Sector column in Stock tables
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
# HELPERS, SECTOR RESOLVER & RATE LIMITERS
# ==============================================================================
def get_upstox_token():
    return os.environ.get(ENV_UPSTOX_TOKEN) or os.environ.get(ENV_FALLBACK_TOKEN)

def get_lookback_iterations_config():
    """
    Resolves NO_OF_ITERATIONS and TIME_OF_INTERVAL_TO_LOOKBACK from environment
    variables (if provided in GitHub Actions) or falls back to Section 0 globals.
    """
    raw_iters = os.environ.get(ENV_NO_OF_ITERATIONS) or os.environ.get(ENV_LOOKBACK_ITERATIONS)
    raw_interval = os.environ.get(ENV_TIME_OF_INTERVAL_TO_LOOKBACK) or os.environ.get(ENV_LOOKBACK_INTERVAL_MIN)
    try:
        iters = int(raw_iters) if raw_iters is not None and str(raw_iters).strip() != "" else int(NO_OF_ITERATIONS)
    except ValueError:
        iters = int(NO_OF_ITERATIONS)
    try:
        interval = int(raw_interval) if raw_interval is not None and str(raw_interval).strip() != "" else int(TIME_OF_INTERVAL_TO_LOOKBACK)
    except ValueError:
        interval = int(TIME_OF_INTERVAL_TO_LOOKBACK)
    return max(0, iters), max(1, interval)

def now_ist():
    return datetime.now(IST).replace(tzinfo=None)

def _close_dt(day):
    return datetime.combine(day, dtime(SESSION_CLOSE_MIN // 60, SESSION_CLOSE_MIN % 60))

def _open_dt(day):
    return datetime.combine(day, dtime(SESSION_OPEN_MIN // 60, SESSION_OPEN_MIN % 60))

def _norm_sym(s):
    return re.sub(r'[\s_\-]+', '', str(s).upper())

def resolve_primary_sector(symbol):
    """
    Checks all Sectoral Indices where `symbol` is a constituent and picks
    the single Sectoral Index where `symbol` has the HIGHEST percentage weight.
    """
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
    if name in _MASTER_CACHE:
        return _MASTER_CACHE[name]
    url = f"{MASTER_BASE_URL}/{name}.json.gz"
    for attempt in range(1, MASTER_DOWNLOAD_RETRIES + 1):
        try:
            resp = requests.get(url, timeout=MASTER_DOWNLOAD_TIMEOUT_SEC)
            if resp.status_code == 200:
                try:
                    rows = _unwrap_master(json.loads(gzip.decompress(resp.content).decode('utf-8')))
                    if rows:
                        _MASTER_CACHE[name] = rows
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
            _MASTER_CACHE[name] = rows
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

def _sectoral_indices_universe():
    nse = _download_master("NSE")
    master_idx_map = {}
    for row in (nse or []):
        if str(row.get("segment", "")).upper() == "NSE_INDEX":
            ik = row.get("instrument_key")
            for field_name in ("name", "trading_symbol", "tradingsymbol"):
                val = row.get(field_name)
                if val and ik:
                    master_idx_map[_norm_sym(val)] = ik
            if ik and "|" in ik:
                master_idx_map[_norm_sym(ik.split("|", 1)[1])] = ik

    out = []
    for sec_name, cfg in SECTORAL_INDEX_CONFIG.items():
        resolved_key = None
        for alias in [sec_name] + list(cfg.get("aliases", [])):
            norm_a = _norm_sym(alias)
            if norm_a in master_idx_map:
                resolved_key = master_idx_map[norm_a]
                break
        if not resolved_key:
            resolved_key = cfg["spot_key"]
        out.append({
            "symbol": sec_name,
            "key": resolved_key,
            "pinned": True,
            "is_sector_index": True,
            "sector": sec_name,
            "sector_weight": 100.0,
        })
    return out

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
    out = {}
    for k, v in in_mode.items():
        sec_name, sec_wt, all_sec = resolve_primary_sector(v["symbol"])
        out[k] = dict(v, pinned=v["symbol"] in pinned, is_sector_index=False, sector=sec_name, sector_weight=sec_wt, all_sectors=all_sec)
    for k, v in everything.items():
        if v["symbol"] in pinned and k not in out:
            sec_name, sec_wt, all_sec = resolve_primary_sector(v["symbol"])
            out[k] = dict(v, pinned=True, is_sector_index=False, sector=sec_name, sector_weight=sec_wt, all_sectors=all_sec)
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
                universe.append({"key": key, "symbol": sym, "pinned": True, "is_sector_index": False, "sector": "INDEX_OPTIONS", "sector_weight": 100.0})

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
                universe.append({
                    "key": key,
                    "symbol": str(_field(row, "trading_symbol", "tradingsymbol") or key),
                    "pinned": False,
                    "is_sector_index": False,
                    "sector": idx,
                    "sector_weight": 100.0,
                })
    return universe

def get_dynamic_universe(mode, cutoff_dt, use_intraday, needles=()):
    if mode in ("STOCK_FNO", "CASH_EQUITY"):
        return _equity_universe(mode, needles)
    if mode == "INDEX_OPTIONS":
        return _index_options_universe(cutoff_dt, use_intraday, needles)
    return []

# ==============================================================================
# 2. CONTINUOUS KINETIC TRIPWIRE ENGINE (SHARED BY INDICES & STOCKS)
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

    direction = np.where(delta > 0, 1, np.where(delta < 0, -1, 0))
    obv = np.cumsum(direction * eff_vol)

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
            if hard_mult > 0 and hard_adverse >= hard_mult * base_atr:
                reason = f"Hard stop ({hard_mult:g} ATR trailing floor)"
            elif lock_trig_r > 0 and max_r >= lock_trig_r and curr_r <= lock_floor_r:
                reason = f"Profit Lock (+{lock_trig_r:g}R Peak -> +{lock_floor_r:g}R Breakeven Floor)"
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

def prepare_series(master, target_dt, is_sector_index=False):
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

    is_late_session = (master['Min'].values >= LAST_NEW_ANCHOR_MIN)

    base_atr_calc, atr_bars = compute_base_atr(master.iloc[:ts])
    if base_atr_calc is None or atr_bars < ATR_MIN_BARS:
        return None, "short_history"
    atr_pct_floor = float(INDEX_MIN_ATR_PCT) if is_sector_index else float(MIN_ATR_PCT)
    base_atr = max(float(base_atr_calc), float(close[ts]) * atr_pct_floor, float(ABSOLUTE_MIN_ATR))

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

def row_at(S, end, symbol, sector="", sector_weight=0.0, is_sector_index=False):
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
        'Kills': [], 'AtrBars': S['atr_bars'], 'BaseAtr': base_atr, 'PriorStops': 0, 'BankedR': 0.0,
        'Sector': sector or DEFAULT_UNMAPPED_SECTOR,
        'SectorWeight': float(sector_weight or 0.0),
        'IsSectorIndex': bool(is_sector_index),
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

# ==============================================================================
# 3. WORKERS, SEGREGATION & MULTI-EPISODE MERGE LOGIC
# ==============================================================================
def _history_worker_daily(args):
    item, start, end, progress = args
    try:
        if item.get("pinned") or item.get("is_sector_index"):
            return item
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
    item, checkpoints, extra_lookback_cps, use_intraday, history_days, mode, progress = args
    sym = item['symbol']
    is_sec_idx = bool(item.get('is_sector_index', False))
    sec_name = item.get('sector', DEFAULT_UNMAPPED_SECTOR)
    sec_wt = float(item.get('sector_weight', 0.0))
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
            return [(cp, "no_data", sym) for cp in checkpoints], []

        master = prepare_master(frames)
        master = master[master['Datetime'] < final_cutoff].reset_index(drop=True)
        if mode == "INDEX_OPTIONS" and not is_sec_idx:
            day = master[master['Datetime'].dt.date == target_dt]
            if day.empty:
                return [(cp, "no_data", sym) for cp in checkpoints], []
            if day['Close'].iloc[-1] < OPT_MIN_PRICE or day['Volume'].sum() < OPT_MIN_VOLUME:
                return [(cp, "illiquid", sym) for cp in checkpoints], []

        S, why = prepare_series(master, target_dt, is_sector_index=is_sec_idx)
        if S is None:
            return [(cp, why, sym) for cp in checkpoints], []

        # 1. Exact evaluation for primary checkpoints (untouched)
        ends = np.searchsorted(S['dt'], np.array(checkpoints, dtype='datetime64[ns]'), side='left')
        out = []
        by_cp_row = {}
        for cp, end in zip(checkpoints, ends):
            row = row_at(S, int(end), sym, sector=sec_name, sector_weight=sec_wt, is_sector_index=is_sec_idx)
            if row is None:
                out.append((cp, "no_data", sym))
            else:
                row['CheckpointTime'] = cp
                out.append((cp, "ok", row))
                by_cp_row[cp] = row

        # 2. In-memory evaluation for any additional lookback iteration checkpoints (e.g., 09:15 open bar, or 10:19/10:34/10:49)
        extra_out = []
        session_open_dt = _open_dt(target_dt)
        for l_cp in (extra_lookback_cps or []):
            if l_cp in by_cp_row:
                extra_out.append((l_cp, "ok", by_cp_row[l_cp]))
                continue
            if l_cp == session_open_dt:
                # At 09:15 session open, include the 09:15 1-minute bar if present
                l_end = int(np.searchsorted(S['dt'], np.datetime64(l_cp), side='right'))
            else:
                l_end = int(np.searchsorted(S['dt'], np.datetime64(l_cp), side='left'))
            l_row = row_at(S, l_end, sym, sector=sec_name, sector_weight=sec_wt, is_sector_index=is_sec_idx)
            if l_row is not None:
                l_row['CheckpointTime'] = l_cp
                extra_out.append((l_cp, "ok", l_row))

        return out, extra_out
    except Exception as e:
        ERRORS.add(f"1m:{item.get('symbol')}", e)
        return [(cp, "error", item.get('symbol')) for cp in checkpoints], []
    finally:
        progress.tick()

def _clip(s, w, allow_clip=True):
    if not allow_clip:
        return str(s)
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

def segregate_and_rank_buckets(by_symbol_map):
    bulls_live, bears_live, graveyard_candidates = [], [], []

    for symbol, rows in by_symbol_map.items():
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

    def sort_reason(r):
        return 0 if any(x in str(r.get('RejectReason', '')) for x in ["Hard stop", "Profit Lock", "Kinetic SL", "Superseded"]) else 1 if "Adverse" in str(r.get('RejectReason', '')) else 2 if "Warzone" in str(r.get('RejectReason', '')) else 3
    graveyard_candidates.sort(key=lambda x: (sort_reason(x), -_r_multiple(x) if _r_multiple(x) != 0 else -abs(x.get('DayChangePct', 0.0))))

    dedup_grave, seen_grave = [], set()
    for row in graveyard_candidates:
        gid = (row['Symbol'], row.get('ActiveAnchor', '-'), row.get('KilledTime', '-'))
        if gid not in seen_grave:
            seen_grave.add(gid)
            dedup_grave.append(row)

    return surviving_bulls, surviving_bears, dedup_grave

# ==============================================================================
# 3B. IN-MEMORY MULTI-ITERATION FIRST-TIME POPUP ENGINE (ZERO FILE STORAGE)
# ==============================================================================
def compute_lookback_iteration_schedule(cutoff_dt, num_iterations=None, interval_min=None):
    """
    Given a snapshot `cutoff_dt` (e.g. 09:30 or 11:04), `NO_OF_ITERATIONS` (e.g. 3),
    and `TIME_OF_INTERVAL_TO_LOOKBACK` (e.g. 5 or 15 mins):
      - Prior iteration cutoffs: [T - 1*step, T - 2*step, ..., T - N*step] (clamped to >= 09:15 session open)
      - Window for Anchor Time validation: [window_start_dt, cutoff_dt] (e.g. 09:15..09:30 or 10:19..11:04)
    """
    cfg_iters, cfg_interval = get_lookback_iterations_config()
    n_iters = cfg_iters if num_iterations is None else max(0, int(num_iterations))
    step_min = cfg_interval if interval_min is None else max(1, int(interval_min))

    session_open_dt = _open_dt(cutoff_dt.date())
    window_start_dt = max(session_open_dt, cutoff_dt - timedelta(minutes=n_iters * step_min))

    prior_iter_dts = []
    for k in range(1, n_iters + 1):
        it_dt = cutoff_dt - timedelta(minutes=k * step_min)
        if it_dt >= session_open_dt and it_dt not in prior_iter_dts:
            prior_iter_dts.append(it_dt)

    # Sort chronologically (e.g. [10:19, 10:34, 10:49]) for sequential first-seen tracking
    prior_iter_dts_chrono = sorted(prior_iter_dts)
    return {
        "enabled": bool(ENABLE_ITERATION_LOOKBACK and n_iters > 0),
        "num_iterations": n_iters,
        "interval_min": step_min,
        "window_start_dt": window_start_dt,
        "window_end_dt": cutoff_dt,
        "prior_iter_dts_desc": prior_iter_dts,
        "prior_iter_dts_chrono": prior_iter_dts_chrono,
    }

def _anchor_within_lookback_window(anchor_str, window_start_dt, window_end_dt):
    if not anchor_str or anchor_str == "-":
        return False
    clean_t = str(anchor_str).replace("*", "").strip()
    try:
        parsed_t = datetime.strptime(clean_t, "%H:%M").time()
        anchor_dt = datetime.combine(window_end_dt.date(), parsed_t)
        return window_start_dt <= anchor_dt <= window_end_dt
    except ValueError:
        return False

def evaluate_first_time_popups_in_memory(
    by_symbol_all_cps,
    current_bulls,
    current_bears,
    lookback_meta
):
    """
    Purely in-memory comparison across lookback iterations (zero JSON/file storage):
    1. Evaluates `segregate_and_rank_buckets` for each prior lookback iteration in
       `prior_iter_dts_chrono` (e.g. [09:15, 09:20, 09:25] or [10:19, 10:34, 10:49]).
    2. Compares the current run's surviving `current_bulls` and `current_bears`
       (e.g. 09:30 or 11:04) against the prior iteration results:
       - Requires `ActiveAnchor` time to be within `[window_start_dt, window_end_dt]`
         (e.g. 09:15..09:30 or 10:19..11:04).
       - Note: An anchor formed inside the window (e.g. at 10:22 inside 10:19..11:04)
         naturally appears at 10:34 and remains active with that SAME anchor at 10:49 and 11:04.
         Therefore, holding the SAME anchor (`anc == curr_anchor`) across subsequent
         iterations inside the window is the continuation of that single popup, NOT a prior
         disqualifying episode!
       - Segregates into:
         a) `new_bulls` / `new_bears` (First-Time Popups — Zero Prior Same-Direction Stops, No `*`):
            Setups with `ActiveAnchor` in `[window_start_dt, window_end_dt]`, `PriorStops == 0`
            (no `*` re-entry badge in the same direction; opposite-direction kills earlier in the
            day do NOT disqualify a fresh reversal!), and never seen with a different anchor in
            any prior iteration.
            (e.g. at 11:04: NIFTY 22550 CE @ 10:22, SENSEX 73100 PE @ 10:22, SENSEX 73300 CE @ 10:22, SENSEX 73400 CE @ 10:22)
         b) `window_first_bulls` / `window_first_bears` (All Window Popups in `window_start..window_end`, Including `*` Re-Anchors):
            All surviving setups at `window_end_dt` whose `ActiveAnchor` formed within
            `[window_start_dt, window_end_dt]` (both fresh `PriorStops == 0` anchors like `10:22`
            and new `*` anchors formed inside the window like `10:58*`, `10:59*`, `10:43*`),
            tagged with `FirstPopupIter` (exact iteration where this anchor first popped up)
            and `PopupType` (`"1ST-TIME"` vs `"1ST-IN-ITERS*"` vs `"RE-ANCHOR*"`).
    """
    if not lookback_meta or not lookback_meta.get("enabled"):
        return {
            "new_bulls": [],
            "new_bears": [],
            "window_first_bulls": [],
            "window_first_bears": [],
            "iter_stats": [],
        }

    w_start = lookback_meta["window_start_dt"]
    w_end = lookback_meta["window_end_dt"]
    prior_dts = lookback_meta["prior_iter_dts_chrono"]

    # Map: sym -> list of (iter_label, active_anchor_str) in prior iterations
    prior_bull_appearances = {}
    prior_bear_appearances = {}
    iter_stats = []

    for it_dt in prior_dts:
        it_cps = set(generate_checkpoints(it_dt, interval_min=CHECKPOINT_INTERVAL_MIN) if BACKTRACE_CHECKPOINTS else [it_dt])
        it_cps.add(it_dt)

        sub_map = {}
        for sym, rows in by_symbol_all_cps.items():
            f_rows = [r for r in rows if r['CheckpointTime'] <= it_dt and r['CheckpointTime'] in it_cps]
            if not f_rows:
                f_rows = [r for r in rows if r['CheckpointTime'] <= it_dt]
            if f_rows:
                sub_map[sym] = f_rows

        it_bulls, it_bears, it_grave = segregate_and_rank_buckets(sub_map) if sub_map else ([], [], [])
        it_label = it_dt.strftime("%H:%M")
        iter_stats.append({
            "iter_time": it_label,
            "bull_count": len(it_bulls),
            "bear_count": len(it_bears),
            "grave_count": len(it_grave),
        })

        for r in it_bulls:
            sym = r['Symbol']
            prior_bull_appearances.setdefault(sym, []).append((it_label, str(r.get('ActiveAnchor') or '-').replace('*', '').strip()))
        for r in it_bears:
            sym = r['Symbol']
            prior_bear_appearances.setdefault(sym, []).append((it_label, str(r.get('ActiveAnchor') or '-').replace('*', '').strip()))

    curr_label = w_end.strftime("%H:%M")
    new_bulls, new_bears = [], []
    window_first_bulls, window_first_bears = [], []

    def _classify_rows(current_list, prior_app_map, out_pure_first, out_all_window):
        for row in current_list:
            sym = row['Symbol']
            curr_anc = str(row.get('ActiveAnchor') or '-').replace('*', '').strip()
            if not _anchor_within_lookback_window(curr_anc, w_start, w_end):
                continue

            appearances = prior_app_map.get(sym, [])
            same_anc_iters = [it_lbl for (it_lbl, anc) in appearances if anc == curr_anc]
            other_anc_iters = [it_lbl for (it_lbl, anc) in appearances if anc != curr_anc]

            first_it = same_anc_iters[0] if same_anc_iters else curr_label
            prior_stops_same_dir = int(row.get('PriorStops', 0) or 0)
            banked_r = float(row.get('BankedR', 0.0) or 0.0)
            has_star_badge = (prior_stops_same_dir > 0 and abs(banked_r) > BANKED_R_DISPLAY_EPSILON)

            row_copy = dict(row)
            row_copy['FirstPopupIter'] = first_it

            if prior_stops_same_dir == 0 and not has_star_badge and len(other_anc_iters) == 0:
                row_copy['PopupType'] = "1ST-TIME"
                out_pure_first.append(row_copy)
                out_all_window.append(row_copy)
            elif len(other_anc_iters) == 0:
                # Never appeared in any prior lookback iteration [e.g. 10:19, 10:34, 10:49],
                # though it had an earlier stop prior to the lookback window
                row_copy['PopupType'] = "1ST-IN-ITERS*"
                out_all_window.append(row_copy)
            else:
                # Had an older anchor in a prior lookback iteration that stopped out and re-anchored inside the window
                row_copy['PopupType'] = "RE-ANCHOR*"
                out_all_window.append(row_copy)

    _classify_rows(current_bulls, prior_bull_appearances, new_bulls, window_first_bulls)
    _classify_rows(current_bears, prior_bear_appearances, new_bears, window_first_bears)

    return {
        "new_bulls": new_bulls,
        "new_bears": new_bears,
        "window_first_bulls": window_first_bulls,
        "window_first_bears": window_first_bears,
        "iter_stats": iter_stats,
    }

def _state_display(row):
    return f"[STOP {row.get('KilledTime')}]" if row['State'] == "[STOPPED]" and row.get('KilledTime') and row.get('KilledTime') != '-' else row['State']

def _fav_color(pct, direction):
    fav = pct * (1 if direction == "BULL" else -1)
    return COLOR_GREEN_FG if fav > 0 else COLOR_RED_FG if fav < 0 else COLOR_DIM

def _format_val(v, dir_color):
    return f"{dir_color}{v}{COLOR_RESET}" if v > 0 else f"{COLOR_DIM}{v}{COLOR_RESET}"

# ==============================================================================
# 4. CONSOLE RENDERER & UNTRUNCATED EMAIL / ACCORDION HTML ENGINE
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

    all_shown = idx_bulls + idx_bears + idx_grave + f_stock_bulls + f_stock_bears + f_stock_grave
    sym_w = max(MIN_SYMBOL_WIDTH, min(MAX_SYMBOL_WIDTH, max((len(r['Symbol']) for r in all_shown), default=MIN_SYMBOL_WIDTH)))
    sec_w = SECTOR_COL_WIDTH

    print(f"\n{COLOR_BOLD}=== STATEFUL INSTITUTIONAL VOLATILITY TRACKER [{mode}]{filter_banner} ==={COLOR_RESET}")
    print(
        f"Target Snapshot: {cutoff_dt.strftime('%Y-%m-%d %H:%M')} | Checkpoints: {len(checkpoints)} | "
        f"Sectoral Indices (Bull: {len(idx_bulls)}, Bear: {len(idx_bears)}, Grave: {len(idx_grave)}) | "
        f"Stocks Shown (Bull: {len(f_stock_bulls)}, Bear: {len(f_stock_bears)}, Grave: {len(f_stock_grave)})\n"
    )

    def print_active_table(title, icon, data_list, show_sector_col=True, show_first_iter_col=False):
        if not data_list:
            return
        print(f"\n{COLOR_BOLD}{icon} {title}{COLOR_RESET}")
        grid_hdr = " ".join(f"{m}X" for m in GRID_MULTIPLIERS)
        sec_hdr = f"{'Sector (MaxWt%)':<{sec_w}} | " if show_sector_col else ""
        iter_hdr = f"{'1stIter':^7} {'PopupType':^13} | " if show_first_iter_col else ""
        header_str = (
            f" {COLOR_CYAN}{'Script':<{sym_w}} {'LTP/Exit':>9} | "
            f"{sec_hdr}{iter_hdr}"
            f"{'Seen':^5} {'Anchor':^7} {'Anch Val':>9} {'R-Mult (Bank)':>15} | "
            f"ATR: {grid_hdr} | BB: {grid_hdr} | {'State':^13}{COLOR_RESET}"
        )
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
                r_mult_str, anchor_disp = f"{live_r:>+5.2f}R         ", f"{row['ActiveAnchor']} "
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
            ltp_val = f"{row['KilledPrice']:>8.2f}x" if "[STOP" in shown_state and row.get('KilledPrice', 0) > 0 else f"{row['LTP']:>9.2f} "
            sec_label = f"{row.get('Sector', DEFAULT_UNMAPPED_SECTOR)}"
            sec_cell = f"{_clip(sec_label, sec_w):<{sec_w}} | " if show_sector_col else ""
            if show_first_iter_col:
                f_it = row.get('FirstPopupIter', row['CheckpointTime'].strftime('%H:%M'))
                p_type = row.get('PopupType', '1ST-TIME')
                iter_cell = f"{f_it:^7} {p_type:^13} | "
            else:
                iter_cell = ""
            print(
                f" {COLOR_BOLD}{_clip(row['Symbol'], sym_w):<{sym_w}}{COLOR_RESET} {ltp_val} | "
                f"{sec_cell}{iter_cell}"
                f"{row['CheckpointTime'].strftime('%H:%M'):^5} {anchor_disp:^7} {row['AnchorPrice']:>9.2f} "
                f"{r_color}{r_mult_str:>15}{COLOR_RESET} |   {a_str}  |   {b_str}  | {state_text}"
            )

    def print_grave_table(title, icon, data_list, show_sector_col=True):
        if not data_list:
            return
        print(f"\n{COLOR_BOLD}{icon} {title}{COLOR_RESET}")
        sec_hdr = f"{'Sector':<{sec_w}} | " if show_sector_col else ""
        header_str = (
            f" {COLOR_CYAN}{'Script':<{sym_w}} | {sec_hdr}{'Signal':^6} {'Anchor':^6} {'Anch Val':>9} "
            f"{'Killed':^6} {'Kill Val':>9} {'LTP':>9} {'R-Mult':>7} {'Move%':>7} | "
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
            reason_color = COLOR_YELLOW if any(x in reason for x in ["Kinetic SL", "Hard stop", "Adverse", "Profit Lock", "Superseded"]) else COLOR_RED_FG if any(x in reason for x in ["Chop", "Warzone"]) else COLOR_DIM
            sec_cell = f"{_clip(row.get('Sector', DEFAULT_UNMAPPED_SECTOR), sec_w):<{sec_w}} | " if show_sector_col else ""
            print(
                f" {COLOR_BOLD}{_clip(row['Symbol'], sym_w):<{sym_w}}{COLOR_RESET} | {sec_cell}{sig_colored} "
                f"{a_time:^6} {a_price} {k_time:^6} {k_price} {row['LTP']:>9.2f} "
                f"{g_r_color}{g_r_str}{COLOR_RESET} {move_color}{move_pct_str}{COLOR_RESET} | {reason_color}{reason}{COLOR_RESET}"
            )

    # --- EXISTING RESULTSET STAGE 1 (UNTOUCHED) ---
    if idx_bulls or idx_bears or idx_grave:
        print(f"{COLOR_CYAN}{'=' * 95}{COLOR_RESET}")
        print(f"{COLOR_BOLD}🏛️  STAGE 1: SECTORAL INDICES (Evaluated with 3-Lock Kinetic Engine){COLOR_RESET}")
        print(f"{COLOR_CYAN}{'=' * 95}{COLOR_RESET}")
        print_active_table("SECTORAL INDICES — TOP BULL SETUPS (Valid Anchors Surviving)", "🏛️🔥", idx_bulls, show_sector_col=False)
        print_active_table("SECTORAL INDICES — TOP BEAR SETUPS (Valid Anchors Surviving)", "🏛️🩸", idx_bears, show_sector_col=False)
        print_grave_table("SECTORAL INDICES — THE GRAVEYARD (Completed / Filtered / Rejected)", "🏛️🚫", idx_grave, show_sector_col=False)

    # --- EXISTING RESULTSET STAGE 2 (UNTOUCHED) ---
    print(f"\n{COLOR_CYAN}{'=' * 95}{COLOR_RESET}")
    stage2_label = f"STAGE 2: STOCKS FOR SECTOR [{active_sector_filter.upper()}]" if norm_filter else "STAGE 2: ALL STOCKS (Mapped by Highest Sectoral Index Weightage)"
    print(f"{COLOR_BOLD}📈  {stage2_label}{COLOR_RESET}")
    print(f"{COLOR_CYAN}{'=' * 95}{COLOR_RESET}")
    print_active_table("STOCKS — TOP BULL SETUPS (Valid Anchors Surviving)", "🔥", f_stock_bulls, show_sector_col=True)
    print_active_table("STOCKS — TOP BEAR SETUPS (Valid Anchors Surviving)", "🩸", f_stock_bears, show_sector_col=True)
    if not f_stock_bulls and not f_stock_bears:
        print(f"{COLOR_YELLOW}No surviving stock anchors for this view.{COLOR_RESET}")
    print_grave_table("STOCKS — THE GRAVEYARD (Completed Episodes / Filtered / Rejected)", "🚫", f_stock_grave, show_sector_col=True)

    # --- NEW SEGREGATED RESULTSET: MULTI-ITERATION FIRST-TIME POPUPS (ADDED WITHOUT TOUCHING EXISTING) ---
    if lookback_meta and lookback_meta.get("enabled") and stock_popup_res is not None:
        w_start_s = lookback_meta["window_start_dt"].strftime("%H:%M")
        w_end_s = lookback_meta["window_end_dt"].strftime("%H:%M")
        prior_s = ", ".join(d.strftime("%H:%M") for d in lookback_meta["prior_iter_dts_desc"]) or "None"
        iter_summary_s = " | ".join(
            f"{st['iter_time']} (▲{st['bull_count']} ▼{st['bear_count']})"
            for st in (stock_popup_res.get("iter_stats") or [])
        )

        new_idx_bulls = (idx_popup_res or {}).get("new_bulls", [])
        new_idx_bears = (idx_popup_res or {}).get("new_bears", [])
        win_idx_bulls = (idx_popup_res or {}).get("window_first_bulls", [])
        win_idx_bears = (idx_popup_res or {}).get("window_first_bears", [])

        new_stk_bulls = stock_popup_res.get("new_bulls", [])
        new_stk_bears = stock_popup_res.get("new_bears", [])
        win_stk_bulls = stock_popup_res.get("window_first_bulls", [])
        win_stk_bears = stock_popup_res.get("window_first_bears", [])

        if norm_filter:
            new_stk_bulls = [r for r in new_stk_bulls if _norm_sym(r.get('Sector', '')) == norm_filter or norm_filter in _norm_sym(r.get('Sector', ''))]
            new_stk_bears = [r for r in new_stk_bears if _norm_sym(r.get('Sector', '')) == norm_filter or norm_filter in _norm_sym(r.get('Sector', ''))]
            win_stk_bulls = [r for r in win_stk_bulls if _norm_sym(r.get('Sector', '')) == norm_filter or norm_filter in _norm_sym(r.get('Sector', ''))]
            win_stk_bears = [r for r in win_stk_bears if _norm_sym(r.get('Sector', '')) == norm_filter or norm_filter in _norm_sym(r.get('Sector', ''))]

        print(f"\n{COLOR_CYAN}{'=' * 95}{COLOR_RESET}")
        print(
            f"{COLOR_BOLD}🆕  STAGE 3 (NEW SEGREGATED RESULTSET): FIRST-TIME POPUPS ACROSS LOOKBACK ITERATIONS{COLOR_RESET}\n"
            f"   {COLOR_DIM}» Config: NO_OF_ITERATIONS={lookback_meta['num_iterations']} | "
            f"TIME_OF_INTERVAL_TO_LOOKBACK={lookback_meta['interval_min']}m | "
            f"Anchor Window: {w_start_s} to {w_end_s} | Prior Runs Compared: [{prior_s}] vs Current [{w_end_s}]{COLOR_RESET}"
        )
        if iter_summary_s:
            print(f"   {COLOR_DIM}» Prior Iteration Stock Counts: {iter_summary_s}{COLOR_RESET}")
        print(f"{COLOR_CYAN}{'=' * 95}{COLOR_RESET}")

        if new_idx_bulls or new_idx_bears:
            print_active_table(
                f"PURE FIRST-TIME SECTORAL INDEX POPUPS (No Prior Stops, Anchor {w_start_s}..{w_end_s}) — BULL",
                "🏛️🆕🔥", new_idx_bulls, show_sector_col=False, show_first_iter_col=True
            )
            print_active_table(
                f"PURE FIRST-TIME SECTORAL INDEX POPUPS (No Prior Stops, Anchor {w_start_s}..{w_end_s}) — BEAR",
                "🏛️🆕🩸", new_idx_bears, show_sector_col=False, show_first_iter_col=True
            )

        print_active_table(
            f"PURE FIRST-TIME STOCK POPUPS (No Prior Same-Dir Stops, Anchor {w_start_s}..{w_end_s}) — BULL ({len(new_stk_bulls)})",
            "🆕🔥", new_stk_bulls, show_sector_col=True, show_first_iter_col=True
        )
        print_active_table(
            f"PURE FIRST-TIME STOCK POPUPS (No Prior Same-Dir Stops, Anchor {w_start_s}..{w_end_s}) — BEAR ({len(new_stk_bears)})",
            "🆕🩸", new_stk_bears, show_sector_col=True, show_first_iter_col=True
        )
        if not new_stk_bulls and not new_stk_bears:
            print(f"\n{COLOR_YELLOW}No pure first-time (non-*) stock popups in window {w_start_s}..{w_end_s}.{COLOR_RESET}")

        if win_idx_bulls or win_idx_bears:
            print_active_table(
                f"ALL WINDOW SECTORAL INDEX POPUPS ({w_start_s}..{w_end_s}, Including * Re-Anchors) — BULL",
                "🏛️⏱️🔥", win_idx_bulls, show_sector_col=False, show_first_iter_col=True
            )
            print_active_table(
                f"ALL WINDOW SECTORAL INDEX POPUPS ({w_start_s}..{w_end_s}, Including * Re-Anchors) — BEAR",
                "🏛️⏱️🩸", win_idx_bears, show_sector_col=False, show_first_iter_col=True
            )

        print_active_table(
            f"ALL WINDOW STOCK POPUPS ({w_start_s}..{w_end_s}, Including * Re-Anchors) — BULL ({len(win_stk_bulls)})",
            "⏱️🔥", win_stk_bulls, show_sector_col=True, show_first_iter_col=True
        )
        print_active_table(
            f"ALL WINDOW STOCK POPUPS ({w_start_s}..{w_end_s}, Including * Re-Anchors) — BEAR ({len(win_stk_bears)})",
            "⏱️🩸", win_stk_bears, show_sector_col=True, show_first_iter_col=True
        )

def build_full_untruncated_plain_text(
    mode, cutoff_dt, checkpoints,
    idx_bulls, idx_bears, idx_grave,
    stock_bulls, stock_bears, stock_grave,
    lookback_meta=None,
    idx_popup_res=None,
    stock_popup_res=None
):
    """
    Generates the 100% untruncated Plain-Text email body containing:
    1. Stage 1: Sectoral Indices (Top Bulls, Top Bears, Graveyard) [Untouched]
    2. Stage 2: Sector-by-Sector Breakdown (Each Sectoral Index followed by its Bull, Bear, and Graveyard stocks) [Untouched]
    3. Stage 3: Complete All-Stocks Master Tables (All Top Bulls, All Top Bears, All Graveyard rows) [Untouched]
    4. Stage 4: New Segregated First-Time Popup Resultset (Lookback Iteration Comparison) [Added]
    """
    e_bulls = stock_bulls if EMAIL_MAX_BUYERS <= 0 else stock_bulls[:EMAIL_MAX_BUYERS]
    e_bears = stock_bears if EMAIL_MAX_SELLERS <= 0 else stock_bears[:EMAIL_MAX_SELLERS]
    e_grave = stock_grave if EMAIL_MAX_GRAVEYARD <= 0 else stock_grave[:EMAIL_MAX_GRAVEYARD]

    all_rows = idx_bulls + idx_bears + idx_grave + e_bulls + e_bears + e_grave
    sym_w = max(MIN_SYMBOL_WIDTH, max((len(r['Symbol']) for r in all_rows), default=MIN_SYMBOL_WIDTH))
    sec_w = max(SECTOR_COL_WIDTH, max((len(r.get('Sector', '')) for r in all_rows), default=SECTOR_COL_WIDTH))
    lines = []

    lines.append(f"=== SYSTEM 3.0 INSTITUTIONAL VOLATILITY TRACKER [{mode}] — FULL UNTRUNCATED REPORT ===")
    lines.append(
        f"Snapshot: {cutoff_dt.strftime('%Y-%m-%d %H:%M')} | Checkpoints: {len(checkpoints)} | "
        f"Indices (Bull: {len(idx_bulls)}, Bear: {len(idx_bears)}, Grave: {len(idx_grave)}) | "
        f"Stocks (Bull: {len(e_bulls)}, Bear: {len(e_bears)}, Grave: {len(e_grave)})"
    )
    lines.append("")

    def _fmt_active_block(title, data_list, show_sector=True, show_first_iter=False):
        if not data_list:
            return
        lines.append(f"--- {title} ({len(data_list)}) ---")
        grid_hdr = " ".join(f"{m}X" for m in GRID_MULTIPLIERS)
        sec_hdr = f"{'Sector (MaxWt%)':<{sec_w}} | " if show_sector else ""
        iter_hdr = f"{'1stIter':^7} {'PopupType':^13} | " if show_first_iter else ""
        hdr = f"{'Script':<{sym_w}} {'LTP/Exit':>9} | {sec_hdr}{iter_hdr}{'Seen':^5} {'Anchor':^7} {'Anch Val':>9} {'R-Mult (Bank)':>15} | ATR: {grid_hdr} | BB: {grid_hdr} | {'State':^13}"
        lines.append(hdr)
        lines.append("-" * len(hdr))
        basket_dir = "BULL" if "BULL" in title else "BEAR"
        default_g = {m: 0 for m in GRID_MULTIPLIERS}
        for r in data_list:
            live_r, banked_r = _r_multiple(r, include_banked=False), float(r.get('BankedR', 0.0) or 0.0)
            if r.get('PriorStops', 0) > 0 and abs(banked_r) > BANKED_R_DISPLAY_EPSILON:
                r_mult_str, anchor_disp = f"{live_r:>+5.2f}R ({banked_r:>+4.1f}R)", f"{r['ActiveAnchor']}*"
            else:
                r_mult_str, anchor_disp = f"{live_r:>+5.2f}R         ", f"{r['ActiveAnchor']} "
            ag, bg = r.get('ATR_Grid', default_g), r.get('BB_Grid', default_g)
            a_str = "  ".join(str(ag.get(k, 0)) for k in GRID_MULTIPLIERS)
            b_str = "  ".join(str(bg.get(k, 0)) for k in GRID_MULTIPLIERS)
            shown_state = _state_display(r)
            if "[ BB > KC ]" in r['State']:
                shown_state = f"[ BB > KC ] {'▲' if basket_dir == 'BULL' else '▼'}"
            ltp_val = f"{r['KilledPrice']:>8.2f}x" if "[STOP" in shown_state and r.get('KilledPrice', 0) > 0 else f"{r['LTP']:>9.2f} "
            sec_str = f"{r.get('Sector', DEFAULT_UNMAPPED_SECTOR)} ({r.get('SectorWeight', 0.0):.1f}%)"
            sec_cell = f"{_clip(sec_str, sec_w, EMAIL_CLIP_SYMBOLS):<{sec_w}} | " if show_sector else ""
            if show_first_iter:
                f_it = r.get('FirstPopupIter', r['CheckpointTime'].strftime('%H:%M'))
                p_type = r.get('PopupType', '1ST-TIME')
                iter_cell = f"{f_it:^7} {p_type:^13} | "
            else:
                iter_cell = ""
            lines.append(
                f"{_clip(r['Symbol'], sym_w, EMAIL_CLIP_SYMBOLS):<{sym_w}} {ltp_val} | {sec_cell}{iter_cell}"
                f"{r['CheckpointTime'].strftime('%H:%M'):^5} {anchor_disp:^7} {r['AnchorPrice']:>9.2f} "
                f"{r_mult_str:>15} |   {a_str}  |   {b_str}  | {shown_state:^13}"
            )
        lines.append("")

    def _fmt_grave_block(title, data_list, show_sector=True):
        if not data_list:
            return
        lines.append(f"--- {title} ({len(data_list)}) ---")
        sec_hdr = f"{'Sector':<{sec_w}} | " if show_sector else ""
        hdr = f"{'Script':<{sym_w}} | {sec_hdr}{'Signal':^6} {'Anchor':^6} {'Anch Val':>9} {'Killed':^6} {'Kill Val':>9} {'LTP':>9} {'R-Mult':>7} {'Move%':>7} | Reason"
        lines.append(hdr)
        lines.append("-" * len(hdr))
        for r in data_list:
            sig = r.get('AnchorDir', 'NONE')
            a_time = r['ActiveAnchor'] if r['ActiveAnchor'] else '-'
            a_price = f"{r['AnchorPrice']:>9.2f}" if r['AnchorPrice'] > 0 else f"{'-':>9}"
            k_time = r['KilledTime'] if r['KilledTime'] else '-'
            k_price = f"{r['KilledPrice']:>9.2f}" if r['KilledPrice'] > 0 else f"{'-':>9}"
            if r['AnchorPrice'] > 0 and (r['KilledPrice'] > 0 or r['LTP'] > 0):
                g_r = _r_multiple(r, include_banked=False)
                g_r_str = f"{g_r:>+6.2f}R"
                move_pct_str = f"{r['MovePct']:>+6.2f}%"
            else:
                g_r_str, move_pct_str = f"{'-':>7}", f"{'-':>7}"
            sec_cell = f"{_clip(r.get('Sector', DEFAULT_UNMAPPED_SECTOR), sec_w, EMAIL_CLIP_SYMBOLS):<{sec_w}} | " if show_sector else ""
            lines.append(
                f"{_clip(r['Symbol'], sym_w, EMAIL_CLIP_SYMBOLS):<{sym_w}} | {sec_cell}{sig:^6} "
                f"{a_time:^6} {a_price} {k_time:^6} {k_price} {r['LTP']:>9.2f} {g_r_str} {move_pct_str} | {r.get('RejectReason', '')}"
            )
        lines.append("")

    # 1. Stage 1: Sectoral Indices (Untouched)
    lines.append("=========================================================================================")
    lines.append("STAGE 1: SECTORAL INDICES (TOP BULLS, TOP BEARS, GRAVEYARD)")
    lines.append("=========================================================================================")
    _fmt_active_block("SECTORAL INDICES — TOP BULL SETUPS", idx_bulls, show_sector=False)
    _fmt_active_block("SECTORAL INDICES — TOP BEAR SETUPS", idx_bears, show_sector=False)
    _fmt_grave_block("SECTORAL INDICES — THE GRAVEYARD", idx_grave, show_sector=False)

    # 2. Stage 2: Per-Sector Stock Grouping (Untouched)
    lines.append("=========================================================================================")
    lines.append("STAGE 2: STOCKS GROUPED BY SECTORAL INDEX (HIGHEST WEIGHTAGE WINNER)")
    lines.append("=========================================================================================")
    all_sec_names = list(SECTORAL_INDEX_CONFIG.keys()) + [DEFAULT_UNMAPPED_SECTOR]
    for sec in all_sec_names:
        sb = [r for r in e_bulls if r.get('Sector') == sec]
        sr = [r for r in e_bears if r.get('Sector') == sec]
        sg = [r for r in e_grave if r.get('Sector') == sec]
        if not sb and not sr and not sg:
            continue
        lines.append(f"*** SECTOR: {sec} (Bull: {len(sb)} | Bear: {len(sr)} | Graveyard: {len(sg)}) ***")
        _fmt_active_block(f"[{sec}] TOP BULL STOCKS", sb, show_sector=False)
        _fmt_active_block(f"[{sec}] TOP BEAR STOCKS", sr, show_sector=False)
        _fmt_grave_block(f"[{sec}] GRAVEYARD STOCKS", sg, show_sector=False)

    # 3. Stage 3: All Stocks Master Tables (Untouched)
    lines.append("=========================================================================================")
    lines.append("STAGE 3: ALL STOCKS MASTER VIEW (DEFAULT SHOW ALL)")
    lines.append("=========================================================================================")
    _fmt_active_block("ALL STOCKS — TOP BULL SETUPS", e_bulls, show_sector=True)
    _fmt_active_block("ALL STOCKS — TOP BEAR SETUPS", e_bears, show_sector=True)
    _fmt_grave_block("ALL STOCKS — THE GRAVEYARD", e_grave, show_sector=True)

    # 4. Stage 4: New Segregated First-Time Popup Resultset (Added)
    if lookback_meta and lookback_meta.get("enabled") and stock_popup_res is not None:
        w_start_s = lookback_meta["window_start_dt"].strftime("%H:%M")
        w_end_s = lookback_meta["window_end_dt"].strftime("%H:%M")
        prior_s = ", ".join(d.strftime("%H:%M") for d in lookback_meta["prior_iter_dts_desc"]) or "None"
        new_idx_b = (idx_popup_res or {}).get("new_bulls", [])
        new_idx_r = (idx_popup_res or {}).get("new_bears", [])
        win_idx_b = (idx_popup_res or {}).get("window_first_bulls", [])
        win_idx_r = (idx_popup_res or {}).get("window_first_bears", [])
        new_stk_b = stock_popup_res.get("new_bulls", [])
        new_stk_r = stock_popup_res.get("new_bears", [])
        win_stk_b = stock_popup_res.get("window_first_bulls", [])
        win_stk_r = stock_popup_res.get("window_first_bears", [])

        lines.append("=========================================================================================")
        lines.append(
            f"STAGE 4 (NEW SEGREGATED RESULTSET): FIRST-TIME POPUPS "
            f"(NO_OF_ITERATIONS={lookback_meta['num_iterations']}, INTERVAL={lookback_meta['interval_min']}m, "
            f"WINDOW={w_start_s}..{w_end_s}, PRIOR=[{prior_s}] vs CURRENT=[{w_end_s}])"
        )
        lines.append("=========================================================================================")
        _fmt_active_block(f"PURE FIRST-TIME SECTORAL INDEX POPUPS ({w_start_s}..{w_end_s}) — BULL", new_idx_b, show_sector=False, show_first_iter=True)
        _fmt_active_block(f"PURE FIRST-TIME SECTORAL INDEX POPUPS ({w_start_s}..{w_end_s}) — BEAR", new_idx_r, show_sector=False, show_first_iter=True)
        _fmt_active_block(f"PURE FIRST-TIME STOCK POPUPS (NO PRIOR SAME-DIR STOPS, ANCHOR {w_start_s}..{w_end_s}) — BULL", new_stk_b, show_sector=True, show_first_iter=True)
        _fmt_active_block(f"PURE FIRST-TIME STOCK POPUPS (NO PRIOR SAME-DIR STOPS, ANCHOR {w_start_s}..{w_end_s}) — BEAR", new_stk_r, show_sector=True, show_first_iter=True)
        _fmt_active_block(f"ALL WINDOW SECTORAL INDEX POPUPS ({w_start_s}..{w_end_s}, INCLUDING * RE-ANCHORS) — BULL", win_idx_b, show_sector=False, show_first_iter=True)
        _fmt_active_block(f"ALL WINDOW SECTORAL INDEX POPUPS ({w_start_s}..{w_end_s}, INCLUDING * RE-ANCHORS) — BEAR", win_idx_r, show_sector=False, show_first_iter=True)
        _fmt_active_block(f"ALL WINDOW STOCK POPUPS ({w_start_s}..{w_end_s}, INCLUDING * RE-ANCHORS) — BULL", win_stk_b, show_sector=True, show_first_iter=True)
        _fmt_active_block(f"ALL WINDOW STOCK POPUPS ({w_start_s}..{w_end_s}, INCLUDING * RE-ANCHORS) — BEAR", win_stk_r, show_sector=True, show_first_iter=True)

    return "\n".join(lines)

def build_email_and_attachment_html(
    mode, cutoff_dt, checkpoints,
    idx_bulls, idx_bears, idx_grave,
    stock_bulls, stock_bears, stock_grave,
    lookback_meta=None,
    idx_popup_res=None,
    stock_popup_res=None
):
    """
    Builds TWO HTML representations:
    1. `email_inline_html`: Uses native HTML5 `<details>` and `<summary>` tags so clicking ANY
       Sectoral Index inside Gmail / Apple Mail / Outlook expands & collapses that sector's
       Top Bulls, Top Bears, and Graveyard stocks WITHOUT needing JavaScript! Also includes
       an open-by-default "SHOW ALL STOCKS & INDICES" master view and the new segregated
       "STAGE 4: FIRST-TIME POPUPS ACROSS LOOKBACK ITERATIONS" section.
    2. `attachment_js_html`: Full standalone HTML dashboard with interactive JS pill buttons,
       click-on-index-row filtering, a "CLEAR FILTER (SHOW ALL)" button, and the new
       segregated First-Time Popup tables.
    """
    e_bulls = stock_bulls if EMAIL_MAX_BUYERS <= 0 else stock_bulls[:EMAIL_MAX_BUYERS]
    e_bears = stock_bears if EMAIL_MAX_SELLERS <= 0 else stock_bears[:EMAIL_MAX_SELLERS]
    e_grave = stock_grave if EMAIL_MAX_GRAVEYARD <= 0 else stock_grave[:EMAIL_MAX_GRAVEYARD]

    def _ser_active(r, basket_dir):
        live_r = _r_multiple(r, include_banked=False)
        banked_r = float(r.get('BankedR', 0.0) or 0.0)
        has_bank = r.get('PriorStops', 0) > 0 and abs(banked_r) > BANKED_R_DISPLAY_EPSILON
        r_str = f"{live_r:+.2f}R ({banked_r:+.1f}R)" if has_bank else f"{live_r:+.2f}R"
        anchor_str = f"{r.get('ActiveAnchor', '-')}*" if has_bank else f"{r.get('ActiveAnchor', '-')}"
        default_g = {m: 0 for m in GRID_MULTIPLIERS}
        ag, bg = r.get('ATR_Grid', default_g), r.get('BB_Grid', default_g)
        atr_s = " ".join(str(ag.get(k, 0)) for k in GRID_MULTIPLIERS)
        bb_s = " ".join(str(bg.get(k, 0)) for k in GRID_MULTIPLIERS)
        st = _state_display(r)
        if "[ BB > KC ]" in r['State']:
            st = f"[ BB > KC ] {'▲' if basket_dir == 'BULL' else '▼'}"
        return {
            "symbol": r['Symbol'],
            "sector": r.get('Sector', DEFAULT_UNMAPPED_SECTOR),
            "weight": f"{float(r.get('SectorWeight', 0.0)):.1f}%",
            "first_iter": r.get('FirstPopupIter', r['CheckpointTime'].strftime('%H:%M')),
            "popup_type": r.get('PopupType', '1ST-TIME'),
            "ltp": f"{r['LTP']:.2f}",
            "seen": r['CheckpointTime'].strftime('%H:%M'),
            "anchor": anchor_str,
            "anchor_val": f"{r['AnchorPrice']:.2f}",
            "r_str": r_str,
            "r_pos": _rank_r_score(r) >= 0,
            "atr_grid": atr_s,
            "bb_grid": bb_s,
            "state": st,
        }

    def _ser_grave(r):
        g_r = _r_multiple(r, include_banked=False) if r['AnchorPrice'] > 0 else 0.0
        return {
            "symbol": r['Symbol'],
            "sector": r.get('Sector', DEFAULT_UNMAPPED_SECTOR),
            "weight": f"{float(r.get('SectorWeight', 0.0)):.1f}%",
            "sig": r.get('AnchorDir', 'NONE'),
            "anchor": r.get('ActiveAnchor') or '-',
            "anchor_val": f"{r['AnchorPrice']:.2f}" if r['AnchorPrice'] > 0 else '-',
            "killed": r.get('KilledTime') or '-',
            "killed_val": f"{r['KilledPrice']:.2f}" if r['KilledPrice'] > 0 else '-',
            "ltp": f"{r['LTP']:.2f}",
            "r_str": f"{g_r:+.2f}R" if r['AnchorPrice'] > 0 else '-',
            "r_pos": g_r >= 0,
            "move_str": f"{r['MovePct']:+.2f}%" if r['AnchorPrice'] > 0 else '-',
            "reason": r.get('RejectReason', ''),
        }

    def _table_active_html(rows_list, basket_dir, show_sec=True, js_clickable_index=False, show_first_iter=False):
        if not rows_list:
            return "<p style='color:#94a3b8;margin:4px 0 10px 0;'>None</p>"
        sec_th = "<th>Sector (Wt%)</th>" if show_sec else ""
        iter_th = "<th>1st Iter</th><th>Popup Type</th>" if show_first_iter else ""
        tr_list = []
        for r in rows_list:
            d = _ser_active(r, basket_dir)
            sec_attr = html.escape(d["symbol"] if not show_sec else d["sector"])
            click_attr = f'onclick="filterBySector(\'{sec_attr}\')" class="idx-row"' if js_clickable_index else ""
            sec_td = f"<td class='sec'>{html.escape(d['sector'])} ({d['weight']})</td>" if show_sec else ""
            iter_td = (
                f"<td align='center' style='color:#38bdf8;font-weight:bold;'>{html.escape(d['first_iter'])}</td>"
                f"<td align='center' style='color:#fde047;'>{html.escape(d['popup_type'])}</td>"
            ) if show_first_iter else ""
            c_cls = "g" if d["r_pos"] else "r"
            tr_list.append(
                f"<tr {click_attr} data-sector='{sec_attr}'>"
                f"<td><b>{html.escape(d['symbol'])}</b></td>"
                f"<td align='right'>{d['ltp']}</td>"
                f"{sec_td}{iter_td}"
                f"<td align='center'>{d['seen']}</td>"
                f"<td align='center'>{html.escape(d['anchor'])}</td>"
                f"<td align='right'>{d['anchor_val']}</td>"
                f"<td align='right' class='{c_cls}'>{html.escape(d['r_str'])}</td>"
                f"<td align='center'>{d['atr_grid']}</td>"
                f"<td align='center'>{d['bb_grid']}</td>"
                f"<td align='center'>{html.escape(d['state'])}</td>"
                f"</tr>"
            )
        return (
            f"<table><thead><tr><th>Script</th><th align='right'>LTP</th>{sec_th}{iter_th}"
            f"<th>Seen</th><th>Anchor</th><th align='right'>AnchVal</th><th align='right'>R-Mult (Bank)</th>"
            f"<th>ATR Grid</th><th>BB Grid</th><th>State</th></tr></thead>"
            f"<tbody>{''.join(tr_list)}</tbody></table>"
        )

    def _table_grave_html(rows_list, show_sec=True, js_clickable_index=False):
        if not rows_list:
            return "<p style='color:#94a3b8;margin:4px 0 10px 0;'>None</p>"
        sec_th = "<th>Sector (Wt%)</th>" if show_sec else ""
        tr_list = []
        for r in rows_list:
            d = _ser_grave(r)
            sec_attr = html.escape(d["symbol"] if not show_sec else d["sector"])
            click_attr = f'onclick="filterBySector(\'{sec_attr}\')" class="idx-row"' if js_clickable_index else ""
            sec_td = f"<td class='sec'>{html.escape(d['sector'])} ({d['weight']})</td>" if show_sec else ""
            s_cls = "g" if d["sig"] == "BULL" else "r" if d["sig"] == "BEAR" else "d"
            c_cls = "g" if d["r_pos"] else "r"
            tr_list.append(
                f"<tr {click_attr} data-sector='{sec_attr}'>"
                f"<td><b>{html.escape(d['symbol'])}</b></td>"
                f"{sec_td}"
                f"<td align='center' class='{s_cls}'>{html.escape(d['sig'])}</td>"
                f"<td align='center'>{html.escape(d['anchor'])}</td>"
                f"<td align='right'>{d['anchor_val']}</td>"
                f"<td align='center'>{html.escape(d['killed'])}</td>"
                f"<td align='right'>{d['killed_val']}</td>"
                f"<td align='right'>{d['ltp']}</td>"
                f"<td align='right' class='{c_cls}'>{html.escape(d['r_str'])}</td>"
                f"<td align='right'>{html.escape(d['move_str'])}</td>"
                f"<td class='rsn'>{html.escape(d['reason'])}</td>"
                f"</tr>"
            )
        return (
            f"<table><thead><tr><th>Script</th>{sec_th}<th>Sig</th><th>Anchor</th>"
            f"<th align='right'>AnchVal</th><th>Killed</th><th align='right'>KillVal</th>"
            f"<th align='right'>LTP</th><th align='right'>R-Mult</th><th align='right'>Move%</th>"
            f"<th>Forensic Exit / Rejection Reason</th></tr></thead>"
            f"<tbody>{''.join(tr_list)}</tbody></table>"
        )

    # Build Native HTML5 <details>/<summary> Accordions for each Sectoral Index (works in Gmail/Apple Mail/Outlook without JS)
    sector_accordions = []
    index_status_lookup = {}
    for r in idx_bulls:
        index_status_lookup[r['Symbol']] = f"🔥 BULL ({_r_multiple(r, include_banked=False):+.2f}R | Anchor {r['ActiveAnchor']})"
    for r in idx_bears:
        index_status_lookup[r['Symbol']] = f"🩸 BEAR ({_r_multiple(r, include_banked=False):+.2f}R | Anchor {r['ActiveAnchor']})"
    for r in idx_grave:
        if r['Symbol'] not in index_status_lookup:
            index_status_lookup[r['Symbol']] = f"🚫 GRAVEYARD ({r.get('RejectReason', 'Stopped')})"

    all_sector_keys = list(SECTORAL_INDEX_CONFIG.keys()) + [DEFAULT_UNMAPPED_SECTOR]
    for sec in all_sector_keys:
        sb = [r for r in e_bulls if r.get('Sector') == sec]
        sr = [r for r in e_bears if r.get('Sector') == sec]
        sg = [r for r in e_grave if r.get('Sector') == sec]
        if not sb and not sr and not sg:
            continue
        idx_badge = index_status_lookup.get(sec, "Sector Group")
        sector_accordions.append(
            f"<details style='margin-bottom:10px;border:1px solid #334155;border-radius:6px;background:#0f172a;padding:8px 12px;'>"
            f"<summary style='cursor:pointer;font-weight:bold;color:#38bdf8;font-size:14px;outline:none;'>"
            f"▶ CLICK TO EXPAND/COLLAPSE: {html.escape(sec)} — [{html.escape(idx_badge)}] "
            f"(Bull Stocks: {len(sb)} | Bear Stocks: {len(sr)} | Graveyard: {len(sg)})"
            f"</summary>"
            f"<div style='padding-top:10px;'>"
            f"<h4>🔥 {html.escape(sec)} — TOP BULL STOCKS ({len(sb)})</h4>"
            f"{_table_active_html(sb, 'BULL', show_sec=False)}"
            f"<h4>🩸 {html.escape(sec)} — TOP BEAR STOCKS ({len(sr)})</h4>"
            f"{_table_active_html(sr, 'BEAR', show_sec=False)}"
            f"<h4>🚫 {html.escape(sec)} — GRAVEYARD STOCKS ({len(sg)})</h4>"
            f"{_table_grave_html(sg, show_sec=False)}"
            f"</div></details>"
        )

    # Build Stage 4 HTML block for First-Time Popups (if enabled)
    stage4_email_html = ""
    stage4_attach_html = ""
    if lookback_meta and lookback_meta.get("enabled") and stock_popup_res is not None:
        w_start_s = lookback_meta["window_start_dt"].strftime("%H:%M")
        w_end_s = lookback_meta["window_end_dt"].strftime("%H:%M")
        prior_s = ", ".join(d.strftime("%H:%M") for d in lookback_meta["prior_iter_dts_desc"]) or "None"
        new_idx_b = (idx_popup_res or {}).get("new_bulls", [])
        new_idx_r = (idx_popup_res or {}).get("new_bears", [])
        win_idx_b = (idx_popup_res or {}).get("window_first_bulls", [])
        win_idx_r = (idx_popup_res or {}).get("window_first_bears", [])
        new_stk_b = stock_popup_res.get("new_bulls", [])
        new_stk_r = stock_popup_res.get("new_bears", [])
        win_stk_b = stock_popup_res.get("window_first_bulls", [])
        win_stk_r = stock_popup_res.get("window_first_bears", [])

        stage4_inner = (
            f"<p style='color:#cbd5e1;margin:4px 0 10px 0;'>"
            f"<b>Lookback Config:</b> NO_OF_ITERATIONS = {lookback_meta['num_iterations']} | "
            f"TIME_OF_INTERVAL_TO_LOOKBACK = {lookback_meta['interval_min']}m | "
            f"<b>Anchor Window:</b> {w_start_s} to {w_end_s} | "
            f"<b>Prior Iterations Compared:</b> [{html.escape(prior_s)}] vs Current [{w_end_s}]</p>"
            f"<h4>🏛️🆕 PURE FIRST-TIME SECTORAL INDEX POPUPS ({w_start_s}..{w_end_s}) — BULL ({len(new_idx_b)})</h4>"
            f"{_table_active_html(new_idx_b, 'BULL', show_sec=False, show_first_iter=True)}"
            f"<h4>🏛️🆕 PURE FIRST-TIME SECTORAL INDEX POPUPS ({w_start_s}..{w_end_s}) — BEAR ({len(new_idx_r)})</h4>"
            f"{_table_active_html(new_idx_r, 'BEAR', show_sec=False, show_first_iter=True)}"
            f"<h4>🆕🔥 PURE FIRST-TIME STOCK POPUPS (No Prior Same-Dir Stops, Anchor {w_start_s}..{w_end_s}) — BULL ({len(new_stk_b)})</h4>"
            f"{_table_active_html(new_stk_b, 'BULL', show_sec=True, show_first_iter=True)}"
            f"<h4>🆕🩸 PURE FIRST-TIME STOCK POPUPS (No Prior Same-Dir Stops, Anchor {w_start_s}..{w_end_s}) — BEAR ({len(new_stk_r)})</h4>"
            f"{_table_active_html(new_stk_r, 'BEAR', show_sec=True, show_first_iter=True)}"
            f"<h4>⏱️🔥 ALL WINDOW STOCK POPUPS ({w_start_s}..{w_end_s}, Including * Re-Anchors) — BULL ({len(win_stk_b)})</h4>"
            f"{_table_active_html(win_stk_b, 'BULL', show_sec=True, show_first_iter=True)}"
            f"<h4>⏱️🩸 ALL WINDOW STOCK POPUPS ({w_start_s}..{w_end_s}, Including * Re-Anchors) — BEAR ({len(win_stk_r)})</h4>"
            f"{_table_active_html(win_stk_r, 'BEAR', show_sec=True, show_first_iter=True)}"
        )

        stage4_email_html = (
            f"<details open style='margin-top:18px;border:2px solid #f59e0b;border-radius:6px;background:#0f172a;padding:10px;'>"
            f"<summary style='cursor:pointer;font-weight:bold;color:#fbbf24;font-size:15px;'>"
            f"🆕 STAGE 4 (NEW SEGREGATED RESULTSET): FIRST-TIME POPUPS ({w_start_s}..{w_end_s} | "
            f"Pure 1st-Time: {len(new_stk_b)}▲ {len(new_stk_r)}▼ | All Window Popups: {len(win_stk_b)}▲ {len(win_stk_r)}▼)"
            f"</summary>"
            f"<div style='padding-top:10px;'>{stage4_inner}</div></details>"
        )
        stage4_attach_html = (
            f"<h3 style='color:#fbbf24;margin-top:20px;'>🆕 STAGE 3 (NEW SEGREGATED RESULTSET): FIRST-TIME POPUPS ({w_start_s}..{w_end_s})</h3>"
            f"<div id='popupContainer'>{stage4_inner}</div>"
        )

    common_css = (
        "body{background:#0b0f19;color:#e2e8f0;font-family:Consolas,monospace;font-size:12px;padding:14px;}"
        "h2,h3,h4{color:#38bdf8;margin:10px 0 6px 0;}"
        "table{width:100%;border-collapse:collapse;margin-bottom:14px;background:#0f172a;}"
        "th{background:#1e293b;color:#38bdf8;padding:6px;border:1px solid #334155;text-align:left;}"
        "td{padding:5px 6px;border:1px solid #1e293b;}"
        ".g{color:#4ade80;font-weight:bold;}.r{color:#f87171;font-weight:bold;}.d{color:#64748b;}"
        ".sec{color:#fde047;}.rsn{color:#fbbf24;}"
        ".idx-row{cursor:pointer;}.idx-row:hover{background:#172554;}"
        ".pill{background:#1e293b;color:#e2e8f0;border:1px solid #475569;padding:5px 10px;margin:2px;border-radius:4px;cursor:pointer;}"
        ".pill:hover,.pill.active{background:#0284c7;color:#fff;}"
        ".clr{background:#ef4444;color:#fff;border:none;padding:6px 14px;border-radius:4px;font-weight:bold;cursor:pointer;}"
    )

    email_inline_html = (
        f"<!DOCTYPE html><html><head><meta charset='utf-8'><style>{common_css}</style></head><body>"
        f"<h2>📡 SYSTEM 3.0 INSTITUTIONAL VOLATILITY TRACKER [{html.escape(mode)}]</h2>"
        f"<p><b>Snapshot:</b> {cutoff_dt.strftime('%Y-%m-%d %H:%M')} | <b>Checkpoints:</b> {len(checkpoints)} | "
        f"<b>Indices:</b> {len(idx_bulls)} Bull, {len(idx_bears)} Bear, {len(idx_grave)} Grave | "
        f"<b>Stocks:</b> {len(e_bulls)} Bull, {len(e_bears)} Bear, {len(e_grave)} Grave</p>"
        f"<h3>🏛️ STAGE 1: SECTORAL INDICES (TOP BULLS, TOP BEARS, GRAVEYARD)</h3>"
        f"<h4>🏛️🔥 SECTORAL INDICES — TOP BULL SETUPS ({len(idx_bulls)})</h4>"
        f"{_table_active_html(idx_bulls, 'BULL', show_sec=False)}"
        f"<h4>🏛️🩸 SECTORAL INDICES — TOP BEAR SETUPS ({len(idx_bears)})</h4>"
        f"{_table_active_html(idx_bears, 'BEAR', show_sec=False)}"
        f"<h4>🏛️🚫 SECTORAL INDICES — THE GRAVEYARD ({len(idx_grave)})</h4>"
        f"{_table_grave_html(idx_grave, show_sec=False)}"
        f"<h3>📂 STAGE 2: CLICK ANY SECTORAL INDEX BELOW TO EXPAND / COLLAPSE ITS STOCKS</h3>"
        f"{''.join(sector_accordions)}"
        f"<details open style='margin-top:18px;border:2px solid #38bdf8;border-radius:6px;background:#0f172a;padding:10px;'>"
        f"<summary style='cursor:pointer;font-weight:bold;color:#4ade80;font-size:15px;'>"
        f"🌐 STAGE 3: DEFAULT FULL VIEW — ALL STOCKS (CLICK TO COLLAPSE / EXPAND ALL)"
        f"</summary>"
        f"<div style='padding-top:10px;'>"
        f"<h4>🔥 ALL STOCKS — TOP BULL SETUPS ({len(e_bulls)})</h4>"
        f"{_table_active_html(e_bulls, 'BULL', show_sec=True)}"
        f"<h4>🩸 ALL STOCKS — TOP BEAR SETUPS ({len(e_bears)})</h4>"
        f"{_table_active_html(e_bears, 'BEAR', show_sec=True)}"
        f"<h4>🚫 ALL STOCKS — THE GRAVEYARD ({len(e_grave)})</h4>"
        f"{_table_grave_html(e_grave, show_sec=True)}"
        f"</div></details>"
        f"{stage4_email_html}"
        f"</body></html>"
    )

    pills_js_html = "".join(
        f"<button class='pill' data-sec='{html.escape(s)}' onclick=\"filterBySector('{html.escape(s)}')\">{html.escape(s)}</button>"
        for s in sorted(SECTORAL_INDEX_CONFIG.keys())
    )

    attachment_js_html = (
        f"<!DOCTYPE html><html><head><meta charset='utf-8'><title>System 3.0 Dashboard</title><style>{common_css}</style></head><body>"
        f"<div style='position:sticky;top:0;background:#111827;padding:10px;border:1px solid #334155;border-radius:6px;margin-bottom:14px;'>"
        f"<div style='display:flex;justify-content:space-between;align-items:center;margin-bottom:8px;'>"
        f"<b>📡 SYSTEM 3.0 [{html.escape(mode)}] — Snapshot: {cutoff_dt.strftime('%Y-%m-%d %H:%M')}</b>"
        f"<div><span id='statusBadge' style='color:#38bdf8;margin-right:10px;font-weight:bold;'>DEFAULT VIEW: SHOWING ALL</span>"
        f"<button class='clr' onclick='clearFilter()'>CLEAR FILTER (SHOW ALL)</button></div></div>"
        f"<div>{pills_js_html}</div></div>"
        f"<h3>🏛️ STAGE 1: SECTORAL INDICES (Click Any Index Row to Filter Stocks Below)</h3>"
        f"<h4>🏛️🔥 SECTORAL INDICES — TOP BULL SETUPS ({len(idx_bulls)})</h4>"
        f"{_table_active_html(idx_bulls, 'BULL', show_sec=False, js_clickable_index=True)}"
        f"<h4>🏛️🩸 SECTORAL INDICES — TOP BEAR SETUPS ({len(idx_bears)})</h4>"
        f"{_table_active_html(idx_bears, 'BEAR', show_sec=False, js_clickable_index=True)}"
        f"<h4>🏛️🚫 SECTORAL INDICES — THE GRAVEYARD ({len(idx_grave)})</h4>"
        f"{_table_grave_html(idx_grave, show_sec=False, js_clickable_index=True)}"
        f"<h3 id='stkHeader'>📈 STAGE 2: ALL STOCKS (Default Show All — Click Any Sector Above to Filter)</h3>"
        f"<div id='stkContainer'>"
        f"<h4>🔥 STOCKS — TOP BULL SETUPS</h4><div id='stkBullWrap'>{_table_active_html(e_bulls, 'BULL', show_sec=True)}</div>"
        f"<h4>🩸 STOCKS — TOP BEAR SETUPS</h4><div id='stkBearWrap'>{_table_active_html(e_bears, 'BEAR', show_sec=True)}</div>"
        f"<h4>🚫 STOCKS — THE GRAVEYARD</h4><div id='stkGraveWrap'>{_table_grave_html(e_grave, show_sec=True)}</div>"
        f"</div>"
        f"{stage4_attach_html}"
        f"<script>"
        f"function filterBySector(sec){{"
        f"document.querySelectorAll('#stkContainer tbody tr, #popupContainer tbody tr').forEach(r=>{{"
        f"r.style.display=(r.getAttribute('data-sector')===sec)?'':'none';}});"
        f"document.querySelectorAll('.pill').forEach(p=>p.classList.toggle('active',p.getAttribute('data-sec')===sec));"
        f"document.getElementById('statusBadge').textContent='FILTERED BY: '+sec;"
        f"document.getElementById('stkHeader').textContent='📈 STAGE 2: STOCKS IN SECTOR ['+sec+']';"
        f"}}"
        f"function clearFilter(){{"
        f"document.querySelectorAll('#stkContainer tbody tr, #popupContainer tbody tr').forEach(r=>{{r.style.display='';}});"
        f"document.querySelectorAll('.pill').forEach(p=>p.classList.remove('active'));"
        f"document.getElementById('statusBadge').textContent='DEFAULT VIEW: SHOWING ALL';"
        f"document.getElementById('stkHeader').textContent='📈 STAGE 2: ALL STOCKS (Default Show All)';"
        f"}}"
        f"</script></body></html>"
    )

    return email_inline_html, attachment_js_html

# ------------------------------------------------------------------------------
# EMAIL TRANSMISSION ENGINE (USES GITHUB REPOSITORY SECRETS)
# ------------------------------------------------------------------------------
def _send_email_message(subject, plain_body, html_body=None, attachment_html=None, tag="email"):
    sender = os.environ.get(ENV_SENDER_EMAIL)
    password = os.environ.get(ENV_SENDER_PASSWORD)
    recipient = os.environ.get(ENV_RECIPIENT_EMAIL)
    smtp_server = os.environ.get(ENV_SMTP_SERVER) or DEFAULT_SMTP_SERVER
    smtp_port = os.environ.get(ENV_SMTP_PORT) or DEFAULT_SMTP_PORT

    missing = [
        name for name, val in (
            (ENV_SENDER_EMAIL, sender),
            (ENV_SENDER_PASSWORD, password),
            (ENV_RECIPIENT_EMAIL, recipient),
        ) if not val
    ]
    if missing:
        print(f"   {COLOR_YELLOW}[{tag}] Cannot send email -- missing secret(s): {', '.join(missing)}{COLOR_RESET}")
        return False

    try:
        port = int(smtp_port)
    except ValueError:
        print(f"   {COLOR_YELLOW}[{tag}] Invalid {ENV_SMTP_PORT}='{smtp_port}'{COLOR_RESET}")
        return False

    msg = EmailMessage()
    msg['Subject'] = subject
    msg['From'] = sender
    msg['To'] = recipient
    msg.set_content(plain_body)

    if html_body:
        msg.add_alternative(html_body, subtype='html')

    if attachment_html and EMAIL_ATTACH_INTERACTIVE_HTML:
        msg.add_attachment(
            attachment_html.encode('utf-8'),
            maintype='text',
            subtype='html',
            filename=EMAIL_ATTACHMENT_FILENAME
        )

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
        print(f"   {COLOR_GREEN_FG}📧 [{tag}] Full untruncated email sent to {recipient} via {smtp_server}:{port}{COLOR_RESET}")
        return True
    except Exception as e:
        print(f"   {COLOR_YELLOW}[{tag}] Failed to send email ({type(e).__name__}: {e}){COLOR_RESET}")
        return False

def print_and_alert_tracked_history(tracked_symbol_raw, by_symbol, live_run=True):
    if not tracked_symbol_raw or tracked_symbol_raw == "***":
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

    sec_name, sec_wt, _ = resolve_primary_sector(matched_sym)
    print(f"\n{COLOR_BOLD}🔍 INTRADAY HISTORY FOR TRACKED SYMBOL: {matched_sym} [Primary Sector: {sec_name} ({sec_wt:.2f}%)]{COLOR_RESET}")
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
            r_mult_str, a_time = f"{live_r:>+5.2f}R         ", f"{row.get('ActiveAnchor', '-')} "
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

    if len(history) >= 2 and live_run and SEND_TRACK_ALERT_EMAIL:
        last_row, prev_row = history[-1], history[-2]
        def get_full_state(r):
            return f"{r.get('AnchorDir', 'NONE')} {r.get('State', 'NONE')}"
        last_state, prev_state = get_full_state(last_row), get_full_state(prev_row)
        if last_state != prev_state:
            _send_email_message(
                f"[System3] {matched_sym} Alert: {prev_state} -> {last_state}",
                f"System3 detected a state transition for {matched_sym} ({sec_name}) at checkpoint {last_row['_EvalTime'].strftime('%H:%M')}.\n\n"
                f"Previous State: {prev_state}\nCurrent State: {last_state}\n\n"
                f"LTP: {last_row.get('LTP', 0):.2f}\nTotal Session Move: {_r_multiple(last_row, include_banked=True):+.2f} R-Multiples\n",
                tag="track"
            )

# ==============================================================================
# MAIN RUNNER
# ==============================================================================
def run_screener(
    mode=TRADING_MODE, days=BACKTRACE_DAYS, target_date_str=None, target_time_str=None,
    track_symbol="", sector_filter="", html_out=HTML_DASHBOARD_PATH, interactive_cli=False,
    lookback_iterations=None, lookback_interval_min=None
):
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

    if not track_symbol:
        track_symbol = os.environ.get(ENV_TRACK_SYMBOL, "").strip()

    lookback_meta = compute_lookback_iteration_schedule(
        cutoff_dt,
        num_iterations=lookback_iterations,
        interval_min=lookback_interval_min
    )

    print(f"\n{COLOR_CYAN}📡 Initializing Tracker [{mode}] | Time Machine: {cutoff_dt.strftime('%Y-%m-%d %H:%M:%S')} (Live: {not explicit}){COLOR_RESET}")
    if lookback_meta["enabled"]:
        prior_str = ", ".join(d.strftime("%H:%M") for d in lookback_meta["prior_iter_dts_desc"]) or "None"
        print(
            f"   {COLOR_DIM}» Lookback Iteration Engine: NO_OF_ITERATIONS={lookback_meta['num_iterations']} | "
            f"TIME_OF_INTERVAL_TO_LOOKBACK={lookback_meta['interval_min']}m | "
            f"Prior Runs: [{prior_str}] | Anchor Window: {lookback_meta['window_start_dt']:%H:%M}..{lookback_meta['window_end_dt']:%H:%M}{COLOR_RESET}"
        )
    for n in notes:
        print(f"   {COLOR_YELLOW}» {n}{COLOR_RESET}")

    needles = [track_symbol] if (track_symbol and track_symbol != "***") else []
    universe_raw = get_dynamic_universe(mode, cutoff_dt, today_session, needles=needles)
    if STATS.auth_failed or not universe_raw:
        return 1
    print(f"   {COLOR_DIM}» Universe: {len(universe_raw)} instruments{COLOR_RESET}")

    if mode == "INDEX_OPTIONS":
        stock_candidates = universe_raw
    else:
        prog = Progress("prefilter", len(universe_raw))
        lookback_cal_days = days * PREFILTER_LOOKBACK_MULT + PREFILTER_LOOKBACK_PAD_DAYS
        with concurrent.futures.ThreadPoolExecutor(max_workers=WORKERS) as ex:
            stock_candidates = [r for r in ex.map(_history_worker_daily, [(it, target_dt - timedelta(days=lookback_cal_days), target_dt, prog) for it in universe_raw]) if r is not None]
        prog.done()
        if STATS.auth_failed or not stock_candidates:
            return 1
        print(f"   {COLOR_DIM}» Prefilter: {len(universe_raw)} -> {len(stock_candidates)} instruments{COLOR_RESET}")

    sector_index_candidates = _sectoral_indices_universe() if (ENABLE_SECTORAL_INDICES and mode != "INDEX_OPTIONS") else []
    all_candidates = sector_index_candidates + stock_candidates

    checkpoints = generate_checkpoints(cutoff_dt) if BACKTRACE_CHECKPOINTS else [cutoff_dt]
    print(f"   {COLOR_DIM}» Backtrace checkpoints ({len(checkpoints)}): {', '.join(c.strftime('%H:%M') for c in checkpoints)}{COLOR_RESET}")

    # Collect any extra checkpoints needed by lookback iterations so they are evaluated in-memory without extra API calls
    extra_lookback_cps = []
    if lookback_meta["enabled"]:
        cp_set = set(checkpoints)
        extra_set = set()
        for it_dt in lookback_meta["prior_iter_dts_chrono"]:
            it_cps = generate_checkpoints(it_dt) if BACKTRACE_CHECKPOINTS else [it_dt]
            for c in it_cps + [it_dt]:
                if c not in cp_set:
                    extra_set.add(c)
        extra_lookback_cps = sorted(extra_set)

    prog = Progress("checkpoints (indices + stocks)", len(all_candidates))
    with concurrent.futures.ThreadPoolExecutor(max_workers=WORKERS) as ex:
        per_symbol = list(ex.map(
            process_stock_checkpoints,
            [(item, checkpoints, extra_lookback_cps, today_session, MIN1_HISTORY_DAYS, mode, prog) for item in all_candidates]
        ))
    prog.done()

    if STATS.auth_failed:
        return 1
    outcomes = [o for sym_outcomes, _ in per_symbol for o in sym_outcomes]
    results = [p for _, s, p in outcomes if s == "ok"]
    extra_results = [p for _, extra_outcomes in per_symbol for _, s, p in extra_outcomes if s == "ok"]
    counts = {k: sum(1 for _, s, _ in outcomes if s == k) for k in ("no_data", "illiquid", "short_history", "error")}
    print(f"   {COLOR_DIM}» Scan summary: {len(all_candidates)} instruments ({len(sector_index_candidates)} indices + {len(stock_candidates)} stocks) x {len(checkpoints)} checkpoints = {len(outcomes)} evaluations | ok {len(results)} | no data {counts['no_data']} | illiquid {counts['illiquid']} | short history {counts['short_history']} | errors {counts['error']}{COLOR_RESET}")
    if not results:
        return 1

    # Primary by_index and by_stock maps (100% untouched for existing resultset)
    by_index, by_stock = {}, {}
    for row in results:
        if row.get('IsSectorIndex'):
            by_index.setdefault(row['Symbol'], []).append(row)
        else:
            by_stock.setdefault(row['Symbol'], []).append(row)

    idx_bulls, idx_bears, idx_grave = segregate_and_rank_buckets(by_index) if by_index else ([], [], [])
    stock_bulls, stock_bears, stock_grave = segregate_and_rank_buckets(by_stock)

    # Build combined in-memory checkpoint map for lookback iterations (includes 09:15 / non-standard step checkpoints)
    by_index_all_cps = {k: list(v) for k, v in by_index.items()}
    by_stock_all_cps = {k: list(v) for k, v in by_stock.items()}
    for row in extra_results:
        if row.get('IsSectorIndex'):
            by_index_all_cps.setdefault(row['Symbol'], []).append(row)
        else:
            by_stock_all_cps.setdefault(row['Symbol'], []).append(row)

    idx_popup_res = evaluate_first_time_popups_in_memory(by_index_all_cps, idx_bulls, idx_bears, lookback_meta) if by_index_all_cps else {
        "new_bulls": [], "new_bears": [], "window_first_bulls": [], "window_first_bears": [], "iter_stats": []
    }
    stock_popup_res = evaluate_first_time_popups_in_memory(by_stock_all_cps, stock_bulls, stock_bears, lookback_meta)

    if track_symbol and track_symbol != "***":
        combined_map = dict(by_index, **by_stock)
        print_and_alert_tracked_history(track_symbol, combined_map, live_run=not explicit)

    # 1. Render Console Output (Existing Stage 1 & Stage 2 untouched + Stage 3 Segregated First-Time Popups added)
    render_console_dashboard(
        mode, cutoff_dt, checkpoints,
        idx_bulls, idx_bears, idx_grave,
        stock_bulls, stock_bears, stock_grave,
        active_sector_filter=sector_filter,
        lookback_meta=lookback_meta,
        idx_popup_res=idx_popup_res,
        stock_popup_res=stock_popup_res
    )

    # 2. Build Untruncated Email Content & Send via SMTP Secrets (if enabled by SEND_DASHBOARD_EMAIL)
    email_inline_html, attachment_js_html = build_email_and_attachment_html(
        mode, cutoff_dt, checkpoints,
        idx_bulls, idx_bears, idx_grave,
        stock_bulls, stock_bears, stock_grave,
        lookback_meta=lookback_meta,
        idx_popup_res=idx_popup_res,
        stock_popup_res=stock_popup_res
    )

    if SAVE_LOCAL_HTML_FILE and html_out:
        try:
            with open(html_out, "w", encoding="utf-8") as f:
                f.write(attachment_js_html)
            print(f"   {COLOR_GREEN_FG}» Local HTML Dashboard saved to: {html_out}{COLOR_RESET}")
        except Exception as e:
            print(f"   {COLOR_YELLOW}» Could not write local HTML file ({e}){COLOR_RESET}")

    if SEND_DASHBOARD_EMAIL:
        plain_text_report = build_full_untruncated_plain_text(
            mode, cutoff_dt, checkpoints,
            idx_bulls, idx_bears, idx_grave,
            stock_bulls, stock_bears, stock_grave,
            lookback_meta=lookback_meta,
            idx_popup_res=idx_popup_res,
            stock_popup_res=stock_popup_res
        )
        new_b_cnt = len(stock_popup_res.get("new_bulls", []))
        new_r_cnt = len(stock_popup_res.get("new_bears", []))
        win_b_cnt = len(stock_popup_res.get("window_first_bulls", []))
        win_r_cnt = len(stock_popup_res.get("window_first_bears", []))
        email_subject = (
            f"[System3 {mode}] {cutoff_dt.strftime('%Y-%m-%d %H:%M')} | "
            f"Indices: {len(idx_bulls)}▲ {len(idx_bears)}▼ | "
            f"Stocks: {len(stock_bulls)}▲ {len(stock_bears)}▼ | "
            f"1stTime: {new_b_cnt}▲ {new_r_cnt}▼ (Win: {win_b_cnt}▲ {win_r_cnt}▼)"
        )
        _send_email_message(
            subject=email_subject,
            plain_body=plain_text_report,
            html_body=email_inline_html,
            attachment_html=attachment_js_html,
            tag="dashboard-email"
        )

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

    # --- Multi-Iteration Lookback Controls ---
    p.add_argument("--no-of-iterations", "--iterations", dest="no_of_iterations", type=int, default=None, help="Number of prior lookback iterations to compare against (default: NO_OF_ITERATIONS in Section 0).")
    p.add_argument("--lookback-interval", "--time-of-interval-to-lookback", dest="lookback_interval", type=int, default=None, help="Lookback interval in minutes between iterations (default: TIME_OF_INTERVAL_TO_LOOKBACK in Section 0).")
    p.add_argument("--no-iteration-lookback", action="store_true", help="Disable multi-iteration first-time popup comparison.")

    # --- Email & Sectoral Controls ---
    p.add_argument("--send-email", dest="send_email", action="store_true", default=SEND_DASHBOARD_EMAIL, help="Send untruncated dashboard email via SMTP secrets.")
    p.add_argument("--no-send-email", dest="send_email", action="store_false", help="Disable sending dashboard email.")
    p.add_argument("--save-html", dest="save_html", action="store_true", default=SAVE_LOCAL_HTML_FILE, help="Save local HTML file to disk.")
    p.add_argument("--no-sector-indices", action="store_true", help="Disable Stage 1 Sectoral Index evaluation.")
    p.add_argument("--sector", type=str, default="", help="Filter console Stock tables to a specific Sectoral Index.")
    p.add_argument("--html-out", type=str, default=HTML_DASHBOARD_PATH)
    p.add_argument("--interactive", action="store_true")

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
    if not get_upstox_token():
        print(f"{COLOR_RED_FG}[!] Missing {ENV_UPSTOX_TOKEN} / {ENV_FALLBACK_TOKEN} environment variable.{COLOR_RESET}")
        sys.exit(1)
    args = parse_args()

    SEND_DASHBOARD_EMAIL = args.send_email
    SAVE_LOCAL_HTML_FILE = args.save_html
    MIN1_HISTORY_DAYS = args.history_days
    OPT_MIN_PRICE = args.opt_min_price
    OPT_MIN_VOLUME = args.opt_min_volume
    EXPIRY_SELECTION = args.expiry
    STRIKES_ABOVE_ATM = args.strikes_up
    STRIKES_BELOW_ATM = args.strikes_down
    CHECKPOINT_INTERVAL_MIN = args.checkpoint_min
    ISOLATE_SESSION_BANDS = args.isolate_session_bands
    ENABLE_SECTORAL_INDICES = not args.no_sector_indices
    ENABLE_ITERATION_LOOKBACK = not args.no_iteration_lookback
    if args.no_of_iterations is not None:
        NO_OF_ITERATIONS = args.no_of_iterations
        LOOKBACK_ITERATIONS = NO_OF_ITERATIONS
    if args.lookback_interval is not None:
        TIME_OF_INTERVAL_TO_LOOKBACK = args.lookback_interval
        LOOKBACK_INTERVAL_MIN = TIME_OF_INTERVAL_TO_LOOKBACK
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
        code = run_screener(
            args.mode, args.days, args.date, args.time,
            track_symbol=args.track,
            sector_filter=args.sector,
            html_out=args.html_out,
            interactive_cli=args.interactive,
            lookback_iterations=args.no_of_iterations,
            lookback_interval_min=args.lookback_interval
        )
    except Exception:
        print(f"\n{COLOR_RED_FG}💥 FATAL: unhandled exception -- full traceback follows.{COLOR_RESET}")
        traceback.print_exc()
        code = 1
    sys.exit(code)
