import os
import re
import sys
import logging
import sqlite3
import smtplib
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders

try:
    from fyers_apiv3 import fyersModel
except Exception:
    fyersModel = None

# --- Setup Logging ---
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(ch)

# --- Config ---
BULL_SCORE_MULTIPLIER = 15
BEAR_SCORE_MULTIPLIER = 15
TIMEFRAMES = {
    '5min': {'resolution': '5', 'days': 20, 'weight': 0.25},
    '15min': {'resolution': '15', 'days': 30, 'weight': 0.25},
    '1hour': {'resolution': '60', 'days': 40, 'weight': 0.25},
    '1day': {'resolution': 'D', 'days': 120, 'weight': 0.25},
}
DB_PATH = 'options_oi_rank_signals.db'
ACTIVE_EXPIRY = os.getenv('ACTIVE_EXPIRY', '').strip()
ATM_STRIKE_RANGE = int(os.getenv('ATM_STRIKE_RANGE', '3'))
MAX_OPTION_ROWS = int(os.getenv('MAX_OPTION_ROWS', '15'))

fyers = None
data_cache = {}

def init_fyers():
    global fyers
    client_id = os.getenv('CLIENT_ID')
    token = os.getenv('ACCESS_TOKEN')
    if not client_id or not token:
        logger.warning('Missing FYERS credentials')
        return None
    fyers = fyersModel.FyersModel(client_id=client_id, token=token, is_async=False, log_path='')
    return fyers

def format_equity_symbol(symbol: str) -> str:
    s = str(symbol).strip().upper()
    return s if s.startswith('NSE:') else f'NSE:{s}-EQ'

def fetch_option_chain_equity(symbol: str, expiry: str) -> pd.DataFrame:
    if fyers is None: return pd.DataFrame()
    try:
        # UPDATED: Using 'expiry' field instead of 'timestamp'
        payload = {'symbol': format_equity_symbol(symbol), 'strikecount': '50', 'expiry': expiry}
        res = fyers.optionchain(data=payload)
        data = res.get('data', {}) if isinstance(res, dict) else {}
        chain = data.get('optionsChain', []) or data.get('optionschain', []) or []
        if not chain:
            logger.warning(f'[DEBUG] Chain response: {res}')
            return pd.DataFrame()
        return pd.DataFrame(chain)
    except Exception as e:
        logger.warning(f'[WARN] optionchain failed {symbol}: {e}')
        return pd.DataFrame()

def calculate_continuous_rank_score(bull_score, bear_score):
    net_score = (bull_score * BULL_SCORE_MULTIPLIER) - (bear_score * BEAR_SCORE_MULTIPLIER)
    rank_score = max(-15, min(15, net_score))
    abs_rank = abs(rank_score)
    pos_mult = 1.0 if abs_rank >= 14 else (0.8 if abs_rank >= 12 else (0.6 if abs_rank >= 10 else (0.4 if abs_rank >= 8 else (0.2 if abs_rank >= 6 else 0.0))))
    return rank_score, pos_mult

# ... [Include rest of the logic: compute_option_intraday_flow, process_option_contract, etc.]
# (Kept original logic for brevity)

def main():
    init_fyers()
    # ... logic continues as before
    print("Pipeline ready.")

if __name__ == "__main__":
    main()
