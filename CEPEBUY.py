#!/usr/bin/env python3
# CEPEBUY.py - Price-based direction: close[i] vs close[i-1]
# Direction = +1 if price went UP, -1 if DOWN. No signal text parsing.

import os
import sys
import logging
import warnings
from datetime import datetime
from pathlib import Path
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import smtplib

import pandas as pd
from collections import defaultdict

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)

SENDER_EMAIL    = os.environ.get('SENDER_EMAIL')
SENDER_PASSWORD = os.environ.get('SENDER_PASSWORD')
RECIPIENT_EMAIL = os.environ.get('RECIPIENT_EMAIL')
SMTP_HOST       = os.environ.get('SMTP_HOST', 'smtp.gmail.com')
SMTP_PORT       = int(os.environ.get('SMTP_PORT', '587'))

ENTRY_CONSEC   = int(os.environ.get('ENTRY_CONSEC', '5'))
ENTRY_CONFIRM  = int(os.environ.get('ENTRY_CONFIRM', '8'))
ENTRY_WINDOW   = int(os.environ.get('ENTRY_WINDOW', '11'))

EXIT_15_CONSEC  = int(os.environ.get('EXIT_15_CONSEC', '5'))
EXIT_15_CONFIRM = int(os.environ.get('EXIT_15_CONFIRM', '10'))
EXIT_15_WINDOW  = int(os.environ.get('EXIT_15_WINDOW', '15'))

EXIT_39_CONSEC  = int(os.environ.get('EXIT_39_CONSEC', '5'))
EXIT_39_CONFIRM = int(os.environ.get('EXIT_39_CONFIRM', '25'))
EXIT_39_WINDOW  = int(os.environ.get('EXIT_39_WINDOW', '39'))

def _find_latest(pattern):
    matches = []; seen = set()
    for root in [Path.cwd(), Path.cwd()/'output', Path(__file__).parent, Path(__file__).parent/'output']:
        if not root.exists(): continue
        for match in root.rglob(pattern):
            ap = str(match.resolve())
            if ap not in seen:
                seen.add(ap)
                try: matches.append((os.path.getmtime(ap), ap))
                except OSError: pass
    if not matches: return None
    matches.sort(reverse=True, key=lambda x: x[0])
    return matches[0][1]

def _load_any(patterns):
    for pat in patterns:
        path = _find_latest(pat)
        if path:
            try:
                df = pd.read_csv(path)
                logger.info('Loaded %s: %d rows', os.path.basename(path), len(df))
                return df
            except Exception: pass
    return pd.DataFrame()

def _direction_from_price(prev_close, curr_close):
    """Compare current close with previous close."""
    if pd.isna(prev_close) or pd.isna(curr_close): return 0
    if curr_close > prev_close: return 1
    if curr_close < prev_close: return -1
    return 0

def check_chain(closes, timestamps, consec, confirm, window, target_dir=None, after_time=None):
    """Find FIRST qualifying chain where price direction matches target_dir."""
    if not closes or len(timestamps) == 0 or len(closes) < 2:
        return None
    
    dirs = []
    for i in range(len(closes)):
        prev = closes[i-1] if i > 0 else closes[i]
        curr = closes[i]
        dirs.append(_direction_from_price(prev, curr))
    
    nzi = [i for i, d in enumerate(dirs) if d != 0]
    nzd = [dirs[i] for i in nzi]
    nzt = [timestamps.iloc[i] for i in nzi]
    
    if len(nzd) < max(consec, window):
        return None
    
    after_ts = pd.to_datetime(after_time, errors='coerce') if after_time else pd.NaT
    
    for i in range(len(nzd) - consec + 1):
        block = nzd[i:i+consec]
        if len(set(block)) == 1 and block[0] != 0:
            direction = block[0]
            if target_dir is not None and direction != target_dir:
                continue
            window_start = max(0, i - (window - consec))
            window_block = nzd[window_start:i+consec]
            if sum(1 for d in window_block if d == direction) >= confirm:
                ts = nzt[i]
                if pd.isna(ts):
                    continue
                if not pd.isna(after_ts) and pd.to_datetime(ts) <= after_ts:
                    continue
                return str(ts)
    return None

def evaluate(iter_df):
    rows = []
    if iter_df.empty: return rows
    cols = {c.lower().strip(): c for c in iter_df.columns}
    def _col(cands):
        for c in cands:
            if c.lower() in cols: return cols[c.lower()]
        return None
    c_underlying = _col(['underlying', 'symbol', 'stock'])
    c_opt_type   = _col(['option type', 'option_type', 'otype', 'type'])
    c_timestamp  = _col(['timestamp', 'time', 'datetime', 'ts'])
    c_close      = _col(['close', 'ltp', 'last'])
    c_strike     = _col(['strike', 'strike_price'])
    c_score      = _col(['current_window_score', 'window_score', 'score'])
    if any(v is None for v in (c_underlying, c_close, c_timestamp)):
        logger.warning('Missing required columns: underlying, close, timestamp')
        return rows
    
    if c_timestamp:
        try:
            iter_df = iter_df.copy()
            with warnings.catch_warnings(): warnings.simplefilter('ignore')
            iter_df[c_timestamp] = pd.to_datetime(iter_df[c_timestamp], errors='coerce')
            iter_df = iter_df.sort_values(c_timestamp)
        except Exception: pass
    
    for (underlying, opt_type), grp in iter_df.groupby([c_underlying, c_opt_type]):
        closes = grp[c_close].tolist()
        timestamps = grp[c_timestamp]
        latest = grp.iloc[-1]
        
        entry_dir = 1 if str(opt_type).strip().upper() == 'CE' else -1
        exit_dir  = -1 if entry_dir == 1 else 1
        
        entry_time  = check_chain(closes, timestamps, ENTRY_CONSEC, ENTRY_CONFIRM, ENTRY_WINDOW, target_dir=entry_dir)
        exit15_time = check_chain(closes, timestamps, EXIT_15_CONSEC, EXIT_15_CONFIRM, EXIT_15_WINDOW, target_dir=exit_dir, after_time=entry_time)
        exit39_time = check_chain(closes, timestamps, EXIT_39_CONSEC, EXIT_39_CONFIRM, EXIT_39_WINDOW, target_dir=exit_dir, after_time=entry_time)
        
        if entry_time:
            rows.append({
                'Stock': underlying,
                'Type': opt_type,
                'Signal': 'BUY CE' if entry_dir == 1 else 'BUY PE',
                'Strike': latest[c_strike] if c_strike else '',
                'LTP': latest[c_close] if c_close else '',
                'Entry': entry_time,
                'Exit15': exit15_time or '-',
                'Exit39': exit39_time or '-',
                'Score': latest[c_score] if c_score else 0,
            })
    return rows

def build_html(rows):
    if not rows: return '<p>No signals today.</p>'
    df = pd.DataFrame(rows)
    out = '<table border="1" cellspacing="0" cellpadding="4" style="border-collapse:collapse;font-family:monospace;font-size:13px;">'
    out += '<tr style="background:#2c3e50;color:white;">' + ''.join('<th>'+c+'</th>' for c in df.columns) + '</tr>'
    for _, r in df.iterrows():
        out += '<tr>' + ''.join('<td>'+str(r[c])+'</td>' for c in df.columns) + '</tr>'
    out += '</table>'
    return out

def send_email(subject, html_body):
    if not all((SENDER_EMAIL, SENDER_PASSWORD, RECIPIENT_EMAIL)):
        logger.warning('Email credentials missing'); return False
    recipients = [a.strip() for a in RECIPIENT_EMAIL.replace(';', ',').split(',') if a.strip()]
    msg = MIMEMultipart('alternative')
    msg['Subject'] = subject; msg['From'] = SENDER_EMAIL; msg['To'] = ','.join(recipients)
    msg.attach(MIMEText(html_body, 'html', 'utf-8'))
    try:
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=30) as s:
            s.starttls(); s.login(SENDER_EMAIL, SENDER_PASSWORD); s.sendmail(SENDER_EMAIL, recipients, msg.as_string())
        logger.info('Email sent: %s', subject); return True
    except Exception as exc: logger.exception('SMTP failed: %s', exc); return False

def main():
    logger.info('=== CEPEBUY (price-based direction) starting ===')
    iter_df = _load_any(['fo_iteration_history_*.csv', 'iteration_history_*.csv', 'fo_iteration_history.csv', 'iteration_history.csv'])
    if iter_df.empty: logger.error('No iteration CSV found'); sys.exit(1)
    rows = evaluate(iter_df)
    if not rows: logger.info('No entry signals today'); return
    html_body = '<html><body style="font-family:sans-serif;">'
    html_body += '<h2>CE/PE Signals - ' + datetime.now().strftime('%d %b %H:%M') + '</h2>'
    html_body += '<p style="color:#666;font-size:12px;">Direction = close[i] vs close[i-1]. Entry: %s consec + %s/%s | 15m Exit: %s consec + %s/%s opposite | 39m Exit: %s consec + %s/%s opposite</p>' % (ENTRY_CONSEC, ENTRY_CONFIRM, ENTRY_WINDOW, EXIT_15_CONSEC, EXIT_15_CONFIRM, EXIT_15_WINDOW, EXIT_39_CONSEC, EXIT_39_CONFIRM, EXIT_39_WINDOW)
    html_body += '<p style="color:#666;font-size:12px;">Entry = FIRST qualifying chain | Exit = first opposite chain AFTER entry</p>'
    html_body += build_html(rows) + '</body></html>'
    send_email('CE/PE Signals ' + datetime.now().strftime('%d %b %H:%M'), html_body)

if __name__ == '__main__':
    main()
