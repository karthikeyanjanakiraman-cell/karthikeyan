
#!/usr/bin/env python3
# CEPEBUY.py - Single table: Entry | Exit15 | Exit39 as columns
# No JSON state. Reads iteration CSV fresh every run.

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
    matches = []
    seen = set()
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

def _direction(signal_str):
    s = str(signal_str).strip().upper()
    if s in ('', 'NAN', 'NONE', 'NEUTRAL', 'HOLD'): return 0
    if s.startswith('BUY') or 'BULL' in s or 'ENTER' in s or 'LONG' in s: return 1
    if s.startswith('SELL') or 'BEAR' in s or 'EXIT' in s or 'SHORT' in s: return -1
    return 0

def check_chain(signals, timestamps, consec, confirm, window):
    if not signals or len(timestamps) == 0: return 0, None
    dirs = [_direction(s) for s in signals]
    non_zero_indices = [i for i, d in enumerate(dirs) if d != 0]
    non_zero_dirs = [dirs[i] for i in non_zero_indices]
    if len(non_zero_dirs) < max(consec, window): return 0, None
    recent = non_zero_dirs[-consec:]
    if len(set(recent)) == 1 and recent[0] != 0:
        direction = recent[0]
        last_n = non_zero_dirs[-window:]
        if sum(1 for d in last_n if d == direction) >= confirm:
            return direction, str(timestamps.iloc[non_zero_indices[-consec]])
    return 0, None

def evaluate_all(df):
    results = {}
    if df.empty: return results
    cols = {c.lower().strip(): c for c in df.columns}
    def _col(cands):
        for c in cands:
            if c.lower() in cols: return cols[c.lower()]
        return None
    c_underlying = _col(['underlying', 'symbol', 'stock'])
    c_opt_type   = _col(['option type', 'option_type', 'otype', 'type'])
    c_signal     = _col(['window_signal', 'window signal', 'signal', 'chain_signal', 'chain signal'])
    c_timestamp  = _col(['timestamp', 'time', 'datetime', 'ts'])
    c_close      = _col(['close', 'ltp', 'last'])
    c_strike     = _col(['strike', 'strike_price'])
    c_score      = _col(['current_window_score', 'window_score', 'score'])
    if any(v is None for v in (c_underlying, c_signal)): return results
    if c_timestamp:
        try:
            df = df.copy()
            with warnings.catch_warnings(): warnings.simplefilter('ignore')
            df[c_timestamp] = pd.to_datetime(df[c_timestamp], errors='coerce')
            df = df.sort_values(c_timestamp)
        except Exception: pass
    for (underlying, opt_type), grp in df.groupby([c_underlying, c_opt_type]):
        signals = grp[c_signal].tolist()
        timestamps = grp[c_timestamp] if c_timestamp else pd.Series([None] * len(signals))
        latest = grp.iloc[-1]
        entry_dir, entry_time = check_chain(signals, timestamps, ENTRY_CONSEC, ENTRY_CONFIRM, ENTRY_WINDOW)
        exit15_dir, exit15_time = check_chain(signals, timestamps, EXIT_15_CONSEC, EXIT_15_CONFIRM, EXIT_15_WINDOW)
        exit39_dir, exit39_time = check_chain(signals, timestamps, EXIT_39_CONSEC, EXIT_39_CONFIRM, EXIT_39_WINDOW)
        if entry_dir != 0 or exit15_dir != 0 or exit39_dir != 0:
            signal_type = ''
            if entry_dir == 1: signal_type = 'BUY CE'
            elif entry_dir == -1: signal_type = 'BUY PE'
            results[(underlying, opt_type)] = {
                'Stock': underlying, 'Type': opt_type,
                'Strike': latest[c_strike] if c_strike else '',
                'LTP': latest[c_close] if c_close else '',
                'Signal': signal_type,
                'Entry': entry_time if entry_time else '',
                'Exit15': exit15_time if exit15_time else '',
                'Exit39': exit39_time if exit39_time else '',
                'Score': latest[c_score] if c_score else 0,
            }
    return results

def build_rows(results):
    rows = list(results.values())
    if not rows: return []
    def sort_key(r):
        has_entry = 1 if r['Entry'] else 0
        return (has_entry, r['Exit15'], r['Exit39'])
    rows.sort(key=sort_key, reverse=True)
    return rows

def build_html(rows):
    if not rows: return '<p>No signals today.</p>'
    df = pd.DataFrame(rows)
    preferred = ['Stock', 'Type', 'Signal', 'Strike', 'LTP', 'Entry', 'Exit15', 'Exit39', 'Score']
    df = df[[c for c in preferred if c in df.columns]]
    out = '<table border="1" cellspacing="0" cellpadding="4" style="border-collapse:collapse;font-family:monospace;font-size:13px;">'
    out += '<tr style="background:#2c3e50;color:white;">' + ''.join('<th>'+c+'</th>' for c in df.columns) + '</tr>'
    for _, r in df.iterrows():
        out += '<tr>' + ''.join('<td>'+(str(r[c]) if pd.notna(r[c]) and str(r[c]) not in ('None','') else '-')+'</td>' for c in df.columns) + '</tr>'
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
    logger.info('=== CEPEBUY starting ===')
    iter_df = _load_any(['fo_iteration_history_*.csv', 'iteration_history_*.csv', 'fo_iteration_history.csv', 'iteration_history.csv'])
    if iter_df.empty: logger.error('No iteration CSV found'); sys.exit(1)
    results = evaluate_all(iter_df)
    rows = build_rows(results)
    if not rows: logger.info('No signals today'); return
    html_body = '<html><body style="font-family:sans-serif;">'
    html_body += '<h2>CE/PE Signals - ' + datetime.now().strftime('%d %b %H:%M') + '</h2>'
    html_body += '<p style="color:#666;font-size:12px;">Entry: %s consec + %s/%s | Exit15: opposite %s/%s | Exit39: opposite %s/%s</p>' % (
        ENTRY_CONSEC, ENTRY_CONFIRM, ENTRY_WINDOW,
        EXIT_15_CONFIRM, EXIT_15_WINDOW,
        EXIT_39_CONFIRM, EXIT_39_WINDOW)
    html_body += '<p style="color:#666;font-size:12px;">Entry = when 5-start triggered | Exit15/Exit39 = when opposite reversal occurred</p>'
    html_body += build_html(rows) + '</body></html>'
    send_email('CE/PE Signals ' + datetime.now().strftime('%d %b %H:%M'), html_body)

if __name__ == '__main__':
    main()
