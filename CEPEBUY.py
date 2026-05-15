
#!/usr/bin/env python3
# CEPEBUY.py — CE / PE Buy Momentum: 5‑start timestamp, CE = Bullish, PE = Bearish
# Only 1 row per stock in email, sorted by qualify count + latest qualifying time

import os
import sys
import json
import smtplib
import logging
from datetime import datetime, date
from pathlib import Path
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

import pandas as pd
from collections import Counter, defaultdict

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)

SENDER_EMAIL    = os.environ.get('SENDER_EMAIL')
SENDER_PASSWORD = os.environ.get('SENDER_PASSWORD')
RECIPIENT_EMAIL = os.environ.get('RECIPIENT_EMAIL')
SMTP_HOST       = os.environ.get('SMTP_HOST', 'smtp.gmail.com')
SMTP_PORT       = int(os.environ.get('SMTP_PORT', '587'))

CONSEC_START   = int(os.environ.get('CONSEC_START', '5'))
CONFIRM_OF     = int(os.environ.get('CONFIRM_OF', '8'))
CONFIRM_WINDOW = int(os.environ.get('CONFIRM_WINDOW', '11'))
STATE_FILE     = os.environ.get('CEPE_STATE_FILE', '/tmp/cepebuy_state.json')

def _search_roots():
    roots = set()
    roots.add(Path.cwd().resolve())
    roots.add((Path.cwd() / 'output').resolve())
    try:
        sd = Path(__file__).parent.resolve()
        roots.add(sd)
        roots.add((sd / 'output').resolve())
        for p in sd.parents[:4]:
            roots.add(p.resolve())
            roots.add((p / 'output').resolve())
    except Exception:
        pass
    for ev in ('GITHUB_WORKSPACE', 'RUNNER_WORKSPACE', 'HOME', 'CI_WORKSPACE'):
        v = os.environ.get(ev)
        if v:
            roots.add(Path(v).resolve())
            roots.add((Path(v) / 'output').resolve())
    for fb in ('/github/workspace', '/home/runner/work', '/workspace', '/mnt/data', '/tmp'):
        if os.path.isdir(fb):
            roots.add(Path(fb).resolve())
    return roots

def _find_latest(pattern):
    matches = []
    seen = set()
    for root in _search_roots():
        if not root.exists():
            continue
        for match in root.rglob(pattern):
            ap = str(match.resolve())
            if ap in seen:
                continue
            seen.add(ap)
            try:
                matches.append((os.path.getmtime(ap), ap))
            except OSError:
                pass
    if not matches:
        return None
    matches.sort(reverse=True, key=lambda x: x[0])
    best = matches[0][1]
    logger.info('Selected %s (mtime=%s)', best, datetime.fromtimestamp(matches[0][0]).strftime('%Y-%m-%d %H:%M:%S'))
    return best

def _load_any(patterns):
    for pat in patterns:
        path = _find_latest(pat)
        if not path:
            continue
        try:
            df = pd.read_csv(path)
            logger.info('Loaded %s: %d rows x %d cols', os.path.basename(path), len(df), len(df.columns))
            return df
        except Exception as exc:
            logger.exception('Failed to read %s: %s', path, exc)
    return pd.DataFrame()

def _load_state():
    today = str(date.today())
    try:
        if os.path.exists(STATE_FILE):
            with open(STATE_FILE, 'r') as f:
                state = json.load(f)
            if state.get('date') == today:
                logger.info('Loaded daily state: %d CE Buy, %d PE Buy', 
                          len(state.get('ce_all', [])), len(state.get('pe_all', [])))
                return state
    except Exception:
        pass
    return {'date': today, 'ce_all': [], 'pe_all': []}

def _save_state(state):
    try:
        with open(STATE_FILE, 'w') as f:
            json.dump(state, f, indent=2, default=str)
    except Exception as exc:
        logger.warning('Could not save state: %s', exc)

def _direction(signal_str):
    s = str(signal_str).strip().upper()
    if s in ('', 'NAN', 'NONE', 'NEUTRAL', 'HOLD'):
        return 0
    if s.startswith('BUY') or 'BULL' in s or 'ENTER' in s or 'LONG' in s or 'CONFIRM' in s:
        return 1
    if s.startswith('SELL') or 'BEAR' in s or 'EXIT' in s or 'SHORT' in s:
        return -1
    return 0

def check_chain(signals, timestamps):
    if not signals or not timestamps:
        return 0, None
    dirs = [_direction(s) for s in signals]
    non_zero = [d for d in dirs if d != 0]
    if len(non_zero) < max(CONSEC_START, CONFIRM_WINDOW):
        return 0, None

    recent = non_zero[-CONSEC_START:]
    if len(set(recent)) == 1 and recent[0] != 0:
        direction = recent[0]
        last_n = non_zero[-CONFIRM_WINDOW:]
        count_same = sum(1 for d in last_n if d == direction)
        if count_same >= CONFIRM_OF:
            # 5‑consecutive start time (5‑start)
            start_idx = -CONSEC_START
            if 0 <= start_idx < len(timestamps):
                five_start = str(timestamps.iloc[start_idx])
            else:
                five_start = str(timestamps.iloc[0])
            return direction, five_start
    return 0, None

def keep_sorted_per_stock(rows):
    if not rows:
        return []
    per_stock = defaultdict(list)
    for r in rows:
        per_stock[r["Underlying"]].append(r)

    result = []
    for stock, rows in per_stock.items():
        # Latest qualifying bar for this stock (for freshness, not for timestamp in table)
        latest = max(rows, key=lambda x: x["Timestamp"])
        # Number of times this stock qualified today
        latest["Qualify Count"] = len(rows)
        result.append(latest)

    # Sort by: 1) count, 2) latest 5‑start time (most active first)
    result.sort(key=lambda x: (x["Qualify Count"], x["Timestamp"]), reverse=True)
    return result

def evaluate_chain_from_iteration(iter_df):
    all_ce_qualified = []
    all_pe_qualified = []

    if iter_df.empty:
        return all_ce_qualified, all_pe_qualified

    cols = {c.lower().strip(): c for c in iter_df.columns}

    def _col(cands):
        for c in cands:
            if c.lower() in cols:
                return cols[c.lower()]
        return None

    c_underlying = _col(['underlying', 'symbol', 'stock'])
    c_opt_type   = _col(['option type', 'option_type', 'otype', 'type'])
    c_signal     = _col(['window_signal', 'window signal', 'signal', 'chain_signal', 'chain signal'])
    c_timestamp  = _col(['timestamp', 'time', 'datetime', 'ts'])
    c_close      = _col(['close', 'ltp', 'last'])
    c_strike     = _col(['strike', 'strike_price'])
    c_window_score = _col(['current_window_score', 'window_score', 'score'])

    if any(v is None for v in (c_underlying, c_signal)):
        logger.warning('Iteration CSV missing required columns. Available: %s', list(iter_df.columns))
        return all_ce_qualified, all_pe_qualified

    if c_timestamp:
        try:
            iter_df = iter_df.copy()
            iter_df[c_timestamp] = pd.to_datetime(iter_df[c_timestamp], errors='coerce')
            iter_df = iter_df.sort_values(c_timestamp)
        except Exception:
            pass

    for (underlying, opt_type), grp in iter_df.groupby([c_underlying, c_opt_type]):
        signals = grp[c_signal].tolist()
        if c_timestamp:
            timestamps = grp[c_timestamp]
        else:
            timestamps = pd.Series([None] * len(signals))

        result, five_start_time = check_chain(signals, timestamps)
        latest = grp.iloc[-1]

        # Bullish 5‑chain → CE Buy
        if result == 1:
            row_data = {
                "Underlying": underlying,
                "Option Type": "CE",
                "Direction": "Bullish",
                "Strike": latest[c_strike] if c_strike else "",
                "Close": latest[c_close] if c_close else "",
                "Last Signal": latest[c_signal],
                "Timestamp": five_start_time,
                "iteration": latest.get("iteration", ""),
                "Window Score": latest[c_window_score] if c_window_score else 0,
            }
            all_ce_qualified.append(row_data)

        # Bearish 5‑chain → PE Buy
        elif result == -1:
            row_data = {
                "Underlying": underlying,
                "Option Type": "PE",
                "Direction": "Bearish",
                "Strike": latest[c_strike] if c_strike else "",
                "Close": latest[c_close] if c_close else "",
                "Last Signal": latest[c_signal],
                "Timestamp": five_start_time,
                "iteration": latest.get("iteration", ""),
                "Window Score": -latest[c_window_score] if c_window_score else 0,
            }
            all_pe_qualified.append(row_data)

    logger.info('CE Buy qualified total (all timestamps): %d', len(all_ce_qualified))
    logger.info('PE Buy qualified total (all timestamps): %d', len(all_pe_qualified))
    return all_ce_qualified, all_pe_qualified

def evaluate_chain_from_candidates(long_df, short_df):
    ce_all = []
    pe_all = []

    combined = pd.concat([long_df, short_df], ignore_index=True) if (not long_df.empty or not short_df.empty) else pd.DataFrame()
    if combined.empty:
        return ce_all, pe_all

    cols = {c.lower().strip(): c for c in combined.columns}

    def _col(cands):
        for c in cands:
            if c.lower() in cols:
                return cols[c.lower()]
        return None

    u  = _col(['underlying', 'symbol', 'stock'])
    cs = _col(['chain signal', 'chain_signal', 'signal', 'window_signal'])
    ts = _col(['timestamp', 'time', 'datetime'])

    if any(v is None for v in (u, cs)):
        logger.warning('Candidates CSV missing columns. Available: %s', list(combined.columns))
        return ce_all, pe_all

    # Bullish entries → CE Buy
    bull_mask = (
        combined[cs].astype(str).str.contains('ENTER', case=False, na=False) &
        combined[cs].astype(str).str.contains('BUY', case=False, na=False)
    )
    bull_df = combined[bull_mask].copy()
    for _, row in bull_df.iterrows():
        row_data = {
            "Underlying": row[u],
            "Option Type": "CE",
            "Direction": "Bullish",
            "Strike": row.get('strike', ''),
            "Close": row.get('close', ''),
            "Last Signal": row[cs],
            "Timestamp": str(row.get(ts, '')),
            "iteration": row.get('iteration', ''),
            "Window Score": row.get('window_score', 0),
        }
        ce_all.append(row_data)

    # Bearish entries → PE Buy
    bear_mask = (
        combined[cs].astype(str).str.contains('ENTER', case=False, na=False) &
        combined[cs].astype(str).str.contains('SELL', case=False, na=False)
    )
    bear_df = combined[bear_mask].copy()
    for _, row in bear_df.iterrows():
        row_data = {
            "Underlying": row[u],
            "Option Type": "PE",
            "Direction": "Bearish",
            "Strike": row.get('strike', ''),
            "Close": row.get('close', ''),
            "Last Signal": row[cs],
            "Timestamp": str(row.get(ts, '')),
            "iteration": row.get('iteration', ''),
            "Window Score": -row.get('window_score', 0),
        }
        pe_all.append(row_data)

    logger.info('Fallback CE Buy entries: %d', len(ce_all))
    logger.info('Fallback PE Buy entries: %d', len(pe_all))
    return ce_all, pe_all

def _build_table(rows, title):
    if not rows:
        return '<h3>' + title + '</h3><p>No signals.</p>'
    df = pd.DataFrame(rows)
    out = '<h3>' + title + '</h3>'
    out += '<table border="1" cellspacing="0" cellpadding="4" style="border-collapse:collapse;">'
    out += '<tr style="background:#dce8f7;">' + ''.join('<th>' + str(c) + '</th>' for c in df.columns) + '</tr>'
    for _, r in df.iterrows():
        out += '<tr>' + ''.join('<td>' + str(r[c]) + '</td>' for c in df.columns) + '</tr>'
    out += '</table>'
    return out

def send_email(subject, html_body):
    if not all((SENDER_EMAIL, SENDER_PASSWORD, RECIPIENT_EMAIL)):
        logger.warning('Email credentials missing; skipping.')
        return False
    recipients = [a.strip() for a in RECIPIENT_EMAIL.replace(';', ',').split(',') if a.strip()]
    if not recipients:
        return False
    msg = MIMEMultipart('alternative')
    msg['Subject'] = subject
    msg['From'] = SENDER_EMAIL
    msg['To'] = ','.join(recipients)
    msg.attach(MIMEText(html_body, 'html', 'utf-8'))
    try:
        logger.info('Connecting %s:%d ...', SMTP_HOST, SMTP_PORT)
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=30) as s:
            s.starttls()
            s.login(SENDER_EMAIL, SENDER_PASSWORD)
            s.sendmail(SENDER_EMAIL, recipients, msg.as_string())
        logger.info('Email sent: %s', subject)
        return True
    except Exception as exc:
        logger.exception('SMTP failed: %s', exc)
        return False

def main():
    logger.info('=== CEPEBUY starting (5-start / 8-of-11 chain logic) ===')
    state = _load_state()

    iter_df = _load_any([
        'fo_iteration_history_*.csv',
        'iteration_history_*.csv',
        'fo_iteration_history.csv',
        'iteration_history.csv',
    ])
    if not iter_df.empty:
        logger.info('Using iteration history with window_signal column.')
        ce_all, pe_all = evaluate_chain_from_iteration(iter_df)
    else:
        logger.warning('Iteration history not found. Falling back to candidate CSVs.')
        long_df = _load_any(['fo_long_candidates_*.csv', 'long_candidates_*.csv', 'fo_long_candidates.csv', 'long_candidates.csv'])
        short_df = _load_any(['fo_short_candidates_*.csv', 'short_candidates_*.csv', 'fo_short_candidates.csv', 'short_candidates.csv'])
        if long_df.empty and short_df.empty:
            logger.error('No data source available. Abort.')
            sys.exit(1)
        ce_all, pe_all = evaluate_chain_from_candidates(long_df, short_df)

    # Attach qualify count separately (for sorting)
    ce_count = Counter(r["Underlying"] for r in ce_all)
    pe_count = Counter(r["Underlying"] for r in pe_all)
    for r in ce_all:
        r["Qualify Count"] = ce_count[r["Underlying"]]
    for r in pe_all:
        r["Qualify Count"] = pe_count[r["Underlying"]]
    
    state['ce_all'] = ce_all
    state['pe_all'] = pe_all
    _save_state(state)

    # Only 1 row per stock, sorted by count + 5‑start timestamp
    ce_today = keep_sorted_per_stock(ce_all)
    pe_today = keep_sorted_per_stock(pe_all)

    if not ce_today and not pe_today:
        logger.info('No CE/PE Buy signals today. No email sent.')
        return

    html_body = '<html><body>'
    html_body += '<h2>CE / PE Buy Momentum Report &mdash; ' + datetime.now().strftime('%d %b %Y %H:%M') + '</h2>'
    html_body += '<p><b>Chain rule:</b> ' + str(CONSEC_START) + ' consecutive + ' + str(CONFIRM_OF) + ' of ' + str(CONFIRM_WINDOW) + ' signals in same direction. Mixed chains rejected.</p>'
    html_body += '<p><b>Logic:</b> Bullish 5‑start chains → CE Buy (bullish momentum), Bearish 5‑start chains → PE Buy (bearish momentum).</p>'
    html_body += '<p><b>Timestamp:</b> Time when the 5‑consecutive 5‑start chain was first completed, not the current run time.</p>'
    
    if ce_today:
        html_body += _build_table(ce_today, 'CE Buy (bullish 5‑start)')
        html_body += '<br/>'
    
    if pe_today:
        html_body += _build_table(pe_today, 'PE Buy (bearish 5‑start)')
    
    html_body += '</body></html>'

    subject = 'CE / PE Buy Momentum Report - ' + datetime.now().strftime('%d %b %H:%M')
    send_email(subject, html_body)

if __name__ == '__main__':
    main()
