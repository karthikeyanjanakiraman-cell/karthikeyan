
#!/usr/bin/env python3
# CEPEBUY.py — CE/PE Momentum Buy Email with 5-start / 8-of-11 chain logic
# Signal column: window_signal (values like Buy++, Buy+, Sell--, Sell-, Neutral)

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
                logger.info('Loaded daily state: %d CE, %d PE already qualified', len(state.get('ce', {})), len(state.get('pe', {})))
                return state
    except Exception:
        pass
    return {'date': today, 'ce': {}, 'pe': {}}

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

def check_chain(signals):
    if not signals:
        return 0
    dirs = [_direction(s) for s in signals]
    non_zero = [d for d in dirs if d != 0]
    if len(non_zero) < max(CONSEC_START, CONFIRM_WINDOW):
        return 0
    recent = non_zero[-CONSEC_START:]
    if len(set(recent)) == 1 and recent[0] != 0:
        direction = recent[0]
        last_n = non_zero[-CONFIRM_WINDOW:]
        count_same = sum(1 for d in last_n if d == direction)
        if count_same >= CONFIRM_OF:
            return direction
    return 0

def evaluate_chain_from_iteration(iter_df):
    ce_qualify = {}
    pe_qualify = {}
    if iter_df.empty:
        return ce_qualify, pe_qualify
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
    if any(v is None for v in (c_underlying, c_opt_type, c_signal)):
        logger.warning('Iteration CSV missing required columns. Available: %s', list(iter_df.columns))
        return ce_qualify, pe_qualify
    if c_timestamp:
        try:
            iter_df = iter_df.copy()
            iter_df[c_timestamp] = pd.to_datetime(iter_df[c_timestamp], errors='coerce')
            iter_df = iter_df.sort_values(c_timestamp)
        except Exception:
            pass
    for (underlying, opt_type), grp in iter_df.groupby([c_underlying, c_opt_type]):
        signals = grp[c_signal].tolist()
        result = check_chain(signals)
        latest = grp.iloc[-1]
        row_data = {
            'Underlying': underlying,
            'Option Type': str(opt_type).upper(),
            'Strike': latest[c_strike] if c_strike else '',
            'Close': latest[c_close] if c_close else '',
            'Last Signal': latest[c_signal],
            'Timestamp': str(latest[c_timestamp]) if c_timestamp else '',
            'iteration': latest.get('iteration', ''),
        }
        if result == 1 and str(opt_type).upper() == 'CE':
            ce_qualify[underlying] = row_data
        elif result == -1 and str(opt_type).upper() == 'PE':
            pe_qualify[underlying] = row_data
    logger.info('Chain evaluation: CE qualify=%d | PE qualify=%d', len(ce_qualify), len(pe_qualify))
    return ce_qualify, pe_qualify

def evaluate_chain_from_candidates(long_df, short_df):
    ce_qualify = {}
    pe_qualify = {}
    combined = pd.concat([long_df, short_df], ignore_index=True) if (not long_df.empty or not short_df.empty) else pd.DataFrame()
    if combined.empty:
        return ce_qualify, pe_qualify
    cols = {c.lower().strip(): c for c in combined.columns}
    def _col(cands):
        for c in cands:
            if c.lower() in cols:
                return cols[c.lower()]
        return None
    u  = _col(['underlying', 'symbol', 'stock'])
    t  = _col(['option type', 'option_type', 'otype', 'type'])
    cs = _col(['chain signal', 'chain_signal', 'signal', 'window_signal'])
    es = _col(['exit signal', 'exit_signal', 'exit'])
    if any(v is None for v in (u, t, cs, es)):
        logger.warning('Candidates CSV missing columns. Available: %s', list(combined.columns))
        return ce_qualify, pe_qualify
    mask = (
        combined[cs].astype(str).str.contains('ENTER', case=False, na=False) &
        combined[es].astype(str).str.contains('OK HOLD', case=False, na=False)
    )
    filt = combined[mask].copy()
    if filt.empty:
        return ce_qualify, pe_qualify
    for _, row in filt.iterrows():
        underlying = row[u]
        opt_type = str(row[t]).upper()
        if opt_type == 'CE' and underlying not in ce_qualify:
            ce_qualify[underlying] = row.to_dict()
        elif opt_type == 'PE' and underlying not in pe_qualify:
            pe_qualify[underlying] = row.to_dict()
    logger.info('Fallback candidate filter: CE=%d | PE=%d', len(ce_qualify), len(pe_qualify))
    return ce_qualify, pe_qualify

def merge_with_state(state, ce_qualify, pe_qualify):
    for sym, row in ce_qualify.items():
        state['ce'][sym] = row
    for sym, row in pe_qualify.items():
        state['pe'][sym] = row
    logger.info('After state merge: CE=%d | PE=%d (cumulative today)', len(state['ce']), len(state['pe']))
    return state['ce'], state['pe']

def _build_table(rows_dict, title):
    if not rows_dict:
        return '<h3>' + title + '</h3><p>No signals.</p>'
    df = pd.DataFrame(list(rows_dict.values()))
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
    iter_df = _load_any(['fo_iteration_history_*.csv', 'iteration_history_*.csv', 'fo_iteration_history.csv', 'iteration_history.csv'])
    if not iter_df.empty:
        logger.info('Using iteration history with window_signal column.')
        ce_qualify, pe_qualify = evaluate_chain_from_iteration(iter_df)
    else:
        logger.warning('Iteration history not found. Falling back to candidate CSVs.')
        long_df = _load_any(['fo_long_candidates_*.csv', 'long_candidates_*.csv', 'fo_long_candidates.csv', 'long_candidates.csv'])
        short_df = _load_any(['fo_short_candidates_*.csv', 'short_candidates_*.csv', 'fo_short_candidates.csv', 'short_candidates.csv'])
        if long_df.empty and short_df.empty:
            logger.error('No data source available. Abort.')
            sys.exit(1)
        ce_qualify, pe_qualify = evaluate_chain_from_candidates(long_df, short_df)
    ce_today, pe_today = merge_with_state(state, ce_qualify, pe_qualify)
    _save_state(state)
    if not ce_today and not pe_today:
        logger.info('No CE/PE BUY signals today. No email sent.')
        return
    html_body = '<html><body>'
    html_body += '<h2>CE / PE Momentum Buy Report &mdash; ' + datetime.now().strftime('%d %b %Y %H:%M') + '</h2>'
    html_body += '<p><b>Chain rule:</b> ' + str(CONSEC_START) + ' consecutive + ' + str(CONFIRM_OF) + ' of ' + str(CONFIRM_WINDOW) + ' signals in same direction. Mixed chains rejected.</p>'
    html_body += _build_table(ce_today, 'CE BUY (' + str(len(ce_today)) + ' stocks)')
    html_body += '<br/>'
    html_body += _build_table(pe_today, 'PE BUY (' + str(len(pe_today)) + ' stocks)')
    html_body += '</body></html>'
    subject = 'CE / PE Momentum Buy Report - ' + datetime.now().strftime('%d %b %H:%M')
    send_email(subject, html_body)

if __name__ == '__main__':
    main()
