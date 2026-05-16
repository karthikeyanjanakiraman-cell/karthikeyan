#!/usr/bin/env python3
# CEPEBUY.py - Top 20 bullish and bearish from asit*.csv,
# then CE for bullish and PE for bearish from iteration_history*.csv,
# with price-only chain validation near ATM strike.

import os
import sys
import logging
import warnings
from datetime import datetime, timedelta
from pathlib import Path
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import smtplib

import pandas as pd

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)

SENDER_EMAIL = os.environ.get('SENDER_EMAIL')
SENDER_PASSWORD = os.environ.get('SENDER_PASSWORD')
RECIPIENT_EMAIL = os.environ.get('RECIPIENT_EMAIL')
SMTP_HOST = os.environ.get('SMTP_HOST', 'smtp.gmail.com')
SMTP_PORT = int(os.environ.get('SMTP_PORT', '587'))

TOP_N = int(os.environ.get('TOP_N', '20'))

ENTRY_CONSEC = int(os.environ.get('ENTRY_CONSEC', '1'))
ENTRY_CONFIRM = int(os.environ.get('ENTRY_CONFIRM', '1'))
ENTRY_WINDOW = int(os.environ.get('ENTRY_WINDOW', '1'))

EXIT_15_CONSEC = int(os.environ.get('EXIT_15_CONSEC', '1'))
EXIT_15_CONFIRM = int(os.environ.get('EXIT_15_CONFIRM', '1'))
EXIT_15_WINDOW = int(os.environ.get('EXIT_15_WINDOW', '1'))

EXIT_39_CONSEC = int(os.environ.get('EXIT_39_CONSEC', '1'))
EXIT_39_CONFIRM = int(os.environ.get('EXIT_39_CONFIRM', '1'))
EXIT_39_WINDOW = int(os.environ.get('EXIT_39_WINDOW', '1'))

def _find_latest(pattern):
    matches = []
    seen = set()
    for root in [Path.cwd(), Path.cwd() / 'output', Path(__file__).parent, Path(__file__).parent / 'output']:
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
    matches.sort(key=lambda x: x[0], reverse=True)
    return matches[0][1]

def _load_any(patterns):
    for pat in patterns:
        path = _find_latest(pat)
        if path:
            try:
                df = pd.read_csv(path)
                logger.info('Loaded %s: %d rows', os.path.basename(path), len(df))
                return df, path
            except Exception as exc:
                logger.warning('Failed reading %s: %s', path, exc)
    return pd.DataFrame(), None

def _col(df, candidates):
    cols = {c.lower().strip(): c for c in df.columns}
    for c in candidates:
        if c.lower() in cols:
            return cols[c.lower()]
    return None

def _direction_from_price(prev_close, curr_close):
    if pd.isna(prev_close) or pd.isna(curr_close):
        return 0
    if curr_close > prev_close:
        return 1
    if curr_close < prev_close:
        return -1
    return 0

def check_chain(closes, timestamps, consec, confirm, window, target_dir=None, after_time=None):
    if not closes or len(timestamps) == 0 or len(closes) < 2:
        return None, None

    dirs = []
    for i in range(len(closes)):
        prev = closes[i - 1] if i > 0 else closes[i]
        curr = closes[i]
        dirs.append(_direction_from_price(prev, curr))

    nzi = [i for i, d in enumerate(dirs) if d != 0]
    nzd = [dirs[i] for i in nzi]
    nzt = [timestamps.iloc[i] for i in nzi]
    nzp = [closes[i] for i in nzi]

    if len(nzd) < max(consec, window):
        return None, None

    after_ts = pd.to_datetime(after_time, errors='coerce') if after_time else pd.NaT

    for i in range(len(nzd) - consec + 1):
        block = nzd[i:i + consec]
        if len(set(block)) == 1 and block[0] != 0:
            direction = block[0]
            if target_dir is not None and direction != target_dir:
                continue
            window_start = max(0, i - (window - consec))
            window_block = nzd[window_start:i + consec]
            if sum(1 for d in window_block if d == direction) >= confirm:
                ts = nzt[i]
                price = nzp[i]
                if pd.isna(ts):
                    continue
                if not pd.isna(after_ts) and pd.to_datetime(ts) <= after_ts:
                    continue
                return str(ts), price
    return None, None

def eval_exit_chain(window_df, close_col, ts_col, exit_consec, exit_confirm, exit_window, exit_dir, entry_time):
    if window_df.empty or len(window_df) < 2:
        return None, None
    return check_chain(
        window_df[close_col].tolist(),
        window_df[ts_col],
        exit_consec,
        exit_confirm,
        exit_window,
        target_dir=exit_dir,
        after_time=entry_time,
    )

def load_asit_top20():
    df, path = _load_any(['asit*.csv'])
    if df.empty:
        return pd.DataFrame(), None

    c_symbol = _col(df, ['symbol', 'underlying', 'stock'])
    c_rank = _col(df, ['rankscore15tier', 'score', 'rank'])
    c_bull = _col(df, ['bullmultitfscore'])
    c_bear = _col(df, ['bearmultitfscore'])
    c_trend = _col(df, ['dominanttrend'])
    c_can = _col(df, ['cantradetoday'])
    c_ltp = _col(df, ['ltp'])
    if c_symbol is None:
        raise ValueError('asit CSV missing Symbol column')

    work = df.copy()
    if c_can:
        can = work[c_can].astype(str).str.upper().isin(['TRUE', '1', 'YES', 'Y'])
        work = work[can | work[c_can].isna()]

    for c in [c_rank, c_bull, c_bear, c_ltp]:
        if c:
            work[c] = pd.to_numeric(work[c], errors='coerce')

    if c_bull and c_bear:
        bull = work[(work[c_bull] >= work[c_bear])].copy()
        bear = work[(work[c_bear] > work[c_bull])].copy()
    elif c_trend:
        t = work[c_trend].astype(str).str.upper()
        bull = work[t.str.contains('BULL') | t.str.contains('UP')].copy()
        bear = work[t.str.contains('BEAR') | t.str.contains('DOWN')].copy()
    else:
        bull = work.copy()
        bear = work.copy()

    if c_rank:
        bull = bull.sort_values(c_rank, ascending=False)
        bear = bear.sort_values(c_rank, ascending=False)

    bull = bull[[c_symbol] + ([c_rank] if c_rank else [])].drop_duplicates(subset=[c_symbol]).head(TOP_N).copy()
    bear = bear[[c_symbol] + ([c_rank] if c_rank else [])].drop_duplicates(subset=[c_symbol]).head(TOP_N).copy()

    bull['Side'] = 'CE'
    bear['Side'] = 'PE'
    bull.columns = ['Stock'] + ([ 'RankScore15Tier' ] if c_rank else []) + ['Side']
    bear.columns = ['Stock'] + ([ 'RankScore15Tier' ] if c_rank else []) + ['Side']

    out = pd.concat([bull, bear], ignore_index=True)
    logger.info('Top bullish: %d, top bearish: %d', len(bull), len(bear))
    return out, path

def nearest_atm_strike(grp, close_col, strike_col):
    if strike_col is None:
        return None
    strikes = pd.to_numeric(grp[strike_col], errors='coerce').dropna().unique().tolist()
    if not strikes:
        return None
    ltp = pd.to_numeric(grp[close_col], errors='coerce').dropna()
    if ltp.empty:
        return None
    price = float(ltp.iloc[-1])
    return min(strikes, key=lambda x: abs(float(x) - price))

def evaluate(iter_df, top_df):
    rows = []
    if iter_df.empty or top_df.empty:
        return rows

    c_underlying = _col(iter_df, ['underlying', 'symbol', 'stock'])
    c_opt_type = _col(iter_df, ['option type', 'option_type', 'otype', 'type'])
    c_timestamp = _col(iter_df, ['timestamp', 'time', 'datetime', 'ts'])
    c_close = _col(iter_df, ['close', 'ltp', 'last', 'price'])
    c_strike = _col(iter_df, ['strike', 'strike_price'])

    if any(v is None for v in (c_underlying, c_opt_type, c_timestamp, c_close)):
        logger.warning('Missing required columns in iteration history')
        return rows

    iter_df = iter_df.copy()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        iter_df[c_timestamp] = pd.to_datetime(iter_df[c_timestamp], errors='coerce')
    iter_df = iter_df.sort_values(c_timestamp)

    for _, sel in top_df.iterrows():
        stock = str(sel['Stock']).strip()
        side = str(sel['Side']).strip().upper()

        grp = iter_df[iter_df[c_underlying].astype(str).str.upper() == stock.upper()].copy()
        if grp.empty:
            continue

        grp = grp[grp[c_opt_type].astype(str).str.upper() == side].copy()
        if grp.empty:
            continue

        grp = grp.sort_values(c_timestamp)
        latest = grp.iloc[-1]

        atm_strike = nearest_atm_strike(grp, c_close, c_strike)
        if atm_strike is not None and c_strike is not None:
            strike_series = pd.to_numeric(grp[c_strike], errors='coerce')
            sub = grp[strike_series == float(atm_strike)].copy()
            if not sub.empty:
                grp = sub

        closes = pd.to_numeric(grp[c_close], errors='coerce').tolist()
        timestamps = grp[c_timestamp]

        entry_dir = 1 if side == 'CE' else -1
        exit_dir = -1 if entry_dir == 1 else 1

        entry_time, entry_price = check_chain(
            closes, timestamps,
            ENTRY_CONSEC, ENTRY_CONFIRM, ENTRY_WINDOW,
            target_dir=entry_dir
        )
        if not entry_time:
            continue

        entry_dt = pd.to_datetime(entry_time)
        post = grp[grp[c_timestamp] > entry_dt]
        last_ts = grp[c_timestamp].max()

        exit15_time = None
        if pd.notna(last_ts) and (last_ts - entry_dt) >= timedelta(minutes=15):
            w15 = post[post[c_timestamp] > (last_ts - timedelta(minutes=15))]
            exit15_time, _ = eval_exit_chain(
                w15, c_close, c_timestamp,
                EXIT_15_CONSEC, EXIT_15_CONFIRM, EXIT_15_WINDOW,
                exit_dir, entry_time
            )

        exit39_time = None
        if pd.notna(last_ts) and (last_ts - entry_dt) >= timedelta(minutes=39):
            w39 = post[post[c_timestamp] > (last_ts - timedelta(minutes=39))]
            exit39_time, _ = eval_exit_chain(
                w39, c_close, c_timestamp,
                EXIT_39_CONSEC, EXIT_39_CONFIRM, EXIT_39_WINDOW,
                exit_dir, entry_time
            )

        rows.append({
            'Stock': stock,
            'Side': side,
            'Signal': 'BUY CE' if side == 'CE' else 'BUY PE',
            'ATMStrike': atm_strike if atm_strike is not None else '',
            'EntryPrice': round(entry_price, 2) if entry_price is not None else '',
            'LTP': latest[c_close] if c_close in latest else '',
            'Entry': entry_time,
            'Exit15': exit15_time or '-',
            'Exit39': exit39_time or '-',
        })

    return rows

def build_html(rows):
    if not rows:
        return '<p>No signals today.</p>'
    df = pd.DataFrame(rows)
    out = '<table border="1" cellspacing="0" cellpadding="4" style="border-collapse:collapse;font-family:monospace;font-size:13px;">'
    out += '<tr style="background:#2c3e50;color:white;">' + ''.join('<th>' + c + '</th>' for c in df.columns) + '</tr>'
    for _, r in df.iterrows():
        out += '<tr>' + ''.join('<td>' + str(r[c]) + '</td>' for c in df.columns) + '</tr>'
    out += '</table>'
    return out

def send_email(subject, html_body):
    if not all((SENDER_EMAIL, SENDER_PASSWORD, RECIPIENT_EMAIL)):
        logger.warning('Email credentials missing')
        return False
    recipients = [a.strip() for a in RECIPIENT_EMAIL.replace(';', ',').split(',') if a.strip()]
    msg = MIMEMultipart('alternative')
    msg['Subject'] = subject
    msg['From'] = SENDER_EMAIL
    msg['To'] = ','.join(recipients)
    msg.attach(MIMEText(html_body, 'html', 'utf-8'))
    try:
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
    logger.info('=== CEPEBUY starting ===')
    top_df, asit_path = load_asit_top20()
    if top_df.empty:
        logger.error('No bullish/bearish rows found in asit CSV')
        sys.exit(1)

    iter_df, iter_path = _load_any(['iteration_history*.csv', 'fo_iteration_history*.csv'])
    if iter_df.empty:
        logger.error('No iteration history CSV found')
        sys.exit(1)

    rows = evaluate(iter_df, top_df)
    if not rows:
        logger.info('No matching signals')
        return

    html_body = '<html><body style="font-family:sans-serif;">'
    html_body += '<h2>CE/PE Near ATM Signals</h2>'
    html_body += '<p style="color:#666;font-size:12px;">Top 20 bullish names from asit*.csv are mapped to CE, top 20 bearish names to PE. Chain validation uses price only, and ATM is the strike nearest to LTP.</p>'
    html_body += build_html(rows) + '</body></html>'
    send_email('CE/PE Near ATM Signals ' + datetime.now().strftime('%d %b %H:%M'), html_body)

if __name__ == '__main__':
    main()
