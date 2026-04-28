import os
import re
import sys
import logging
import sqlite3
import smtplib
from datetime import datetime, timedelta, time
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


class UTF8Formatter(logging.Formatter):
    def format(self, record):
        msg = record.getMessage()
        msg = msg.replace('âŒ', '[ERROR]').replace('âœ…', '[OK]')
        msg = msg.replace('ðŸŸ¢', '[GREEN]').replace('ðŸŸ¡', '[YELLOW]').replace('ðŸ”´', '[RED]')
        msg = msg.replace('âš ï¸', '[WARN]').replace('ðŸ“Š', '[DATA]').replace('ðŸŽ¯', '[TARGET]')
        record.msg = msg
        return super().format(record)


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if logger.hasHandlers():
    logger.handlers.clear()
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(UTF8Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(console_handler)


MARKET_OPEN_TIME = datetime.strptime('09:15', '%H:%M').time()
MARKET_CLOSE_TIME = datetime.strptime('15:30', '%H:%M').time()
AFTERNOON_WINDOW_START = datetime.strptime('13:30', '%H:%M').time()
AFTERNOON_WINDOW_END = datetime.strptime('14:00', '%H:%M').time()
BULL_SCORE_MULTIPLIER = 15
BEAR_SCORE_MULTIPLIER = 15
TIMEFRAMES = {
    '5min': {'resolution': '5', 'days': 20, 'weight': 0.25},
    '15min': {'resolution': '15', 'days': 30, 'weight': 0.25},
    '1hour': {'resolution': '60', 'days': 40, 'weight': 0.25},
    '1day': {'resolution': 'D', 'days': 120, 'weight': 0.25},
}
DB_PATH = 'options_oi_rank_signals.db'
SMTP_HOST = os.getenv('SMTP_HOST', 'smtp.gmail.com')
SMTP_PORT = int(os.getenv('SMTP_PORT', '587'))
SENDER_EMAIL = os.getenv('SENDER_EMAIL', '')
SENDER_PASSWORD = os.getenv('SENDER_PASSWORD', '')
RECIPIENT_EMAIL = os.getenv('RECIPIENT_EMAIL', '')
ACTIVE_EXPIRY = os.getenv('ACTIVE_EXPIRY', '').strip()
ATM_STRIKE_RANGE = int(os.getenv('ATM_STRIKE_RANGE', '3'))
MAX_OPTION_ROWS = int(os.getenv('MAX_OPTION_ROWS', '15'))

fyers = None
data_cache = {}
failed_symbols = []


def init_fyers():
    global fyers
    client_id = os.getenv('CLIENT_ID') or os.getenv('CLIENTID')
    token = os.getenv('ACCESS_TOKEN') or os.getenv('TOKEN') or os.getenv('ACCESSTOKEN')
    if not client_id or not token or fyersModel is None:
        logger.warning('[WARN] Missing FYERS credentials or package unavailable')
        fyers = None
        return None
    fyers = fyersModel.FyersModel(client_id=client_id, token=token, is_async=False, log_path='')
    logger.info('[OK] Fyers API connected successfully')
    return fyers


def load_fno_symbols_from_sectors(root_dir='sectors') -> List[str]:
    symbols = set()
    if not os.path.isdir(root_dir):
        return []
    for dirpath, _, filenames in os.walk(root_dir):
        for fname in filenames:
            if not fname.lower().endswith('.csv'):
                continue
            path = os.path.join(dirpath, fname)
            try:
                df = pd.read_csv(path)
            except Exception:
                continue
            col = next((c for c in df.columns if str(c).strip().lower() in ['symbol', 'symbols', 'ticker']), None)
            if col is None:
                continue
            vals = df[col].dropna().astype(str).str.strip().str.upper()
            symbols.update([v for v in vals if v])
    return sorted(symbols)


def format_equity_symbol(symbol: str) -> str:
    s = str(symbol).strip().upper()
    return s if s.startswith('NSE:') else f'NSE:{s}-EQ'


def get_historical_data(symbol: str, resolution: str, days_back: int) -> Optional[pd.DataFrame]:
    if fyers is None:
        return None
    cache_key = f'{symbol}_{resolution}_{days_back}'
    if cache_key in data_cache:
        return data_cache[cache_key]
    try:
        now = datetime.now()
        start = now - timedelta(days=days_back)
        payload = {
            'symbol': symbol,
            'resolution': resolution,
            'date_format': '1',
            'range_from': start.strftime('%Y-%m-%d'),
            'range_to': now.strftime('%Y-%m-%d'),
            'cont_flag': '1'
        }
        res = fyers.history(data=payload)
        if res.get('s') != 'ok' or 'candles' not in res or not res['candles']:
            return None
        df = pd.DataFrame(res['candles'], columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s').dt.tz_localize('UTC').dt.tz_convert('Asia/Kolkata').dt.tz_localize(None)
        df = df.sort_values('timestamp').drop_duplicates(subset=['timestamp']).reset_index(drop=True)
        data_cache[cache_key] = df
        return df
    except Exception as e:
        logger.warning(f'[WARN] history failed {symbol} {resolution}: {e}')
        return None


def calculate_continuous_rank_score(bull_score, bear_score):
    net_score = (bull_score * BULL_SCORE_MULTIPLIER) - (bear_score * BEAR_SCORE_MULTIPLIER)
    rank_score = max(-15, min(15, net_score))
    abs_rank = abs(rank_score)
    if abs_rank >= 14:
        position_multiplier = 1.0
    elif abs_rank >= 12:
        position_multiplier = 0.80
    elif abs_rank >= 10:
        position_multiplier = 0.60
    elif abs_rank >= 8:
        position_multiplier = 0.40
    elif abs_rank >= 6:
        position_multiplier = 0.20
    else:
        position_multiplier = 0.0
    return rank_score, position_multiplier


def calculate_dynamic_dte_with_decay():
    now = datetime.now()
    market_close = datetime.combine(now.date(), MARKET_CLOSE_TIME)
    if now > market_close:
        market_close = datetime.combine(now.date() + timedelta(days=1), MARKET_CLOSE_TIME)
    time_remaining = market_close - now
    hours_remaining = time_remaining.total_seconds() / 3600
    dte_fraction = max(hours_remaining / 24, 0.001)
    theta_decay_stage = 'SLOW' if hours_remaining > 6 else ('NORMAL' if hours_remaining > 3 else 'FAST')
    theta_risk = 'LOW' if hours_remaining > 6 else ('MEDIUM' if hours_remaining > 2 else 'HIGH')
    return {
        'dte_fraction': dte_fraction,
        'hours_remaining': hours_remaining,
        'theta_decay_stage': theta_decay_stage,
        'theta_risk': theta_risk,
    }


def compute_option_intraday_flow(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    x = df.copy().sort_values('timestamp').reset_index(drop=True)
    close = pd.to_numeric(x['close'], errors='coerce').astype(float)
    volume = pd.to_numeric(x['volume'], errors='coerce').fillna(0).astype(float)
    high = pd.to_numeric(x['high'], errors='coerce').astype(float)
    low = pd.to_numeric(x['low'], errors='coerce').astype(float)
    delta = close.diff().fillna(0)
    direction = delta.apply(lambda v: 1 if v > 0 else (-1 if v < 0 else 0))
    x['Cumulative_OBV'] = (direction * volume).cumsum()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.rolling(14, min_periods=14).mean()
    avg_loss = loss.rolling(14, min_periods=14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    x['RSI'] = (100 - (100 / (1 + rs))).fillna(0)
    typical = (high + low + close) / 3.0
    cum_pv = (typical * volume).cumsum()
    cum_vol = volume.cumsum().replace(0, np.nan)
    x['VWAP'] = (cum_pv / cum_vol).fillna(0)
    return x


def calculate_option_bull_factor_score(row: pd.Series) -> float:
    score = 0.0
    if row.get('PriceChangePct', 0) > 0:
        score += 0.20
    if row.get('OIChangePct', 0) > 0:
        score += 0.25
    if row.get('OBVDelta', 0) > 0:
        score += 0.20
    if row.get('VolumeSurge', 0) >= 1.2:
        score += 0.10
    if row.get('RSI', 50) >= 55:
        score += 0.10
    if row.get('LTP', 0) >= row.get('VWAP', 0):
        score += 0.10
    if row.get('OptionType', '') == 'CE':
        score += 0.05
    return min(score, 1.0)


def calculate_option_bear_factor_score(row: pd.Series) -> float:
    score = 0.0
    if row.get('PriceChangePct', 0) < 0:
        score += 0.20
    if row.get('OIChangePct', 0) > 0:
        score += 0.25
    if row.get('OBVDelta', 0) < 0:
        score += 0.20
    if row.get('VolumeSurge', 0) >= 1.2:
        score += 0.10
    if row.get('RSI', 50) <= 45:
        score += 0.10
    if row.get('LTP', 0) <= row.get('VWAP', 0):
        score += 0.10
    if row.get('OptionType', '') == 'PE':
        score += 0.05
    return min(score, 1.0)


def check_afternoon_sweet_spot():
    now_t = datetime.now().time()
    return AFTERNOON_WINDOW_START <= now_t <= AFTERNOON_WINDOW_END


def nearest_strike_window(strikes: List[float], spot: float, atm_range: int) -> List[float]:
    if not strikes:
        return []
    strikes = sorted(set([float(s) for s in strikes if pd.notna(s)]))
    atm = min(strikes, key=lambda x: abs(x - spot))
    idx = strikes.index(atm)
    lo = max(0, idx - atm_range)
    hi = min(len(strikes), idx + atm_range + 1)
    return strikes[lo:hi]

def fetch_option_chain_equity(symbol: str, expiry: str) -> pd.DataFrame:
    if fyers is None: return pd.DataFrame()
    
    # 1. Try both NSE:SYMBOL-EQ and just SYMBOL (Fyers is inconsistent)
    possible_symbols = [format_equity_symbol(symbol), symbol.strip().upper()]
    
    for sym in possible_symbols:
        try:
            # FYERS V3: 'expiry' must be YYYY-MM-DD
            payload = {'symbol': sym, 'strikecount': '50', 'expiry': expiry}
            res = fyers.optionchain(data=payload)
            
            data = res.get('data', {}) if isinstance(res, dict) else {}
            chain = data.get('optionsChain', []) or data.get('optionschain', []) or []
            
            if chain:
                df = pd.DataFrame(chain)
                df.columns = [c.lower().replace(' ', '_') for c in df.columns]
                return df
            
            # If empty, log the response to see if it's an "Invalid Symbol" error
            logger.info(f'[DEBUG] Chain empty for {sym}. Response: {res.get("message", "No message")}')
            
        except Exception as e:
            logger.warning(f'[WARN] API call failed for {sym}: {e}')
            continue
            
    return pd.DataFrame()


def process_option_contract(option_symbol: str, base_row: pd.Series, timeframe_name: str, resolution: str, days: int) -> Optional[Dict]:
    df = get_historical_data(option_symbol, resolution, days)
    if df is None or len(df) < 10:
        return None
    flow = compute_option_intraday_flow(df)
    if flow.empty:
        return None
    latest = flow.iloc[-1]
    prev = flow.iloc[-2] if len(flow) >= 2 else latest
    session_open = flow['open'].iloc[0]
    ltp = float(latest['close'])
    price_change_pct = ((ltp - session_open) / session_open * 100) if session_open else 0.0
    obv_delta = float(latest['Cumulative_OBV'] - prev['Cumulative_OBV'])
    recent_vol = float(flow['volume'].tail(3).mean()) if len(flow) >= 3 else float(flow['volume'].iloc[-1])
    base_vol = float(flow['volume'].tail(20).mean()) if len(flow) >= 5 else recent_vol
    volume_surge = (recent_vol / base_vol) if base_vol > 0 else 1.0
    oi_val = pd.to_numeric(base_row.get('oi', np.nan), errors='coerce')
    prev_oi_val = pd.to_numeric(base_row.get('prev_oi', np.nan), errors='coerce')
    oi = float(oi_val) if pd.notna(oi_val) else np.nan
    prev_oi = float(prev_oi_val) if pd.notna(prev_oi_val) else np.nan
    oi_change_pct = ((oi - prev_oi) / prev_oi * 100) if pd.notna(oi) and pd.notna(prev_oi) and prev_oi > 0 else 0.0
    row = {
        'Timeframe': timeframe_name,
        'Option_Symbol': option_symbol,
        'OptionType': str(base_row.get('option_type', '')).upper(),
        'Strike': float(pd.to_numeric(base_row.get('strike_price', np.nan), errors='coerce')),
        'LTP': ltp,
        'VWAP': float(latest['VWAP']),
        'RSI': float(latest['RSI']),
        'OBVDelta': obv_delta,
        'PriceChangePct': price_change_pct,
        'VolumeSurge': volume_surge,
        'OI': oi,
        'PrevOI': prev_oi,
        'OIChangePct': oi_change_pct,
        'weight': TIMEFRAMES[timeframe_name]['weight'],
    }
    row['BullScoreTF'] = calculate_option_bull_factor_score(pd.Series(row))
    row['BearScoreTF'] = calculate_option_bear_factor_score(pd.Series(row))
    row['RankScoreTF'], _ = calculate_continuous_rank_score(row['BullScoreTF'], row['BearScoreTF'])
    return row


def process_underlying_options_asit_v30(symbol: str) -> Optional[Dict]:
    logger.info(f'[PROCESS] Starting {symbol}')
    if not ACTIVE_EXPIRY:
        logger.warning('[WARN] ACTIVE_EXPIRY missing')
        return None
    chain = fetch_option_chain_equity(symbol, ACTIVE_EXPIRY)
    if chain.empty:
        logger.warning(f'[PROCESS] {symbol}: empty chain')
        return None
    if 'underlying_ltp' in chain.columns and pd.notna(pd.to_numeric(chain['underlying_ltp'], errors='coerce')).any():
        spot = float(pd.to_numeric(chain['underlying_ltp'], errors='coerce').dropna().iloc[0])
    else:
        eq_df = get_historical_data(format_equity_symbol(symbol), '5', 5)
        if eq_df is None or eq_df.empty:
            return None
        spot = float(eq_df['close'].iloc[-1])
    chain = chain.copy()
    chain['strike_price'] = pd.to_numeric(chain.get('strike_price', np.nan), errors='coerce')
    selected_strikes = nearest_strike_window(chain['strike_price'].dropna().tolist(), spot, ATM_STRIKE_RANGE)
    chain = chain[chain['strike_price'].isin(selected_strikes)].copy()
    if chain.empty:
        return None

    contract_rows = []
    for _, c_row in chain.iterrows():
        option_symbol = str(c_row.get('symbol', '')).strip()
        if not option_symbol:
            continue
        tf_rows = []
        for tf_name, tf_cfg in TIMEFRAMES.items():
            out = process_option_contract(option_symbol, c_row, tf_name, tf_cfg['resolution'], tf_cfg['days'])
            if out is not None:
                tf_rows.append(out)
        if not tf_rows:
            continue
        tf_df = pd.DataFrame(tf_rows)
        total_bull = float((tf_df['BullScoreTF'] * tf_df['weight']).sum())
        total_bear = float((tf_df['BearScoreTF'] * tf_df['weight']).sum())
        rank_score, position_multiplier = calculate_continuous_rank_score(total_bull, total_bear)
        dominant_trend = 'BULLISH' if rank_score > 0 else ('BEARISH' if rank_score < 0 else 'NEUTRAL')
        latest_5m = tf_df[tf_df['Timeframe'] == '5min']
        ref = latest_5m.iloc[0] if not latest_5m.empty else tf_df.iloc[0]
        dte = calculate_dynamic_dte_with_decay()
        contract_rows.append({
            'Underlying': symbol,
            'Option_Symbol': option_symbol,
            'Expiry': ACTIVE_EXPIRY,
            'Strike': ref.get('Strike'),
            'OptionType': ref.get('OptionType'),
            'Spot': spot,
            'LTP': ref.get('LTP'),
            'PriceChangePct': ref.get('PriceChangePct'),
            'OI': ref.get('OI'),
            'PrevOI': ref.get('PrevOI'),
            'OIChangePct': ref.get('OIChangePct'),
            'OBVDelta': ref.get('OBVDelta'),
            'RSI': ref.get('RSI'),
            'VWAP': ref.get('VWAP'),
            'VolumeSurge': ref.get('VolumeSurge'),
            'BullMultiTFScore': total_bull,
            'BearMultiTFScore': total_bear,
            'RankScore15Tier': rank_score,
            'DominantTrend': dominant_trend,
            'PositionSizeMultiplier': position_multiplier,
            'EntryConfidence': abs(rank_score) / 15.0,
            'IsAfternoonSweetSpot': check_afternoon_sweet_spot(),
            'DTEHours': dte['hours_remaining'],
            'ThetaDecayStage': dte['theta_decay_stage'],
            'ThetaRisk': dte['theta_risk'],
            '5min_Score15Tier': float(tf_df.loc[tf_df['Timeframe'] == '5min', 'RankScoreTF'].iloc[0]) if not tf_df[tf_df['Timeframe'] == '5min'].empty else np.nan,
            '15min_Score15Tier': float(tf_df.loc[tf_df['Timeframe'] == '15min', 'RankScoreTF'].iloc[0]) if not tf_df[tf_df['Timeframe'] == '15min'].empty else np.nan,
            '1hour_Score15Tier': float(tf_df.loc[tf_df['Timeframe'] == '1hour', 'RankScoreTF'].iloc[0]) if not tf_df[tf_df['Timeframe'] == '1hour'].empty else np.nan,
            '1day_Score15Tier': float(tf_df.loc[tf_df['Timeframe'] == '1day', 'RankScoreTF'].iloc[0]) if not tf_df[tf_df['Timeframe'] == '1day'].empty else np.nan,
        })
    if not contract_rows:
        return None
    df_contracts = pd.DataFrame(contract_rows)
    df_contracts = df_contracts.sort_values('RankScore15Tier', key=np.abs, ascending=False).reset_index(drop=True)
    return df_contracts.to_dict('records')[0]


def rank_all_underlyings_options_v30(symbols_list: List[str]) -> pd.DataFrame:
    logger.info(f'[RANK] Processing {len(symbols_list)} stocks...')
    results = []
    total = len(symbols_list)
    for idx, sym in enumerate(symbols_list, 1):
        print(f'[{idx}/{total}] Processing {sym}')
        try:
            r = process_underlying_options_asit_v30(sym)
            if r:
                results.append(r)
            else:
                failed_symbols.append(sym)
        except Exception:
            failed_symbols.append(sym)
    if not results:
        return pd.DataFrame()
    df = pd.DataFrame(results)
    df = df.sort_values('RankScore15Tier', key=abs, ascending=False).reset_index(drop=True)
    return df


def init_daily_db():
    today_str = datetime.now().strftime('%Y-%m-%d')
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        'CREATE TABLE IF NOT EXISTS option_signals ('
        'run_id INTEGER PRIMARY KEY AUTOINCREMENT,'
        'date TEXT, runtime TEXT, Underlying TEXT, Option_Symbol TEXT, Expiry TEXT, '
        'Strike REAL, OptionType TEXT, RankScore15Tier REAL, BullMultiTFScore REAL, BearMultiTFScore REAL, '
        'DominantTrend TEXT, PositionSizeMultiplier REAL, EntryConfidence REAL, LTP REAL, OI REAL, PrevOI REAL, OIChangePct REAL)'
    )
    conn.execute('DELETE FROM option_signals WHERE date != ?', (today_str,))
    conn.commit()
    conn.close()


def store_results_in_db(df: pd.DataFrame):
    if df is None or df.empty:
        return
    today_str = datetime.now().strftime('%Y-%m-%d')
    runtime_str = datetime.now().strftime('%H:%M:%S')
    conn = sqlite3.connect(DB_PATH)
    x = df.copy()
    x['date'] = today_str
    x['runtime'] = runtime_str
    cols = ['date', 'runtime', 'Underlying', 'Option_Symbol', 'Expiry', 'Strike', 'OptionType', 'RankScore15Tier', 'BullMultiTFScore', 'BearMultiTFScore', 'DominantTrend', 'PositionSizeMultiplier', 'EntryConfidence', 'LTP', 'OI', 'PrevOI', 'OIChangePct']
    x = x[[c for c in cols if c in x.columns]]
    x.to_sql('option_signals', conn, if_exists='append', index=False)
    conn.commit()
    conn.close()


def build_watchlists(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if df is None or df.empty:
        return pd.DataFrame(), pd.DataFrame()
    bull = df[df['RankScore15Tier'] > 0].sort_values('RankScore15Tier', ascending=False).head(MAX_OPTION_ROWS).copy()
    bear = df[df['RankScore15Tier'] < 0].sort_values('RankScore15Tier', ascending=True).head(MAX_OPTION_ROWS).copy()
    return bull, bear


def build_html_table(df: pd.DataFrame) -> str:
    if df is None or df.empty:
        return '<p>No entries.</p>'
    cols = ['Underlying', 'Option_Symbol', 'Strike', 'OptionType', 'LTP', 'OI', 'PrevOI', 'OIChangePct', 'BullMultiTFScore', 'BearMultiTFScore', 'RankScore15Tier', 'DominantTrend', 'PositionSizeMultiplier']
    y = df[[c for c in cols if c in df.columns]].copy()
    for c in y.columns:
        if pd.api.types.is_numeric_dtype(y[c]):
            y[c] = y[c].map(lambda v: '' if pd.isna(v) else f'{float(v):.2f}')
    return y.to_html(index=False, border=1, justify='center')


def send_email_rank_watchlist(csv_filename: str, bullish_df: pd.DataFrame, bearish_df: pd.DataFrame) -> bool:
    if not all([SENDER_EMAIL, SENDER_PASSWORD, RECIPIENT_EMAIL]):
        logger.warning('[EMAIL] Missing email credentials')
        return False
    try:
        bullish_html = build_html_table(bullish_df)
        bearish_html = build_html_table(bearish_df)
        body = (
            '<html><body style="font-family:Arial,sans-serif;">'
            '<p>Hello,</p>'
            '<p>Please find attached the Options OI Asit-style analysis results.</p>'
            f'<p><b>File:</b> {os.path.basename(csv_filename)}<br><b>Generated:</b> {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>'
            f'<h2>Bullish Watchlist</h2>{bullish_html}'
            f'<h2>Bearish Watchlist</h2>{bearish_html}'
            '<p>PrevOI means previous session close open interest. RankScore15Tier uses multi-timeframe weighted bull/bear scoring.</p>'
            '</body></html>'
        )
        msg = MIMEMultipart()
        msg['From'] = SENDER_EMAIL
        msg['To'] = RECIPIENT_EMAIL
        msg['Subject'] = f'Options OI RankScore Watchlist - {datetime.now().strftime("%Y-%m-%d %H:%M IST")}'
        msg.attach(MIMEText(body, 'html', 'utf-8'))
        with open(csv_filename, 'rb') as f:
            part = MIMEBase('application', 'octet-stream')
            part.set_payload(f.read())
            encoders.encode_base64(part)
            part.add_header('Content-Disposition', f'attachment; filename={os.path.basename(csv_filename)}')
            msg.attach(part)
        if SMTP_PORT == 465:
            with smtplib.SMTP_SSL(SMTP_HOST, SMTP_PORT, timeout=40) as server:
                server.login(SENDER_EMAIL, SENDER_PASSWORD)
                server.send_message(msg)
        else:
            with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=40) as server:
                server.ehlo()
                server.starttls()
                server.ehlo()
                server.login(SENDER_EMAIL, SENDER_PASSWORD)
                server.send_message(msg)
        return True
    except Exception as e:
        logger.warning(f'[EMAIL] send failed: {e}')
        return False


def main():
    logger.info('[LAUNCH] Starting single-file Options OI Asit Rank scanner')
    if not ACTIVE_EXPIRY:
        logger.warning('[WARN] ACTIVE_EXPIRY not set')
    init_fyers()
    init_daily_db()
    symbols = load_fno_symbols_from_sectors('sectors')
    if not symbols:
        logger.warning('[WARN] No symbols found in sectors folder')
        return
    results_df = rank_all_underlyings_options_v30(symbols)
    if results_df.empty:
        logger.warning('[WARN] No results generated')
        return
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    csv_filename = f'options_oi_asit_rank_{timestamp}.csv'
    results_df.to_csv(csv_filename, index=False)
    store_results_in_db(results_df)
    bull_df, bear_df = build_watchlists(results_df)
    send_email_rank_watchlist(csv_filename, bull_df, bear_df)
    print(f'CSV Saved: {csv_filename}')
    print(f'SUMMARY Processed {len(results_df)} symbols | Failed {len(failed_symbols)}')


if __name__ == '__main__':
    main()
