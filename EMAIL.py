import configparser
import os
import sys
import sqlite3
import smtplib
import logging
from datetime import datetime, timedelta
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders

import numpy as np
import pandas as pd
import ta
from scipy.stats import percentileofscore, norm
from fyers_apiv3 import fyersModel

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

class UTF8Formatter(logging.Formatter):
    def format(self, record):
        msg = record.getMessage()
        msg = msg.replace('âŒ', '[ERROR]').replace('âœ…', '[OK]')
        msg = msg.replace('ðŸŸ¢', '[GREEN]').replace('ðŸŸ¡', '[YELLOW]').replace('ðŸ”´', '[RED]')
        msg = msg.replace('âš ï¸', '[WARN]').replace('ðŸ“Š', '[DATA]').replace('ðŸŽ¯', '[TARGET]')
        record.msg = msg
        return super().format(record)

log_format = '%(asctime)s - %(levelname)s - %(message)s'
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    fh = logging.FileHandler('strategy_v3_0.log', encoding='utf-8')
    fh.setFormatter(UTF8Formatter(log_format))
    logger.addHandler(fh)
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(UTF8Formatter(log_format))
    logger.addHandler(ch)

config = configparser.ConfigParser()
config.read('config.ini')

def get_cfg(section, key, env_name=None, default=None, is_int=False):
    if env_name:
        val = os.getenv(env_name)
        if val not in (None, ''):
            return int(val) if is_int else val
    if section and key and config.has_option(section, key):
        val = config.get(section, key)
        return int(val) if is_int else val
    return default

try:
    client_id = get_cfg('fyers_credentials', 'client_id', env_name='CLIENT_ID')
    token = get_cfg('fyers_credentials', 'access_token', env_name='ACCESS_TOKEN') or get_cfg('fyers_credentials', 'token', env_name='TOKEN')
    if not client_id or not token:
        raise ValueError('Missing CLIENT_ID or ACCESS_TOKEN')
    fyers = fyersModel.FyersModel(client_id=client_id, token=token)
    logger.info('[OK] Fyers API connected successfully')
except Exception as e:
    logger.warning(f'[WARN] Fyers auth error: {e}')
    fyers = None

DATA_CACHE = {}
FAILED_SYMBOLS = []
ALL_RESULTS = []
DB_PATH = 'intraday_signals.db'
MARKET_OPEN_TIME = datetime.strptime('09:15', '%H:%M').time()
MARKET_CLOSE_TIME = datetime.strptime('15:30', '%H:%M').time()
AFTERNOON_WINDOW_START = datetime.strptime('13:30', '%H:%M').time()
AFTERNOON_WINDOW_END = datetime.strptime('14:00', '%H:%M').time()
RISK_FREE_RATE = 0.06
MIN_DELTA_TARGET = 0.30
MAX_DELTA_TARGET = 0.60
DAILY_PROFIT_TARGET = 100000
MAX_DAILY_LOSS = 50000
MAX_STRIKE_DISTANCE_FROM_LTP = 2.5
TIMEFRAMES = {
    '5min': {'resolution': '5', 'days': 30, 'weight': 0.10},
    '15min': {'resolution': '15', 'days': 50, 'weight': 0.20},
    '1hour': {'resolution': '60', 'days': 50, 'weight': 0.25},
    '4hour': {'resolution': '240', 'days': 50, 'weight': 0.20},
    '1day': {'resolution': 'D', 'days': 365, 'weight': 0.25},
}

def calculate_dynamic_dte_with_decay():
    now = datetime.now()
    market_close = datetime.combine(now.date(), MARKET_CLOSE_TIME)
    if now > market_close:
        market_close = datetime.combine(now.date() + timedelta(days=1), MARKET_CLOSE_TIME)
    rem = market_close - now
    hours_remaining = rem.total_seconds() / 3600
    return {
        'dte_fraction': max(hours_remaining / 24, 0.001),
        'hours_remaining': hours_remaining,
        'minutes_remaining': hours_remaining * 60,
        'theta_decay_stage': 'SLOW' if hours_remaining > 6 else ('NORMAL' if hours_remaining > 3 else 'FAST'),
        'theta_risk': 'LOW' if hours_remaining > 6 else ('MEDIUM' if hours_remaining > 2 else 'HIGH'),
    }

def calculate_continuous_rank_score(bull_score, bear_score):
    net_score = (bull_score - bear_score) * 15.0
    rank_score = max(-15, min(15, net_score))
    abs_rank = abs(rank_score)
    if abs_rank >= 14:
        pm = 1.0
    elif abs_rank >= 12:
        pm = 0.80
    elif abs_rank >= 10:
        pm = 0.60
    elif abs_rank >= 8:
        pm = 0.40
    elif abs_rank >= 6:
        pm = 0.20
    else:
        pm = 0.0
    return rank_score, pm

def get_dynamic_strike_step(ltp):
    if ltp <= 100:
        return 1.0
    if ltp <= 500:
        return 2.5
    if ltp <= 1000:
        return 5.0
    return 10.0

class DailyPnLTracker:
    def __init__(self, daily_target=DAILY_PROFIT_TARGET):
        self.daily_target = daily_target
        self.current_date = datetime.now().date()
        self.realized_pnl = 0
        self.trades_taken_today = 0
        self.target_achieved = False
    def _reset_daily(self):
        self.current_date = datetime.now().date()
        self.realized_pnl = 0
        self.trades_taken_today = 0
        self.target_achieved = False
    def can_trade(self):
        if datetime.now().date() != self.current_date:
            self._reset_daily()
        return (not self.target_achieved) and self.realized_pnl > -MAX_DAILY_LOSS

def calculate_pullback_metrics(df):
    required_cols = {'high', 'close'}
    if df is None or len(df) < 5 or not required_cols.issubset(df.columns):
        return {'pullback_pct': 0, 'pullback_stage': 'UNKNOWN', 'recent_high': None, 'current_price': None}
    recent_high = df['high'].tail(20).max()
    current_price = df['close'].iloc[-1]
    pullback_pct = ((recent_high - current_price) / recent_high) * 100 if recent_high != 0 else 0
    if pullback_pct < 1:
        stage = 'NO_PULLBACK'
    elif pullback_pct < 2:
        stage = 'SHALLOW_PULLBACK'
    elif pullback_pct < 4:
        stage = 'MODERATE_PULLBACK'
    else:
        stage = 'DEEP_PULLBACK'
    return {'pullback_pct': pullback_pct, 'pullback_stage': stage, 'recent_high': recent_high, 'current_price': current_price}

def check_afternoon_sweet_spot():
    current_time = datetime.now().time()
    return {'is_sweet_spot': AFTERNOON_WINDOW_START <= current_time <= AFTERNOON_WINDOW_END, 'current_time': current_time.strftime('%H:%M:%S'), 'window_start': AFTERNOON_WINDOW_START.strftime('%H:%M'), 'window_end': AFTERNOON_WINDOW_END.strftime('%H:%M')}

def calculate_atr(df, period=14):
    if df is None or len(df) < period:
        return pd.Series([0] * len(df))
    hl = df['high'] - df['low']
    hc = np.abs(df['high'] - df['close'].shift())
    lc = np.abs(df['low'] - df['close'].shift())
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    return tr.rolling(window=period).mean()

def calculate_supertrend(df, period=10, multiplier=3):
    if df is None or len(df) < period:
        return df
    df = df.copy()
    atr = calculate_atr(df, period)
    hl_avg = (df['high'] + df['low']) / 2
    upper_band = hl_avg + (multiplier * atr)
    lower_band = hl_avg - (multiplier * atr)
    st = [0] * len(df)
    direction = [1] * len(df)
    for i in range(1, len(df)):
        if df['close'].iloc[i] > upper_band.iloc[i-1]:
            direction[i] = 1
        elif df['close'].iloc[i] < lower_band.iloc[i-1]:
            direction[i] = -1
        else:
            direction[i] = direction[i-1]
        st[i] = lower_band.iloc[i] if direction[i] == 1 else upper_band.iloc[i]
    df['supertrend'] = st
    df['supertrend_direction'] = direction
    return df

def calculate_technical_indicators(df):
    if df is None or len(df) < 50:
        return None
    df = df.copy()
    df['ema_20'] = ta.trend.ema_indicator(df['close'], window=20)
    df['ema_50'] = ta.trend.ema_indicator(df['close'], window=50)
    df['ema_200'] = ta.trend.ema_indicator(df['close'], window=200)
    df['macd'] = ta.trend.macd(df['close'])
    df['macd_signal'] = ta.trend.macd_signal(df['close'])
    df['adx'] = ta.trend.adx(df['high'], df['low'], df['close'])
    df['adx_pos'] = ta.trend.adx_pos(df['high'], df['low'], df['close'])
    df['adx_neg'] = ta.trend.adx_neg(df['high'], df['low'], df['close'])
    df['rsi'] = ta.momentum.rsi(df['close'], window=14)
    bb = ta.volatility.BollingerBands(df['close'])
    df['bb_high'] = bb.bollinger_hband()
    df['bb_mid'] = bb.bollinger_mavg()
    df['bb_low'] = bb.bollinger_lband()
    df['volume_sma'] = df['volume'].rolling(window=20).mean()
    df = calculate_supertrend(df)
    return df

def calculate_bull_factor_score(df):
    if df is None or len(df) < 30:
        return 0
    latest = df.iloc[-1]
    score = 0
    if latest['close'] > latest.get('ema_20', 0): score += 0.15
    if latest['close'] > latest.get('ema_50', 0): score += 0.15
    if latest['close'] > latest.get('ema_200', 0): score += 0.10
    if latest.get('macd', 0) > latest.get('macd_signal', 0): score += 0.15
    if latest.get('adx', 0) > 25: score += 0.10
    if latest.get('adx_pos', 0) > latest.get('adx_neg', 0): score += 0.10
    if 40 < latest.get('rsi', 50) < 70: score += 0.10
    if latest.get('supertrend_direction', 0) == 1: score += 0.15
    return min(score, 1.0)

def calculate_bear_factor_score(df):
    if df is None or len(df) < 30:
        return 0
    latest = df.iloc[-1]
    score = 0
    if latest['close'] < latest.get('ema_20', float('inf')): score += 0.15
    if latest['close'] < latest.get('ema_50', float('inf')): score += 0.15
    if latest['close'] < latest.get('ema_200', float('inf')): score += 0.10
    if latest.get('macd', 0) < latest.get('macd_signal', 0): score += 0.15
    if latest.get('adx', 0) > 25: score += 0.10
    if latest.get('adx_neg', 0) > latest.get('adx_pos', 0): score += 0.10
    if 30 < latest.get('rsi', 50) < 60: score += 0.10
    if latest.get('supertrend_direction', 0) == -1: score += 0.15
    return min(score, 1.0)

def validate_ohlc_data(df, symbol, timeframe):
    if df is None or df.empty:
        return False
    req = ['open', 'high', 'low', 'close', 'volume']
    if any(c not in df.columns for c in req):
        return False
    if len(df) < 30:
        return False
    if df[req].isnull().any().any():
        return False
    return True

def get_historical_data_with_validation(symbol, resolution, days_back):
    if fyers is None:
        return None
    key = f'{symbol}_{resolution}_{days_back}'
    if key in DATA_CACHE:
        return DATA_CACHE[key]
    try:
        now = datetime.now()
        data = {'symbol': symbol, 'resolution': resolution, 'date_format': '1', 'range_from': (now - timedelta(days=days_back)).strftime('%Y-%m-%d'), 'range_to': now.strftime('%Y-%m-%d'), 'cont_flag': '1'}
        response = fyers.history(data=data)
        if response.get('s') != 'ok' or 'candles' not in response or not response['candles']:
            return None
        df = pd.DataFrame(response['candles'], columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
        df = df.sort_values('timestamp').reset_index(drop=True)
        if not validate_ohlc_data(df, symbol, resolution):
            return None
        DATA_CACHE[key] = df
        return df
    except Exception as e:
        logger.error(str(e))
        return None

def calculate_exit_signals_with_two_indicator_rule(df, signal_type):
    signal_type = (signal_type or '').upper()
    if df is None or len(df) < 30:
        return {'exit_signals_count': 0, 'exit_reason': 'INSUFFICIENT_DATA', 'should_exit': False}
    signals = []
    df_st = calculate_supertrend(df)
    if 'supertrend_direction' in df_st.columns:
        d = df_st['supertrend_direction'].iloc[-1]
        if (signal_type == 'BULLISH' and d == -1) or (signal_type == 'BEARISH' and d == 1):
            signals.append('SUPERTREND_EXIT')
    macd = ta.trend.macd(df['close'])
    sig = ta.trend.macd_signal(df['close'])
    if len(macd) >= 2 and len(sig) >= 2:
        if signal_type == 'BULLISH' and macd.iloc[-1] < sig.iloc[-1]:
            signals.append('MACD_NEGATIVE')
        elif signal_type == 'BEARISH' and macd.iloc[-1] > sig.iloc[-1]:
            signals.append('MACD_POSITIVE')
    adx = ta.trend.adx(df['high'], df['low'], df['close'])
    if len(adx) > 0 and adx.iloc[-1] < 20:
        signals.append('ADX_WEAK')
    if calculate_dynamic_dte_with_decay()['theta_risk'] == 'HIGH':
        signals.append('THETA_RISK')
    return {'exit_signals_count': len(signals), 'exit_reason': ' + '.join(signals) if signals else 'NONE', 'should_exit': len(signals) >= 2}

def recommend_option_strikes_with_greeks_liquid_v30(symbol, ltp, signal_type, iv_estimate=0.25):
    dte = calculate_dynamic_dte_with_decay()
    T = dte['dte_fraction']
    if T < 0.001:
        return None
    step = get_dynamic_strike_step(ltp)
    max_dist = max(ltp * MAX_STRIKE_DISTANCE_FROM_LTP, step)
    strikes = []
    for i in range(-10, 11):
        strike = round(ltp + i * step)
        if strike > 0 and abs(strike - ltp) <= max_dist:
            strikes.append(strike)
    recs = []
    for strike in strikes:
        opt_type = 'call' if signal_type == 'BULLISH' else 'put'
        delta = norm.cdf((np.log(ltp / strike) + (RISK_FREE_RATE + 0.5 * iv_estimate ** 2) * T) / (iv_estimate * np.sqrt(T))) if opt_type == 'call' else norm.cdf((np.log(ltp / strike) + (RISK_FREE_RATE + 0.5 * iv_estimate ** 2) * T) / (iv_estimate * np.sqrt(T))) - 1
        theta = 0.0
        premium = max(ltp - strike, 0) if opt_type == 'call' else max(strike - ltp, 0)
        if MIN_DELTA_TARGET <= abs(delta) <= MAX_DELTA_TARGET:
            recs.append({'strike': strike, 'option_type': opt_type, 'delta': delta, 'theta': theta, 'premium_estimate': premium, 'dte_hours': dte['hours_remaining'], 'theta_decay_stage': dte['theta_decay_stage'], 'theta_risk': dte['theta_risk']})
    if not recs:
        return None
    recs.sort(key=lambda x: abs(abs(x['delta']) - 0.50))
    return recs[0]

def calculate_iv_percentile(current_iv, df, window=252):
    if df is None or len(df) < 20:
        return 50
    hist = []
    for i in range(min(len(df), window)):
        if i >= 30:
            returns = np.log(df['close'].iloc[:i] / df['close'].iloc[:i].shift(1))
            hist.append(returns.tail(30).std() * np.sqrt(252))
    return percentileofscore(hist, current_iv) if hist else 50

def confirm_momentum_for_entry(df, signal_type):
    if df is None or len(df) < 3:
        return False
    recent = df.tail(3)
    if signal_type == 'BULLISH':
        return sum(1 for _, r in recent.iterrows() if r['close'] > r['open']) >= 2
    return sum(1 for _, r in recent.iterrows() if r['close'] < r['open']) >= 2

def calculate_intraday_price_metrics(df):
    if df is None or len(df) < 2:
        return {'price_change_pct': 0, 'volume_surge': 1.0}
    open_price = df['open'].iloc[0]
    current_price = df['close'].iloc[-1]
    avg_volume = df['volume'].tail(20).mean()
    current_volume = df['volume'].iloc[-1]
    return {'price_change_pct': ((current_price - open_price) / open_price) * 100, 'volume_surge': current_volume / avg_volume if avg_volume > 0 else 1.0, 'open_price': open_price, 'current_price': current_price}

def process_stock_multitimeframe_v30(symbol):
    tf_results = {}
    for tf_name, tf_conf in TIMEFRAMES.items():
        df = get_historical_data_with_validation(symbol, tf_conf['resolution'], tf_conf['days'])
        if df is None:
            continue
        ind = calculate_technical_indicators(df)
        if ind is None:
            continue
        tf_results[tf_name] = {'df': ind, 'bull_score': calculate_bull_factor_score(ind), 'bear_score': calculate_bear_factor_score(ind), 'weight': tf_conf['weight']}
    if not tf_results:
        return None
    total_bull = sum(r['bull_score'] * r['weight'] for r in tf_results.values())
    total_bear = sum(r['bear_score'] * r['weight'] for r in tf_results.values())
    rank_score, pm = calculate_continuous_rank_score(total_bull, total_bear)
    dominant = 'BULLISH' if rank_score > 0 else ('BEARISH' if rank_score < 0 else 'NEUTRAL')
    latest_df = tf_results['5min']['df'] if '5min' in tf_results else list(tf_results.values())[0]['df']
    ltp = latest_df['close'].iloc[-1]
    exit_decision = calculate_exit_signals_with_two_indicator_rule(latest_df, dominant)
    pullback_info = calculate_pullback_metrics(latest_df)
    sweet_spot_info = check_afternoon_sweet_spot()
    dte_info = calculate_dynamic_dte_with_decay()
    base_conf = min(1.0, abs(rank_score) / 15.0)
    if pullback_info['pullback_stage'] in ('SHALLOW_PULLBACK', 'MODERATE_PULLBACK'):
        base_conf *= 1.10
    elif pullback_info['pullback_stage'] == 'DEEP_PULLBACK':
        base_conf *= 0.90
    if exit_decision['exit_signals_count'] >= 1:
        base_conf *= 0.60
    if dte_info['theta_risk'] == 'HIGH':
        base_conf *= 0.80
    option_rec = recommend_option_strikes_with_greeks_liquid_v30(symbol, ltp, dominant)
    return {
        'Symbol': symbol,
        'RankScore15Tier': rank_score,
        'BullMultiTFScore': total_bull,
        'BearMultiTFScore': total_bear,
        'DominantTrend': dominant,
        'TrendStrength': abs(rank_score) / 15.0,
        'PositionSizeMultiplier': pm,
        'EntryConfidence': max(0.0, min(1.0, base_conf)),
        'LTP': ltp,
        'ExitSignalsCount': exit_decision['exit_signals_count'],
        'ExitReason': exit_decision['exit_reason'],
        'ShouldExit': exit_decision['should_exit'],
        'PullbackPct': pullback_info['pullback_pct'],
        'PullbackStage': pullback_info['pullback_stage'],
        'IsAfternoonSweetSpot': sweet_spot_info['is_sweet_spot'],
        'DTEHours': dte_info['hours_remaining'],
        'ThetaDecayStage': dte_info['theta_decay_stage'],
        'ThetaRisk': dte_info['theta_risk'],
        'OptionStrike': option_rec['strike'] if option_rec else None,
        'OptionDelta': option_rec['delta'] if option_rec else None,
        'OptionTheta': option_rec['theta'] if option_rec else None,
        'CanTradeToday': True,
    }

def rank_all_stocks_multitimeframe_v30(symbols_list):
    results = []
    for sym in symbols_list:
        r = process_stock_multitimeframe_v30(sym)
        if r:
            results.append(r)
        else:
            FAILED_SYMBOLS.append(sym)
    if not results:
        return pd.DataFrame()
    df = pd.DataFrame(results)
    return df.sort_values('RankScore15Tier', key=lambda s: s.abs(), ascending=False).reset_index(drop=True)

def init_daily_db():
    today = datetime.now().date().strftime('%Y-%m-%d')
    conn = sqlite3.connect(DB_PATH)
    conn.execute('CREATE TABLE IF NOT EXISTS stock_signals (run_id INTEGER PRIMARY KEY AUTOINCREMENT, date TEXT, runtime TEXT, Symbol TEXT, RankScore15Tier REAL, BullMultiTFScore REAL, BearMultiTFScore REAL, DominantTrend TEXT, TrendStrength REAL, PositionSizeMultiplier REAL, EntryConfidence REAL, LTP REAL, CanTradeToday TEXT, ExitSignalsCount INTEGER, ExitReason TEXT)')
    conn.execute('DELETE FROM stock_signals WHERE date != ?', (today,))
    conn.commit(); conn.close()

def store_results_in_db(df):
    if df is None or df.empty:
        return
    conn = sqlite3.connect(DB_PATH)
    x = df.copy()
    x['date'] = datetime.now().date().strftime('%Y-%m-%d')
    x['runtime'] = datetime.now().strftime('%H:%M:%S')
    cols = ['date','runtime','Symbol','RankScore15Tier','BullMultiTFScore','BearMultiTFScore','DominantTrend','TrendStrength','PositionSizeMultiplier','EntryConfidence','LTP','CanTradeToday','ExitSignalsCount','ExitReason']
    for c in cols:
        if c not in x.columns:
            x[c] = 0 if c == 'ExitSignalsCount' else (True if c == 'CanTradeToday' else ('UNKNOWN' if c == 'ExitReason' else None))
    x[cols].to_sql('stock_signals', conn, if_exists='append', index=False)
    conn.commit(); conn.close()

def build_display_df(df_side: pd.DataFrame, side: str, sector_map: dict = None) -> pd.DataFrame:
    if df_side is None or df_side.empty:
        return pd.DataFrame()
    df = df_side.copy()
    if 'Diff' not in df.columns:
        df['Diff'] = df.get('EntryConfidence', 0).fillna(0) * 100 - df.get('ExitSignalsCount', 0).fillna(0) * 10
    return df.sort_values('Diff', ascending=False).reset_index(drop=True)

def send_email_rank_watchlist(csvfilename, msg=None):
    sender_email = os.getenv('SENDER_EMAIL')
    sender_password = os.getenv('SENDER_PASSWORD')
    recipient_email = os.getenv('RECIPIENT_EMAIL')
    if not all([sender_email, sender_password, recipient_email]):
        return False
    msg = msg or MIMEMultipart()
    msg['From'] = sender_email
    msg['To'] = recipient_email
    msg['Subject'] = f'Intraday Rank Watchlist - {datetime.now().strftime("%Y-%m-%d %H:%M")}'
    msg.attach(MIMEText(f'<html><body><h3>Rank Watchlist</h3><p>Attached: {os.path.basename(csvfilename)}</p></body></html>', 'html'))
    with open(csvfilename, 'rb') as f:
        part = MIMEBase('application', 'octet-stream')
        part.set_payload(f.read())
    encoders.encode_base64(part)
    part.add_header('Content-Disposition', f'attachment; filename={os.path.basename(csvfilename)}')
    msg.attach(part)
    server = smtplib.SMTP(os.getenv('SMTP_SERVER', 'smtp.gmail.com'), int(os.getenv('SMTP_PORT', 587)))
    server.starttls()
    server.login(sender_email, sender_password)
    server.sendmail(sender_email, recipient_email, msg.as_string())
    server.quit()
    return True

if __name__ == '__main__':
    print('Clean rewritten EMAIL.py ready')
