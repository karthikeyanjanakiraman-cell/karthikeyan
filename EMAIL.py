#!/usr/bin/env python3
"""
ASIT v16.2 - FRACTAL PHYSICS ENGINE
HARDENED FOR WEEKENDS/HOLIDAYS | AUTO-DETECTS LAST TRADING DAY
"""

import os
import logging
import smtplib
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from concurrent.futures import ThreadPoolExecutor, as_completed
from fyers_apiv3 import fyersModel

# --- SETUP ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

# --- CREDENTIALS ---
CLIENT_ID = os.environ.get("CLIENT_ID")
ACCESS_TOKEN = os.environ.get("ACCESS_TOKEN")
SENDER_EMAIL = os.environ.get("SENDER_EMAIL")
SENDER_PASSWORD = os.environ.get("SENDER_PASSWORD") # USE APP PASSWORD
RECIPIENT_EMAIL = os.environ.get("RECIPIENT_EMAIL")

fyers = fyersModel.FyersModel(client_id=CLIENT_ID, token=ACCESS_TOKEN, is_async=False)

def get_15m_history_6m(symbol):
    try:
        now = datetime.now()
        # Fetching 180 days to ensure we have enough history
        res = fyers.history({
            "symbol": symbol, "resolution": "15", "date_format": 1,
            "range_from": (now - timedelta(days=180)).strftime("%Y-%m-%d"),
            "range_to": now.strftime("%Y-%m-%d"), "cont_flag": 1
        })
        if res and res.get('candles'):
            df = pd.DataFrame(res['candles'], columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
            return df
    except Exception as e:
        logger.error(f"Error fetching {symbol}: {e}")
    return None

def get_fractal_decay_score(df_today):
    total_vol = df_today['volume'].sum()
    if total_vol < 10000: return 1.0 
    
    def get_time_for_vol(vol_target):
        cum_vol = df_today['volume'].cumsum()
        idx = np.searchsorted(cum_vol.values, vol_target)
        if idx >= len(df_today): return len(df_today) * 15
        return (idx + 1) * 15

    try:
        t50 = get_time_for_vol(total_vol * 0.5)
        t75 = get_time_for_vol(total_vol * 0.75) - t50
        t87 = get_time_for_vol(total_vol * 0.875) - (t50 + t75)
        decay_macro = t75 / t50 if t50 > 0 else 1.0
        decay_micro = t87 / t75 if t75 > 0 else 1.0
        return (decay_macro + decay_micro) / 2
    except: return 1.0

def process_physics(df):
    if df is None or len(df) < 10: return None
    
    df['date'] = df['timestamp'].dt.date
    # FIX: Use the last available date, not today's actual calendar date
    last_date = df['date'].unique()[-1]
    
    df['vol_diff'] = df['volume'].diff().fillna(0)
    df['vol_viol'] = df['vol_diff'].abs()
    df['vol_accel'] = df['vol_diff'].clip(lower=0)
    
    df_session = df[df['date'] == last_date].copy()
    if df_session.empty: return None
    
    T_session = len(df_session) * 15
    target_mass = df_session['volume'].sum()
    target_viol = df_session['vol_viol'].sum()
    target_accel = df_session['vol_accel'].sum()
    
    # History Compare
    r_mass, r_viol, r_accel = T_session, T_session, T_session
    for d in df['date'].unique()[:-1]:
        d_dat = df[df['date'] == d]
        cum_m, cum_v, cum_a = d_dat['volume'].cumsum().values, d_dat['vol_viol'].cumsum().values, d_dat['vol_accel'].cumsum().values
        if np.any(cum_m >= target_mass): r_mass = min(r_mass, (np.argmax(cum_m >= target_mass)+1)*15)
        if np.any(cum_v >= target_viol): r_viol = min(r_viol, (np.argmax(cum_v >= target_viol)+1)*15)
        if np.any(cum_a >= target_accel): r_accel = min(r_accel, (np.argmax(cum_a >= target_accel)+1)*15)

    rank = ((r_mass/T_session*15) + (r_viol/T_session*15) + (r_accel/T_session*15)) / 3.0
    decay = get_fractal_decay_score(df_session)
    if decay > 1.0: rank /= decay 
    
    vwap = (df_session['volume'] * ((df_session['high']+df_session['low']+df_session['close'])/3)).sum() / target_mass
    direction = 1 if df_session['close'].iloc[-1] >= vwap else -1
    
    return {
        'rank': rank * direction, 'decay': decay, 'mass': (r_mass/T_session*15),
        'viol': (r_viol/T_session*15), 'accel': (r_accel/T_session*15), 
        'ltp': df_session['close'].iloc[-1]
    }

def scan_symbol(symbol):
    df = get_15m_history_6m(symbol)
    data = process_physics(df)
    if data:
        data['Symbol'] = symbol.replace('NSE:', '').replace('-EQ', '')
        return data
    return None

def send_email(results):
    df = pd.DataFrame(results)
    if 'rank' not in df.columns:
        logger.error("No valid results to send (Physics criteria not met).")
        return

    bulls = df[df['rank'] > 0].sort_values('rank', ascending=False).head(15)
    bears = df[df['rank'] < 0].sort_values('rank', ascending=True).head(15)
    
    msg = MIMEMultipart("alternative")
    msg["Subject"] = f"Fractal Physics Matrix - {datetime.now().strftime('%Y-%m-%d %H:%M')}"
    
    html = f"<html><body><h2>Fractal Physics Matrix</h2><h3>Bulls</h3><table border='1'><tr><th>Sym</th><th>Mass</th><th>Viol</th><th>Accel</th><th>Decay</th><th>Net Speed</th></tr>"
    html += "".join([f"<tr><td>{r['Symbol']}</td><td>{r['mass']:.1f}</td><td>{r['viol']:.1f}</td><td>{r['accel']:.1f}</td><td style='color:{'red' if r['decay']>1.2 else 'green'}'>{r['decay']:.2f}</td><td>{r['rank']:.2f}</td></tr>" for _, r in bulls.iterrows()])
    html += "</table></body></html>"
    msg.attach(MIMEText(html, "html"))
    
    with smtplib.SMTP("smtp.gmail.com", 587) as s:
        s.starttls()
        s.login(SENDER_EMAIL, SENDER_PASSWORD)
        s.sendmail(SENDER_EMAIL, RECIPIENT_EMAIL, msg.as_string())

def main():
    # Replace the list below with your actual F&O list fetcher
    symbols = ["NSE:RELIANCE-EQ", "NSE:TCS-EQ", "NSE:INFY-EQ", "NSE:HDFCBANK-EQ"] 
    logger.info(f"Scanning {len(symbols)} symbols...")
    results = []
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = {executor.submit(scan_symbol, sym): sym for sym in symbols}
        for future in as_completed(futures):
            res = future.result()
            if res: results.append(res)
    
    if results: send_email(results)
    else: logger.warning("Scan complete, but no symbols passed the physics filters.")

if __name__ == "__main__":
    main()

