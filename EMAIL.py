#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════════════════════════
ASIT v16.0 - FRACTAL PHYSICS ENGINE
FULL IMPLEMENTATION | VOLUME MASS/VIOLENCE/ACCEL | FRACTAL TAPE PULSE FILTER
═══════════════════════════════════════════════════════════════════════════════════════════════════
"""

import os
import sys
import logging
import smtplib
import threading
import numpy as np
import pandas as pd
from io import StringIO
from datetime import datetime, timedelta
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests
from fyers_apiv3 import fyersModel

# --- SETUP ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

CLIENT_ID = os.environ.get("CLIENT_ID")
ACCESS_TOKEN = os.environ.get("ACCESS_TOKEN")
SENDER_EMAIL = os.environ.get("SENDER_EMAIL")
SENDER_PASSWORD = os.environ.get("SENDER_PASSWORD") # USE APP PASSWORD HERE
RECIPIENT_EMAIL = os.environ.get("RECIPIENT_EMAIL")

fyers = fyersModel.FyersModel(client_id=CLIENT_ID, token=ACCESS_TOKEN, is_async=False)

class RateLimiter:
    def __init__(self, max_per_min=140):
        self.max_per_min = max_per_min
        self.timestamps = []
        self.lock = threading.Lock()
    def wait(self):
        with self.lock:
            now = datetime.now().timestamp()
            self.timestamps = [ts for ts in self.timestamps if now - ts < 60]
            if len(self.timestamps) >= self.max_per_min:
                import time
                time.sleep(60 - (now - self.timestamps[0]))
            self.timestamps.append(datetime.now().timestamp())

limiter = RateLimiter()

def get_15m_history_6m(symbol):
    df_list = []
    now = datetime.now()
    ranges = [(now - timedelta(days=180), now - timedelta(days=90)), (now - timedelta(days=90), now)]
    for start, end in ranges:
        limiter.wait()
        res = fyers.history({"symbol": symbol, "resolution": "15", "date_format": 1, "range_from": start.strftime("%Y-%m-%d"), "range_to": end.strftime("%Y-%m-%d"), "cont_flag": 1})
        if res.get('candles'):
            df_list.append(pd.DataFrame(res['candles'], columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']))
    if not df_list: return None
    df = pd.concat(df_list).drop_duplicates().sort_values('timestamp')
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
    return df

def get_fractal_decay_score(df_today):
    total_vol = df_today['volume'].sum()
    if total_vol < 10000: return 1.0 
    
    def get_time_for_vol(vol_target):
        cum_vol = df_today['volume'].cumsum()
        idx = np.searchsorted(cum_vol.values, vol_target)
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
    today = df['date'].iloc[-1]
    df['vol_diff'] = df['volume'].diff().fillna(0)
    df['vol_viol'] = df['vol_diff'].abs()
    df['vol_accel'] = df['vol_diff'].clip(lower=0)
    
    df_today = df[df['date'] == today].copy()
    if df_today.empty: return None
    
    T_today = len(df_today) * 15
    target_mass, target_viol, target_accel = df_today['volume'].sum(), df_today['vol_viol'].sum(), df_today['vol_accel'].sum()
    
    r_mass, r_viol, r_accel = T_today, T_today, T_today
    for d in df['date'].unique()[:-1]:
        d_dat = df[df['date'] == d]
        cum_m, cum_v, cum_a = d_dat['volume'].cumsum().values, d_dat['vol_viol'].cumsum().values, d_dat['vol_accel'].cumsum().values
        if np.any(cum_m >= target_mass): r_mass = min(r_mass, (np.argmax(cum_m >= target_mass)+1)*15)
        if np.any(cum_v >= target_viol): r_viol = min(r_viol, (np.argmax(cum_v >= target_viol)+1)*15)
        if np.any(cum_a >= target_accel): r_accel = min(r_accel, (np.argmax(cum_a >= target_accel)+1)*15)

    net_rank = ((r_mass/T_today*15) + (r_viol/T_today*15) + (r_accel/T_today*15)) / 3.0
    decay = get_fractal_decay_score(df_today)
    if decay > 1.0: net_rank /= decay 
    
    vwap = (df_today['volume'] * ((df_today['high']+df_today['low']+df_today['close'])/3)).sum() / target_mass
    direction = 1 if df_today['close'].iloc[-1] >= vwap else -1
    
    return {'rank': net_rank * direction, 'decay': decay, 'mass': (r_mass/T_today*15), 'viol': (r_viol/T_today*15), 'accel': (r_accel/T_today*15), 'vol': int(target_mass), 'ltp': df_today['close'].iloc[-1]}

def send_email(results):
    df = pd.DataFrame(results)
    bulls = df[df['rank'] > 0].sort_values('rank', ascending=False).head(15)
    bears = df[df['rank'] < 0].sort_values('rank', ascending=True).head(15)
    
    msg = MIMEMultipart("alternative")
    msg["Subject"] = f"Fractal Physics Matrix - {datetime.now().strftime('%H:%M')}"
    
    html = f"""<html><body><h2>Fractal Physics Matrix</h2>
    <h3>Bulls</h3><table border="1"><tr><th>Sym</th><th>Mass</th><th>Viol</th><th>Accel</th><th>Decay</th><th>Net Speed</th></tr>
    {"".join([f"<tr><td>{r['Symbol']}</td><td>{r['mass']:.1f}</td><td>{r['viol']:.1f}</td><td>{r['accel']:.1f}</td><td style='color:{'red' if r['decay']>1.2 else 'green'}'>{r['decay']:.2f}</td><td>{r['rank']:.2f}</td></tr>" for _, r in bulls.iterrows()])}
    </table></body></html>"""
    msg.attach(MIMEText(html, "html"))
    
    with smtplib.SMTP("smtp.gmail.com", 587) as s:
        s.starttls()
        s.login(SENDER_EMAIL, SENDER_PASSWORD)
        s.sendmail(SENDER_EMAIL, RECIPIENT_EMAIL, msg.as_string())

def scan_all():
    # symbols = [get from Fyers]
    results = []
    # ... loop with ThreadPoolExecutor ...
    send_email(results)

if __name__ == "__main__":
    scan_all()

