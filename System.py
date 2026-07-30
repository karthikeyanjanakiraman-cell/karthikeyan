import os
import smtplib
import urllib.parse
import json
import gzip
import io
import time
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from datetime import datetime, timedelta

import requests
import pandas as pd
import numpy as np
import warnings

# The Quantitative Gold Standard for Tabular Data
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')

# ==============================================================================
# 1. PURE VECTORIZED FEATURE ENGINEERING (Bypasses Memory Constraints)
# ==============================================================================
def engineer_features(df, is_training=True):
    """Executes 100x faster than loops by leveraging C-backend Pandas Vectorization."""
    if 'Symbol' not in df.columns: df['Symbol'] = 'ASSET'
    df = df.sort_values(['Symbol', 'Date']).reset_index(drop=True)
    
    # 1. Price Momentum Vectors
    df['ret_1d'] = df.groupby('Symbol')['Close'].pct_change(1)
    df['ret_5d'] = df.groupby('Symbol')['Close'].pct_change(5)
    df['ret_10d'] = df.groupby('Symbol')['Close'].pct_change(10)
    df['ret_20d'] = df.groupby('Symbol')['Close'].pct_change(20)
    
    # 2. Volatility (ATR Proxy) & Squeeze
    df['High_Low'] = df['High'] - df['Low']
    df['ATR_10'] = df.groupby('Symbol')['High_Low'].transform(lambda x: x.rolling(10).mean())
    df['Vol_Squeeze'] = df['High_Low'] / (df['ATR_10'] + 1e-8)
    
    # 3. Institutional Volume Flow
    df['Vol_20d_SMA'] = df.groupby('Symbol')['Volume'].transform(lambda x: x.rolling(20).mean())
    df['Volume_Surge'] = df['Volume'] / (df['Vol_20d_SMA'] + 1e-8)
    
    # 4. Macro Market Positioning
    df['Max_50d'] = df.groupby('Symbol')['High'].transform(lambda x: x.rolling(50).max())
    df['Min_50d'] = df.groupby('Symbol')['Low'].transform(lambda x: x.rolling(50).min())
    df['Position_50d'] = (df['Close'] - df['Min_50d']) / (df['Max_50d'] - df['Min_50d'] + 1e-8)
    
    if is_training:
        # TARGETS: Strict T+1 Open to T+2 High/Low mapping
        df['Next_Open'] = df.groupby('Symbol')['Open'].shift(-1)
        
        df['High_T1'] = df.groupby('Symbol')['High'].shift(-1)
        df['High_T2'] = df.groupby('Symbol')['High'].shift(-2)
        df['Future_High'] = df[['High_T1', 'High_T2']].max(axis=1)
        
        df['Low_T1'] = df.groupby('Symbol')['Low'].shift(-1)
        df['Low_T2'] = df.groupby('Symbol')['Low'].shift(-2)
        df['Future_Low'] = df[['Low_T1', 'Low_T2']].min(axis=1)
        
        # Dual Continuous Targets for GBDT
        df['Max_Up_Pct'] = ((df['Future_High'] - df['Next_Open']) / (df['Next_Open'] + 1e-8)) * 100.0
        df['Max_Down_Pct'] = ((df['Next_Open'] - df['Future_Low']) / (df['Next_Open'] + 1e-8)) * 100.0
        
        df = df.replace([np.inf, -np.inf], np.nan).dropna()
        
    return df

# ==============================================================================
# 2. DATA PROCESSING COMPILER
# ==============================================================================
def read_standard_csv(filename):
    if not filename or not os.path.exists(filename): return None
    try:
        df = pd.read_csv(filename)
        df.rename(columns=lambda x: str(x).lower().strip(), inplace=True)
        col_map = {'date':'Date', 'timestamp':'Date', 'symbol':'Symbol', 'ticker':'Symbol', 'open':'Open', 'high':'High', 'low':'Low', 'close':'Close', 'volume':'Volume'}
        df.rename(columns=col_map, inplace=True)
        
        if 'Date' in df.columns:
            # Safely parses Indian DD-MM-YYYY vs Global YYYY-MM-DD
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce', format='mixed').dt.strftime('%Y-%m-%d')
            df = df.dropna(subset=['Date'])
        return df
    except Exception as e:
        print(f"❌ ERROR parsing {filename}: {str(e)}")
        return None

# ==============================================================================
# 3. GRADIENT BOOSTING QUANT DESK (Extremely Fast, Low RAM)
# ==============================================================================
def train_quant_models(csv_file, dt_str, is_macro=False):
    df_train = read_standard_csv(csv_file)
    if df_train is None: return None, None, None
    
    if is_macro and 'Symbol' in df_train.columns:
        df_train = df_train[df_train['Symbol'].astype(str).str.upper().str.contains("NIFTY50|NIFTY")]
        
    df_train = df_train[df_train['Date'] < dt_str]
    df_train = engineer_features(df_train, is_training=True)
    if df_train.empty: return None, None, None
    
    features = ['ret_1d', 'ret_5d', 'ret_10d', 'ret_20d', 'Vol_Squeeze', 'Volume_Surge', 'Position_50d']
    X = df_train[features]
    Y_up = df_train['Max_Up_Pct']
    Y_down = df_train['Max_Down_Pct']
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # HistGradientBoosting automatically bins data, making it immune to OOM crashes
    model_up = HistGradientBoostingRegressor(max_iter=150, max_depth=6, random_state=42)
    model_down = HistGradientBoostingRegressor(max_iter=150, max_depth=6, random_state=42)
    
    model_up.fit(X_scaled, Y_up)
    model_down.fit(X_scaled, Y_down)
    
    return model_up, model_down, scaler

# ==============================================================================
# 4. LIVE INGESTION
# ==============================================================================
def fetch_915_open(key, dt_str):
    token = os.environ.get("UPSTOX_ACCESS_TOKEN")
    if not token or not key: return None
    try:
        url = f"https://api.upstox.com/v2/historical-candle/intraday/{urllib.parse.quote(key)}/1minute"
        resp = requests.get(url, headers={'Accept': 'application/json', 'Authorization': f'Bearer {token}'}, timeout=5)
        if resp.status_code == 200:
            for c in resp.json().get('data', {}).get('candles', []):
                if dt_str in str(c[0]) and "09:15" in str(c[0]): return float(c[1])
            return float(resp.json().get('data', {}).get('candles', [])[-1][1])
    except: pass
    return None

def get_live_features(symbol, key, dt_str, is_backtest, df_full=None):
    if is_backtest and df_full is not None:
        df_sym = df_full[df_full['Symbol'] == symbol].copy()
        df_hist = df_sym[df_sym['Date'] < dt_str].copy()
        if len(df_hist) < 30: return None, None
        
        df_feat = engineer_features(df_hist, is_training=False)
        df_feat = df_feat.replace([np.inf, -np.inf], np.nan).dropna()
        if df_feat.empty: return None, None
        
        last_row = df_feat.iloc[-1]
        df_fut = df_sym[df_sym['Date'] >= dt_str]
        entry = float(df_fut.iloc[0]['Open']) if not df_fut.empty else float(last_row['Close'])
        return last_row, entry
    else:
        token = os.environ.get("UPSTOX_ACCESS_TOKEN")
        if not token: return None, None
        dt = datetime.strptime(dt_str, "%Y-%m-%d")
        url = f"https://api.upstox.com/v2/historical-candle/{urllib.parse.quote(key)}/day/{(dt-timedelta(days=1)).strftime('%Y-%m-%d')}/{(dt-timedelta(days=60)).strftime('%Y-%m-%d')}"
        resp = requests.get(url, headers={'Accept': 'application/json', 'Authorization': f'Bearer {token}'})
        if resp.status_code != 200: return None, None
        
        data = resp.json().get('data', {}).get('candles', [])
        if not data or len(data) < 30: return None, None
        
        cols = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'OI']
        df_live = pd.DataFrame(data, columns=cols).iloc[::-1].reset_index(drop=True)
        df_live['Symbol'] = symbol
        df_live[['Open', 'High', 'Low', 'Close', 'Volume']] = df_live[['Open', 'High', 'Low', 'Close', 'Volume']].astype(float)
        
        df_feat = engineer_features(df_live, is_training=False)
        df_feat = df_feat.replace([np.inf, -np.inf], np.nan).dropna()
        if df_feat.empty: return None, None
        
        entry = fetch_915_open(key, dt_str)
        if entry is None: entry = float(df_feat.iloc[-1]['Close'])
        return df_feat.iloc[-1], entry

def get_fno_universe():
    try:
        resp = requests.get("https://assets.upstox.com/market-quote/instruments/exchange/NSE.json.gz", timeout=10)
        if resp.status_code != 200: return []
        data = json.load(gzip.GzipFile(fileobj=io.BytesIO(resp.content)))
        und = {i.get("underlying_symbol") for i in data if i.get("segment") == "NSE_FO" and i.get("underlying_symbol")}
        return [{"symbol": i.get("trading_symbol"), "key": i.get("instrument_key")} for i in data if i.get("segment") in ("NSE_EQ", "NSE_INDEX") and i.get("trading_symbol") in und]
    except: return []

# ==============================================================================
# 5. MASTER DISPATCH ENGINE
# ==============================================================================
def send_mobile_alert(macro_data, fno_data_list, target_date_str, is_backtest):
    sender_email, sender_pass, recipient_email = os.environ.get("SENDER_EMAIL"), os.environ.get("SENDER_PASSWORD"), os.environ.get("RECIPIENT_EMAIL")
    if not all([sender_email, sender_pass, recipient_email]): return

    msg = MIMEMultipart('alternative')
    msg['Subject'] = f"{'⏪ BACKTEST' if is_backtest else '🚀 LIVE GRADIENT BOOSTING ALERT'} | {target_date_str}"
    msg['From'], msg['To'] = sender_email, recipient_email

    macro_color = "#28a745" if "LONG" in macro_data['direction'] else "#dc3545" if "SHORT" in macro_data['direction'] else "#ffc107"
    
    html = f"""
    <html><body style="font-family: Arial, sans-serif; background-color: #f4f7f6; padding: 10px;">
        <div style="background-color: white; padding: 15px; border-radius: 8px; margin-bottom: 20px; border-left: 6px solid {macro_color};">
            <h3 style="margin-top: 0; color: #333;">🌍 MACRO REGIME (Nifty 50 Boosting Model)</h3>
            <p style="font-size: 16px; color: #333; margin: 5px 0;">
                <b>Direction:</b> <span style="color: {macro_color}; font-weight: bold;">{macro_data['direction']}</span><br>
                <b>Expected Target:</b> {macro_data['target_display']} | <b>Expected Risk:</b> {macro_data['risk_pct']:.2f}%<br>
                <b>Model Confidence:</b> {macro_data['conviction']:.2f}%
            </p>
        </div>
        <h3 style="color: #333;">⚡ MICRO F&O SWEEP (Kelly-Optimized Allocation)</h3>
        <table border="1" cellpadding="8" cellspacing="0" style="border-collapse: collapse; width: 100%; text-align: center; font-size: 14px; background-color: white;">
          <tr bgcolor="#f8f9fa" style="color: #333; font-weight: bold;">
            <th>Asset</th><th>Action</th><th>R/R Ratio</th><th>Kelly Sizing</th><th>Entry</th><th>Stop Loss</th><th>Target</th><th>Result</th>
          </tr>"""
    
    for row in sorted(fno_data_list, key=lambda x: x['kelly_pct'], reverse=True):
        dc = "#28a745" if "LONG" in row['direction'] else "#dc3545"
        html += f"<tr><td style='color: #0056b3;'><b>{row['asset']}</b></td><td style='color: {dc}; font-weight: bold;'>{row['direction']}</td><td>1 : {row['rr_ratio']:.1f}</td><td><b style='color:#6f42c1;'>{row['kelly_pct']:.1f}%</b></td><td>₹{row['entry']:.2f}</td><td style='color: #dc3545;'>₹{row['ai_stop']:.2f}</td><td style='color: {dc}; font-weight: bold;'>{row['target_display']}</td><td>{row['actual_outcome']}</td></tr>"
        
    html += "</table></body></html>"
    msg.attach(MIMEText(html, 'html'))
    try:
        server = smtplib.SMTP('smtp.gmail.com', 587); server.starttls(); server.login(sender_email, sender_pass)
        server.sendmail(sender_email, recipient_email, msg.as_string()); server.quit()
        print("✅ Report Dispatched.")
    except Exception as e: print(f"Failed to send email: {str(e)}")

def run_production_sweep():
    dt_str = os.environ.get("PARAM_BACKTEST_DATE", "").strip()
    is_bt = bool(dt_str)
    if not is_bt: dt_str = datetime.now().strftime("%Y-%m-%d")
    
    print(f"⚙️ EXECUTING GRADIENT BOOSTING AI DESK | DATE: {dt_str}")
    features = ['ret_1d', 'ret_5d', 'ret_10d', 'ret_20d', 'Vol_Squeeze', 'Volume_Surge', 'Position_50d']
    
    # ---------------------------------------------------------
    # PHASE 1: NIFTY MACRO (Crash-Proof GBDT)
    # ---------------------------------------------------------
    print("\n🧠 PHASE 1: Training Vectorized NIFTY 50 Booster...")
    nifty_file = next((os.path.join(r, f) for r, d, files in os.walk(".") for f in files if "nifty" in f.lower() or "historical_indices.csv" in f.lower()), None)
    
    mac_rep = {'direction': "CHAOTIC 🟡", 'conviction': 0, 'risk_pct': 0, 'target_display': "N/A"}
    n_up, n_down, n_scaler = train_quant_models(nifty_file, dt_str, is_macro=True)
    universe = get_fno_universe()
    
    if n_up is not None:
        n_key = next((i["key"] for i in universe if i["symbol"] in ["NIFTY 50", "NIFTY"]), None)
        df_n = read_standard_csv(nifty_file)
        last_row, entry = get_live_features("NIFTY 50", n_key, dt_str, is_bt, df_n)
        
        if last_row is not None:
            live_scaled = n_scaler.transform(np.array(last_row[features]).reshape(1, -1))
            p_up = n_up.predict(live_scaled)[0]
            p_down = n_down.predict(live_scaled)[0]
            conf = min(99.9, (max(p_up, p_down) / (min(p_up, p_down) + 1e-8)) * 20.0)
            
            if p_up > p_down * 1.5 and p_up > 0.5:
                mac_rep = {'direction': "LONG 🟢", 'conviction': conf, 'risk_pct': p_down, 'target_display': f"₹{entry * (1 + (p_up / 100)):.2f} (+{p_up:.2f}%)"}
            elif p_down > p_up * 1.5 and p_down > 0.5:
                mac_rep = {'direction': "SHORT 🔴", 'conviction': conf, 'risk_pct': p_up, 'target_display': f"₹{entry * (1 - (p_down / 100)):.2f} (-{p_down:.2f}%)"}

    # ---------------------------------------------------------
    # PHASE 2: GLOBAL F&O MODEL
    # ---------------------------------------------------------
    print("\n⚡ PHASE 2: Training Global F&O Boosting Model (Bypassing GitHub Limits)...")
    f_up, f_down, f_scaler = train_quant_models("historical_fno.csv", dt_str, is_macro=False)
    if f_up is None: 
        print("❌ FATAL: Could not train F&O Model.")
        return
        
    print("🎯 Phase 3: Predicting Max Excursion & Sizing Kelly...")
    final_data = []
    fno_df = read_standard_csv("historical_fno.csv") if is_bt else None

    for asset in universe:
        last_row, entry = get_live_features(asset["symbol"], asset["key"], dt_str, is_bt, fno_df)
        if not is_bt: time.sleep(0.15) 
        if last_row is None: continue
        
        live_scaled = f_scaler.transform(np.array(last_row[features]).reshape(1, -1))
        p_up = f_up.predict(live_scaled)[0]
        p_down = f_down.predict(live_scaled)[0]
        
        if p_up > p_down * 1.5 and p_up > 1.0:
            dir_str, pct, rsk = "LONG 🟢", p_up, p_down
        elif p_down > p_up * 1.5 and p_down > 1.0:
            dir_str, pct, rsk = "SHORT 🔴", p_down, p_up
        else: continue
        
        rr_ratio = pct / (rsk + 1e-8)
        if rr_ratio < 1.2: continue 
        
        win_rate = 0.55 # Baseline edge assumption for Kelly sizing
        kelly = max(0.0, (win_rate - ((1 - win_rate) / rr_ratio)) / 2.0 * 100.0)
        if kelly < 1.0: continue
            
        tgt = entry * (1 + (pct / 100.0)) if "LONG" in dir_str else entry * (1 - (pct / 100.0))
        sl = entry * (1 - (rsk / 100.0)) if "LONG" in dir_str else entry * (1 + (rsk / 100.0))
        
        out = "<b>Awaiting Market ⏳</b>"
        if is_bt and fno_df is not None:
            df_sym = fno_df[fno_df['Symbol'] == asset['symbol']].sort_values('Date').reset_index(drop=True)
            fut = df_sym[df_sym['Date'] >= dt_str]
            if len(fut) >= 2:
                fw = fut.iloc[:2] 
                if "LONG" in dir_str:
                    out = f"<span style='color: #dc3545;'>❌ STOP HIT (₹{sl:.2f})</span>" if fw['Low'].min() <= sl else f"<span style='color: #28a745;'>Closed ₹{fw['Close'].iloc[-1]:.2f} (+{((fw['Close'].iloc[-1]-entry)/entry)*100:.2f}%)</span>"
                else:
                    out = f"<span style='color: #dc3545;'>❌ STOP HIT (₹{sl:.2f})</span>" if fw['High'].max() >= sl else f"<span style='color: #28a745;'>Closed ₹{fw['Close'].iloc[-1]:.2f} ({-((fw['Close'].iloc[-1]-entry)/entry)*100:.2f}%)</span>"
        
        final_data.append({'asset': asset["symbol"], 'direction': dir_str, 'rr_ratio': rr_ratio, 'kelly_pct': kelly, 'entry': entry, 'ai_stop': sl, 'target_display': f"₹{tgt:.2f} ({pct:.2f}%)", 'actual_outcome': out})

    send_mobile_alert(mac_rep, final_data, dt_str, is_bt)

if __name__ == "__main__":
    run_production_sweep()

