import os
import smtplib
import urllib.parse
import json
import gzip
import io
import time
import random
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from datetime import datetime, timedelta

import requests
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import faiss
import xgboost as xgb
import warnings

warnings.filterwarnings('ignore')

# ==============================================================================
# 0. DETERMINISTIC ENVIRONMENT LOCK
# ==============================================================================
def set_deterministic_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

# ==============================================================================
# 1. INSTITUTIONAL MULTI-TIMEFRAME FEATURE ENGINE (250-Day Lookback)
# ==============================================================================
def extract_deep_history_features(ohlcv_250):
    """
    Condenses 250 days of structural context into multi-horizon momentum matrices,
    volatility scales, and range locations. Total feature space width: 15 dimensions.
    """
    opens = ohlcv_250[:, 0]
    highs = ohlcv_250[:, 1]
    lows = ohlcv_250[:, 2]
    closes = ohlcv_250[:, 3]
    volumes = ohlcv_250[:, 4]
    
    last_close = closes[-1]
    
    # 🌟 Yearly Macro Scales (250 Days)
    macro_min, macro_max = lows.min(), highs.max()
    pos_in_yearly_range = (last_close - macro_min) / (macro_max - macro_min + 1e-8)
    
    # 🌟 Intermediate Scales (50 Days)
    mid_min, max_mid = lows[-50:].min(), highs[-50:].max()
    pos_in_50d_range = (last_close - mid_min) / (max_mid - mid_min + 1e-8)
    
    # 🌟 Micro Scales (10 Days)
    micro_min, max_micro = lows[-10:].min(), highs[-10:].max()
    pos_in_10d_range = (last_close - micro_min) / (max_micro - micro_min + 1e-8)
    
    # 🌟 Multi-Timeframe Momentum Vectors
    ret_1d = (last_close - closes[-2]) / (closes[-2] + 1e-8) if len(closes) >= 2 else 0
    ret_5d = (last_close - closes[-6]) / (closes[-6] + 1e-8) if len(closes) >= 6 else 0
    ret_20d = (last_close - closes[-21]) / (closes[-21] + 1e-8) if len(closes) >= 21 else 0
    ret_50d = (last_close - closes[-51]) / (closes[-51] + 1e-8) if len(closes) >= 51 else 0
    ret_100d = (last_close - closes[-101]) / (closes[-101] + 1e-8) if len(closes) >= 101 else 0
    ret_250d = (last_close - closes[0]) / (closes[0] + 1e-8)
    
    # 🌟 Volatility Baselines (Short vs Long Term)
    daily_returns_250 = np.diff(closes) / (closes[:-1] + 1e-8)
    vol_long = np.std(daily_returns_250) if len(daily_returns_250) > 0 else 1e-8
    vol_short = np.std(daily_returns_250[-10:]) if len(daily_returns_250) >= 10 else vol_long
    vol_ratio = vol_short / (vol_long + 1e-8)
    
    # 🌟 Volume Outflows & Expansion Tracking
    mean_vol_250 = volumes.mean()
    mean_vol_10 = volumes[-10:].mean()
    vol_expansion_long = volumes[-1] / (mean_vol_250 + 1e-8)
    vol_expansion_short = mean_vol_10 / (mean_vol_250 + 1e-8)

    features = [
        pos_in_yearly_range, pos_in_50d_range, pos_in_10d_range,
        ret_1d, ret_5d, ret_20d, ret_50d, ret_100d, ret_250d,
        vol_long, vol_short, vol_ratio,
        vol_expansion_long, vol_expansion_short,
        closes[-1] / (opens[-1] + 1e-8) # Current daily body structure
    ]
    return np.array(features, dtype=np.float32)

# ==============================================================================
# 2. DATA PROCESSING ENGINE (250-Day Memory Mapping)
# ==============================================================================
def read_and_standardize_csv(filename):
    if not os.path.exists(filename): return None
    df = pd.read_csv(filename)
    rename_map = {}
    
    for c in df.columns:
        cl = str(c).lower().strip()
        if cl in ['date', 'time', 'timestamp']: rename_map[c] = 'Date'
        elif cl in ['symbol', 'ticker', 'asset']: rename_map[c] = 'Symbol'
        elif cl == 'open': rename_map[c] = 'Open'
        elif cl == 'high': rename_map[c] = 'High'
        elif cl == 'low': rename_map[c] = 'Low'
        elif cl == 'close': rename_map[c] = 'Close'
        elif cl in ['volume', 'vol']: rename_map[c] = 'Volume'
        
    df = df.rename(columns=rename_map)
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce', format='mixed').dt.strftime('%Y-%m-%d')
        df = df.dropna(subset=['Date'])
    return df

def load_training_data(csv_filename, target_date_str=None, min_pct=1.0):
    df = read_and_standardize_csv(csv_filename)
    if df is None or 'Date' not in df.columns: return None, None, None
        
    if target_date_str:
        df = df[df['Date'] <= target_date_str]
    
    features_matrices, price_targets, risk_targets = [], [], []
    LOOKBACK = 250
    FUTURE_DAYS = 2 
    
    if "historical_indices" in csv_filename.lower() or "nifty" in csv_filename.lower():
        if 'Symbol' in df.columns:
            mask = df['Symbol'].astype(str).str.upper().str.replace("_", "").str.replace(" ", "").str.contains("NIFTY50|NIFTY")
            if mask.any(): df = df[mask]

    for symbol, group in df.groupby('Symbol') if 'Symbol' in df.columns else [('ASSET', df)]:
        group = group.sort_values('Date').reset_index(drop=True)
        values = group[['Open', 'High', 'Low', 'Close', 'Volume']].values.astype(np.float32)
        
        if len(values) < (LOOKBACK + FUTURE_DAYS): continue
            
        for i in range(len(values) - (LOOKBACK + FUTURE_DAYS) + 1):
            raw_window = values[i : i+LOOKBACK]
            entry_price = values[i+LOOKBACK, 0] # Day T+1 Open (9:15 AM Anchor)
            if entry_price <= 0: continue
            
            future_closes = values[i+LOOKBACK : i+LOOKBACK+FUTURE_DAYS, 3]
            future_highs  = values[i+LOOKBACK : i+LOOKBACK+FUTURE_DAYS, 1]
            future_lows   = values[i+LOOKBACK : i+LOOKBACK+FUTURE_DAYS, 2]
            
            max_close = future_closes.max()
            min_close = future_closes.min()
            
            if (max_close - entry_price) > (entry_price - min_close):
                actual_pct_move = ((max_close - entry_price) / entry_price) * 100.0
                adverse_excursion = abs((future_lows.min() - entry_price) / entry_price) * 100.0
            else:
                actual_pct_move = ((min_close - entry_price) / entry_price) * 100.0
                adverse_excursion = abs((future_highs.max() - entry_price) / entry_price) * 100.0
                
            if abs(actual_pct_move) < min_pct or entry_price < 10.0: continue
                
            features_matrices.append(extract_deep_history_features(raw_window))
            price_targets.append(actual_pct_move)
            risk_targets.append(adverse_excursion)
            
    if len(features_matrices) == 0: return None, None, None
    return np.array(features_matrices, dtype=np.float32), np.array(price_targets, dtype=np.float32), np.array(risk_targets, dtype=np.float32)

# ==============================================================================
# 3. STATISTICAL AI COGNITION ENGINE
# ==============================================================================
def train_quantitative_model(X_features, Y_price, Y_risk):
    xgb_price = xgb.XGBRegressor(n_estimators=150, learning_rate=0.05, max_depth=5, random_state=42).fit(X_features, Y_price)
    xgb_risk = xgb.XGBRegressor(n_estimators=150, learning_rate=0.05, max_depth=5, random_state=42).fit(X_features, Y_risk)

    features_contig = np.ascontiguousarray(X_features, dtype=np.float32)
    faiss.normalize_L2(features_contig)
    
    index = faiss.IndexFlatIP(features_contig.shape[1]) 
    index.add(features_contig)
    return xgb_price, xgb_risk, index, Y_price

# ==============================================================================
# 4. DEEP LIVE INGESTION INTERFACES
# ==============================================================================
def fetch_915_open_from_upstox(instrument_key, target_date_str):
    access_token = os.environ.get("UPSTOX_ACCESS_TOKEN")
    if not access_token or not instrument_key: return None
    encoded_key = urllib.parse.quote(instrument_key)
    url = f"https://api.upstox.com/v2/historical-candle/intraday/{encoded_key}/1minute"
    headers = {'Accept': 'application/json', 'Authorization': f'Bearer {access_token}'}
    try:
        response = requests.get(url, headers=headers, timeout=5)
        if response.status_code == 200:
            candles = response.json().get('data', {}).get('candles', [])
            if candles:
                for c in candles:
                    if target_date_str in str(c[0]) and "09:15" in str(c[0]): return float(c[1]) 
                return float(candles[-1][1]) 
    except: pass
    return None

def fetch_upstox_data_with_250_lookback(instrument_key, target_date_str, lookback_days=380):
    access_token = os.environ.get("UPSTOX_ACCESS_TOKEN")
    if not access_token: return None, None
    
    target_dt = datetime.strptime(target_date_str, "%Y-%m-%d")
    to_date = (target_dt - timedelta(days=1)).strftime("%Y-%m-%d")
    from_date = (target_dt - timedelta(days=lookback_days)).strftime("%Y-%m-%d")
    
    encoded_key = urllib.parse.quote(instrument_key)
    url = f"https://api.upstox.com/v2/historical-candle/{encoded_key}/day/{to_date}/{from_date}"
    headers = {'Accept': 'application/json', 'Authorization': f'Bearer {access_token}'}
    
    response = requests.get(url, headers=headers)
    if response.status_code != 200: return None, None
        
    data = response.json().get('data', {}).get('candles', [])
    if not data or len(data) < 250: return None, None
        
    ohlcv = np.array([candle[1:6] for candle in data], dtype=np.float32)[::-1] 
    ohlcv_250 = ohlcv[-250:]
    
    entry_price = fetch_915_open_from_upstox(instrument_key, target_date_str)
    if entry_price is None: entry_price = float(ohlcv_250[-1][3])
        
    features = extract_deep_history_features(ohlcv_250)
    return features, entry_price

def get_fno_live_features(asset_symbol, asset_key, target_date_str, is_backtest, df_full=None):
    if is_backtest and df_full is not None:
        df_sym = df_full[df_full['Symbol'] == asset_symbol].sort_values('Date').reset_index(drop=True)
        df_history = df_sym[df_sym['Date'] < target_date_str]
        if len(df_history) < 250: return None, None
        
        values = df_history[['Open', 'High', 'Low', 'Close', 'Volume']].values.astype(np.float32)[-250:]
        df_future = df_sym[df_sym['Date'] >= target_date_str]
        if df_future.empty: return None, None
        
        entry_price = float(df_future.iloc[0]['Open']) 
        features = extract_deep_history_features(values)
        return features, entry_price
    else:
        return fetch_upstox_data_with_250_lookback(asset_key, target_date_str)

def get_live_nifty_features_from_csv(csv_filename, target_date_str, instrument_key=None, is_backtest=False):
    df = read_and_standardize_csv(csv_filename)
    if df is None or 'Date' not in df.columns: return None, None
    
    if 'Symbol' in df.columns:
        mask = df['Symbol'].astype(str).str.upper().str.replace("_", "").str.replace(" ", "").str.contains("NIFTY50|NIFTY")
        if mask.any(): df = df[mask]
        else: df = df[df['Symbol'] == df['Symbol'].unique()[0]]
            
    df_history = df[df['Date'] < target_date_str].sort_values('Date').reset_index(drop=True)
    if len(df_history) < 250: return None, None
    
    values = df_history[['Open', 'High', 'Low', 'Close', 'Volume']].values.astype(np.float32)[-250:]
    close_t = values[-1, 3] 
    
    entry_price = None
    df_future = df[df['Date'] >= target_date_str].sort_values('Date').reset_index(drop=True)
    
    if not is_backtest and instrument_key:
        entry_price = fetch_915_open_from_upstox(instrument_key, target_date_str)
        
    if entry_price is None and not df_future.empty: entry_price = float(df_future.iloc[0]['Open'])
    if entry_price is None: entry_price = close_t
        
    features = extract_deep_history_features(values)
    return features, entry_price

def get_dynamic_fno_universe():
    try:
        response = requests.get("https://assets.upstox.com/market-quote/instruments/exchange/NSE.json.gz", timeout=10)
        if response.status_code != 200: return []
        nse_data = json.load(gzip.GzipFile(fileobj=io.BytesIO(response.content)))
        fno_underlying = {item.get("underlying_symbol") for item in nse_data if item.get("segment") == "NSE_FO" and item.get("underlying_symbol")}
        return [{"symbol": item.get("trading_symbol"), "key": item.get("instrument_key")} for item in nse_data if item.get("segment") in ("NSE_EQ", "NSE_INDEX") and item.get("trading_symbol") in fno_underlying]
    except: return []

# ==============================================================================
# 5. DUAL-BRAIN MASTER EXECUTION & DISPATCH
# ==============================================================================
def send_mobile_alert(macro_data, fno_data_list, target_date_str, is_backtest):
    sender_email, sender_pass, recipient_email = os.environ.get("SENDER_EMAIL"), os.environ.get("SENDER_PASSWORD"), os.environ.get("RECIPIENT_EMAIL")
    if not all([sender_email, sender_pass, recipient_email]): return

    msg = MIMEMultipart('alternative')
    prefix = "⏪ BACKTEST" if is_backtest else "🚀 LIVE 250D-MACRO AI ALERT"
    msg['Subject'] = f"{prefix} | {target_date_str}"
    msg['From'], msg['To'] = sender_email, recipient_email

    macro_color = "#28a745" if "LONG" in macro_data['direction'] else "#dc3545"
    sim_warning = f"<div style='background-color: #fff3cd; color: #856404; padding: 10px; text-align: center; font-weight: bold; margin-bottom: 15px;'>⚠️ VALIDATION MODE: SHOWING ACTUAL 2-DAY OUTCOMES</div>" if is_backtest else ""

    html_content = f"""
    <html>
      <body style="font-family: Arial, sans-serif; background-color: #f4f7f6; padding: 10px;">
        {sim_warning}
        <div style="background-color: white; padding: 15px; border-radius: 8px; margin-bottom: 20px; border-left: 6px solid {macro_color};">
            <h3 style="margin-top: 0; color: #333;">🌍 MACRO REGIME (NIFTY 50 - 250D LOOKBACK)</h3>
            <p style="font-size: 16px; color: #333; margin: 5px 0;">
                <b>AI Prediction:</b> <span style="color: {macro_color}; font-weight: bold;">{macro_data['direction']}</span><br>
                <b>AI Target:</b> {macro_data['target_display']} | <b>AI Max Adverse Risk:</b> {macro_data['risk_pct']:.2f}%<br>
                <b>Pattern Similarity Score:</b> {macro_data['conviction']:.2f}%
            </p>
        </div>
        <h3 style="color: #333;">⚡ MICRO F&O SWEEP (250D STRUCTURAL FILTER)</h3>
        <table border="1" cellpadding="8" cellspacing="0" style="border-collapse: collapse; width: 100%; text-align: center; font-size: 14px; background-color: white;">
          <tr bgcolor="#f8f9fa" style="color: #333; font-weight: bold;">
            <th>Asset</th><th>Signal</th><th>Pattern Fit</th><th>Entry Price (9:15)</th><th>AI Max Risk</th><th>AI Projected Target</th><th>Result</th>
          </tr>"""
    
    fno_data_list.sort(key=lambda x: x['conviction'], reverse=True)
    for row in fno_data_list:
        dir_color = "#28a745" if "LONG" in row['direction'] else "#dc3545"
        html_content += f"""
          <tr>
            <td style="color: #0056b3;"><b>{row['asset']}</b></td>
            <td style="color: {dir_color}; font-weight: bold;">{row['direction']}</td>
            <td>{row['conviction']:.2f}%</td>
            <td>₹{row['entry']:.2f}</td>
            <td style="color: #dc3545;">₹{row['ai_stop']:.2f} (-{row['risk_pct']:.2f}%)</td>
            <td style="color: {dir_color}; font-weight: bold;">{row['target_display']}</td>
            <td>{row['actual_outcome']}</td>
          </tr>"""
        
    html_content += "</table></body></html>"
    msg.attach(MIMEText(html_content, 'html'))
    try:
        server = smtplib.SMTP('smtp.gmail.com', 587); server.starttls(); server.login(sender_email, sender_pass)
        server.sendmail(sender_email, recipient_email, msg.as_string()); server.quit()
        print(f"✅ 250D Macro Report Dispatched with {len(fno_data_list)} high-fidelity structures.")
    except Exception as e: print(f"Failed to send email: {str(e)}")

def run_production_sweep():
    set_deterministic_seeds(42)
    target_date_str = os.environ.get("PARAM_BACKTEST_DATE", "").strip()
    is_backtest = bool(target_date_str)
    if not is_backtest: target_date_str = datetime.now().strftime("%Y-%m-%d")
        
    print(f"⚙️ EXECUTING 250-DAY MACRO ENGINE | DATE: {target_date_str}")
    
    nifty_file = None
    for root, dirs, files in os.walk("."):
        for file in files:
            if "nifty" in file.lower() and file.lower().endswith(".csv"): nifty_file = os.path.join(root, file); break
            elif "historical_indices.csv" in file.lower(): nifty_file = os.path.join(root, file); break
        if nifty_file: break

    # ==========================================
    # PHASE 1: MACRO NIFTY 250-DAY BRAIN
    # ==========================================
    print(f"\n🧠 PHASE 1: Compiling NIFTY 50 1-Year Structural Memory Horizon...")
    X_nifty, Y_np, Y_nr = load_training_data(nifty_file, target_date_str, min_pct=0.5)
    
    if X_nifty is None or len(X_nifty) == 0:
        print("❌ Nifty deep feature construction rejected due to insufficient timeline density.")
        return

    nifty_xgb_p, nifty_xgb_r, nifty_faiss, _ = train_quantitative_model(X_nifty, Y_np, Y_nr)
    fno_universe = get_dynamic_fno_universe()
    nifty_key = next((item["key"] for item in fno_universe if item["symbol"] in ["NIFTY 50", "NIFTY"]), None)
    
    nifty_live_features, nifty_entry = get_live_nifty_features_from_csv(nifty_file, target_date_str, nifty_key, is_backtest)
    
    if nifty_live_features is not None:
        live_feat_arr = np.ascontiguousarray(nifty_live_features.reshape(1, -1), dtype=np.float32)
        n_pct = float(nifty_xgb_p.predict(live_feat_arr)[0])
        n_risk_pct = float(nifty_xgb_r.predict(live_feat_arr)[0])
        
        faiss.normalize_L2(live_feat_arr)
        n_score, _ = nifty_faiss.search(live_feat_arr, k=5)
        n_conviction = (max(0.0, n_score[0][0]) ** 0.5) * 100.0
        
        macro_report = {
            'direction': "LONG 🟢" if n_pct > 0 else "SHORT 🔴", 'conviction': n_conviction, 'risk_pct': n_risk_pct,
            'target_display': f"₹{nifty_entry * (1 + (n_pct / 100)):.2f} ({'+' if n_pct>0 else ''}{n_pct:.2f}%)"
        }
        print(f"🌍 250D MACRO REGIME: {macro_report['direction']} (Expected Adverse Excursion Limit: {n_risk_pct:.2f}%)")
    else:
        macro_report = {'direction': "UNKNOWN", 'conviction': 0, 'risk_pct': 0, 'target_display': "N/A"}

    # ==========================================
    # PHASE 2: MICRO F&O 250-DAY BRAIN
    # ==========================================
    print("\n⚡ PHASE 2: Re-Indexing F&O Universe Across 1-Year Structural Matrices...")
    X_fno, Y_fp, Y_fr = load_training_data("historical_fno.csv", target_date_str, min_pct=1.0)
    if X_fno is None or len(X_fno) == 0: return

    fno_xgb_p, fno_xgb_r, fno_faiss, fno_historical_y = train_quantitative_model(X_fno, Y_fp, Y_fr)
    
    print("🎯 Phase 3: Sweeping Active Market Universe...")
    if not fno_universe: return
    
    final_report_data = []
    min_conviction = float(os.environ.get("PARAM_MIN_CONVICTION", 82.50)) # Recalibrated baseline for 15D deep-space flat indexes
    fno_df_full = read_and_standardize_csv("historical_fno.csv") if is_backtest else None

    for asset in fno_universe:
        live_features, entry_price = get_fno_live_features(asset["symbol"], asset["key"], target_date_str, is_backtest, fno_df_full)
        if not is_backtest: time.sleep(0.15) 
        if live_features is None or entry_price is None: continue
        
        live_feat_arr = np.ascontiguousarray(live_features.reshape(1, -1), dtype=np.float32)
        
        pred_pct = float(fno_xgb_p.predict(live_feat_arr)[0])
        pred_risk_pct = float(fno_xgb_r.predict(live_feat_arr)[0])
        
        if abs(pred_pct) < 1.0: continue 
            
        direction = "LONG 🟢" if pred_pct > 0 else "SHORT 🔴"
        target_price = entry_price * (1 + (pred_pct / 100.0))
        ai_stop_loss = entry_price * (1 - (pred_risk_pct / 100.0)) if pred_pct > 0 else entry_price * (1 + (pred_risk_pct / 100.0))
        
        if abs(target_price - entry_price) < abs(entry_price - ai_stop_loss): continue 

        faiss.normalize_L2(live_feat_arr)
        k_search = min(5, fno_faiss.ntotal)
        if k_search == 0: continue
            
        scores, indices = fno_faiss.search(live_feat_arr, k_search)
        cosine_sim = max(0.0, scores[0][0])
        final_conviction = (cosine_sim ** 0.5) * 100.0 
        
        # AI Chaos Filter
        historical_outcomes = fno_historical_y[indices[0]]
        if np.std(historical_outcomes) > abs(pred_pct): continue
        
        if final_conviction >= min_conviction:
            print(f"   🌟 [MATCH FOUND] {asset['symbol']} | Structural Alignment: {final_conviction:.1f}%")
            outcome_text = "<b>Awaiting Market ⏳</b>"
            
            if is_backtest and fno_df_full is not None:
                df_sym = fno_df_full[fno_df_full['Symbol'] == asset['symbol']].sort_values('Date').reset_index(drop=True)
                df_future = df_sym[df_sym['Date'] >= target_date_str]
                if len(df_future) >= 2:
                    fw = df_future.iloc[:2] 
                    mx, mn = fw['High'].max(), fw['Low'].min()
                    if "LONG" in direction:
                        outcome_text = f"<span style='color: #dc3545;'>❌ RISK CAP HIT (₹{ai_stop_loss:.2f})</span>" if fw['Low'].min() <= ai_stop_loss else f"<span style='color: #28a745;'>Closed ₹{fw['Close'].iloc[-1]:.2f} (+{((fw['Close'].iloc[-1]-entry_price)/entry_price)*100:.2f}%)</span>"
                    else:
                        outcome_text = f"<span style='color: #dc3545;'>❌ RISK CAP HIT (₹{ai_stop_loss:.2f})</span>" if fw['High'].max() >= ai_stop_loss else f"<span style='color: #28a745;'>Closed ₹{fw['Close'].iloc[-1]:.2f} ({((fw['Close'].iloc[-1]-entry_price)/entry_price)*100:.2f}%)</span>"
            
            final_report_data.append({
                'asset': asset["symbol"], 'direction': direction, 'conviction': final_conviction,
                'entry': float(entry_price), 'ai_stop': float(ai_stop_loss), 'risk_pct': float(pred_risk_pct),
                'target_display': f"₹{target_price:.2f} ({'+' if pred_pct>0 else ''}{pred_pct:.2f}%)", 'actual_outcome': outcome_text
            })

    if not final_report_data: print(f"\n⚠️ Result: No F&O configurations matching yearly structural states today.")
    send_mobile_alert(macro_report, final_report_data, target_date_str, is_backtest)

if __name__ == "__main__":
    run_production_sweep()
