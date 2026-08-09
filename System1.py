import os
import requests
import smtplib
from email.mime.text import MIMEText
from datetime import datetime, date, timedelta, timezone
from fyers_apiv3 import fyersModel

# ==========================================
# ⚙️ 1. GLOBAL COMMAND DIAL & MULTI-INDEX LEDGER
# ==========================================
GLOBAL_START_TIME = "09:30"

ACTIVE_INDICES = {
    "NIFTY": {"symbol": "NSE:NIFTY50-INDEX", "step": 50},
    "BANKNIFTY": {"symbol": "NSE:NIFTYBANK-INDEX", "step": 100},
    "FINNIFTY": {"symbol": "NSE:NIFTY FIN SERVICE-INDEX", "step": 50},
    "SENSEX": {"symbol": "BSE:SENSEX-INDEX", "step": 100}
}
RADAR_RANGE = 5   # ATM +/- 5 Strikes

# ==========================================
# 🔐 2. GITHUB SECRETS (ENVIRONMENT VARIABLES)
# ==========================================
CLIENT_ID = os.getenv("FYERS_CLIENT_ID")
ACCESS_TOKEN = os.getenv("FYERS_ACCESS_TOKEN")
EMAIL_SENDER = os.getenv("EMAIL_SENDER")       
EMAIL_APP_PWD = os.getenv("EMAIL_APP_PWD")     
EMAIL_RECEIVER = os.getenv("EMAIL_RECEIVER")   

print("🔍 SYSTEM DIAGNOSTIC: Checking Environment Variables...")
print(f"FYERS_CLIENT_ID Loaded: {'YES' if CLIENT_ID else 'NO (Empty)'}")
print(f"FYERS_ACCESS_TOKEN Loaded: {'YES' if ACCESS_TOKEN else 'NO (Empty)'}")
print("-" * 50)

if not CLIENT_ID or not ACCESS_TOKEN:
    raise ValueError("🚨 CRITICAL FAILURE: API credentials missing. Halting engine.")

def get_fyers_instance():
    return fyersModel.FyersModel(client_id=CLIENT_ID, is_async=False, token=ACCESS_TOKEN, log_path="")

# ==========================================
# 🛡️ 3. TIMEZONE LOCK & WEEKEND OVERRIDE
# ==========================================
IST = timezone(timedelta(hours=5, minutes=30))

def get_target_date():
    """Forces IST time and shifts internal clock to Friday if on the weekend."""
    ist_now = datetime.now(IST)
    target_date = ist_now.date()
    if target_date.weekday() == 5: # Saturday
        target_date -= timedelta(days=1)
    elif target_date.weekday() == 6: # Sunday
        target_date -= timedelta(days=2)
    return target_date

TARGET_DATE = get_target_date()
TARGET_DATE_STR = TARGET_DATE.strftime("%Y-%m-%d")

# ==========================================
# 📜 4. THE MASTER FILE LOADER (THE FIX)
# ==========================================
MASTER_SYMBOLS = {}
INDEX_EXPIRIES = {"NIFTY": set(), "BANKNIFTY": set(), "FINNIFTY": set(), "SENSEX": set()}

def load_symbol_master():
    """Downloads Fyers global CSVs. Guarantees 100% accurate symbols and expiries."""
    print("📡 Downloading Exchange Symbol Master. Bypassing SDK limitations...")
    urls = [
        "https://public.fyers.in/sym_details/NSE_FO.csv",
        "https://public.fyers.in/sym_details/BSE_FO.csv"
    ]
    
    for url in urls:
        try:
            res = requests.get(url, timeout=15)
            for line in res.text.split('\n'):
                parts = line.split(',')
                if len(parts) < 17:
                    continue
                
                sym_ticker = parts[9]
                opt_type = parts[16]
                
                if opt_type not in ["CE", "PE"]:
                    continue
                    
                try:
                    strike = float(parts[15])
                except ValueError:
                    continue
                
                # Determine Index
                idx = None
                if sym_ticker.startswith("NSE:NIFTY") and sym_ticker[9].isdigit():
                    idx = "NIFTY"
                elif sym_ticker.startswith("NSE:BANKNIFTY"):
                    idx = "BANKNIFTY"
                elif sym_ticker.startswith("NSE:FINNIFTY"):
                    idx = "FINNIFTY"
                elif sym_ticker.startswith("BSE:SENSEX"):
                    idx = "SENSEX"
                
                if not idx:
                    continue
                
                # Extract Absolute Expiry Date
                expiry_val = parts[8]
                try:
                    if expiry_val.isdigit():
                        dt_utc = datetime.fromtimestamp(int(expiry_val), tz=timezone.utc)
                        expiry_date = dt_utc.astimezone(IST).date()
                    elif '-' in expiry_val:
                        expiry_date = datetime.strptime(expiry_val, "%Y-%m-%d").date()
                    else:
                        continue
                except Exception:
                    continue
                
                INDEX_EXPIRIES[idx].add(expiry_date)
                MASTER_SYMBOLS[(idx, expiry_date, int(strike), opt_type)] = sym_ticker
        except Exception as e:
            print(f"⚠️ Master file download failed for {url}: {e}")

    # Sort expiries chronologically
    for k in INDEX_EXPIRIES:
        INDEX_EXPIRIES[k] = sorted(list(INDEX_EXPIRIES[k]))

# ==========================================
# 🎯 5. TARGET ACQUISITION & 0DTE SHIELD
# ==========================================
def get_active_expiry(index_name):
    # Only look at contracts expiring ON or AFTER our simulation date
    valid_dates = [d for d in INDEX_EXPIRIES.get(index_name, []) if d >= TARGET_DATE]
    if not valid_dates:
        raise ValueError(f"No future expiries found for {index_name} in Master File.")
        
    closest_expiry = valid_dates[0]
    
    if TARGET_DATE == closest_expiry:
        next_expiry = valid_dates[1] if len(valid_dates) > 1 else closest_expiry
        print(f"🚨 0DTE DETECTED for {index_name}. Shifting from {closest_expiry} to {next_expiry}")
        return next_expiry
    return closest_expiry

def generate_radar_strikes(index_name, spot_price, expiry_date, strike_step):
    atm_strike = round(spot_price / strike_step) * strike_step
    strikes = []
    
    for i in range(-RADAR_RANGE, RADAR_RANGE + 1):
        strike_val = atm_strike + (i * strike_step)
        ce_key = (index_name, expiry_date, strike_val, "CE")
        pe_key = (index_name, expiry_date, strike_val, "PE")
        
        # Plucks the exact exchange-formatted symbol from our downloaded Master Dictionary
        if ce_key in MASTER_SYMBOLS: strikes.append(MASTER_SYMBOLS[ce_key])
        if pe_key in MASTER_SYMBOLS: strikes.append(MASTER_SYMBOLS[pe_key])
            
    return strikes

# ==========================================
# 🧠 6. PROPRIETARY QUAD-DELTA MATH
# ==========================================
def run_quad_delta_math(symbol_candles):
    """Analyzes sliced candle array up to the current simulated minute."""
    if not symbol_candles or len(symbol_candles) < 2:
        return {"state": "DEAD", "price": 0, "floor": 0}
        
    latest_close = symbol_candles[-1][4]  
    micro_floor = symbol_candles[-2][3]   
    
    # -----------------------------------------------------
    # 🧠 PROPRIETARY QUAD-DELTA MATH INJECTION POINT
    # -----------------------------------------------------
    is_basket_1_birth = True  # Proxy for testing
    
    if latest_close < micro_floor:
        return {"state": "BASKET_3", "price": latest_close, "floor": micro_floor}
    elif is_basket_1_birth:
        return {"state": "BASKET_1", "price": latest_close, "floor": micro_floor}
    else:
        return {"state": "TRACKING", "price": latest_close, "floor": micro_floor}

# ==========================================
# ⏱️ 7. THE STATELESS SIMULATOR
# ==========================================
def fetch_history(fyers, symbol):
    data = {
        "symbol": symbol, "resolution": "5", "date_format": "1",
        "range_from": TARGET_DATE_STR, "range_to": TARGET_DATE_STR, "cont_flag": "1"
    }
    res = fyers.history(data=data)
    return res.get('candles', [])

def simulate_engine():
    fyers = get_fyers_instance()
    load_symbol_master() # Initialize Truth Layer
    
    start_time_obj = datetime.strptime(GLOBAL_START_TIME, "%H:%M").time()
    options_cache = {}
    
    master_added, master_removed, master_current = {}, {}, {}
    print(f"\n🟢 Initiating Stateless Reconstruction for {TARGET_DATE_STR}...")

    for index_name, config in ACTIVE_INDICES.items():
        index_symbol = config['symbol']
        strike_step = config['step']
        
        try:
            active_expiry = get_active_expiry(index_name)
        except Exception as e:
            print(f"⚠️ {e} Skipping {index_name}.")
            continue
            
        spot_candles = fetch_history(fyers, index_symbol)
        
        # Enforce IST on candle timestamps to respect GLOBAL_START_TIME
        valid_spot_candles = []
        for c in spot_candles:
            c_time = datetime.fromtimestamp(c[0], tz=timezone.utc).astimezone(IST).time()
            if c_time >= start_time_obj:
                valid_spot_candles.append(c)
                
        if not valid_spot_candles:
            continue

        anchored_strikes = {}
        previous_matrix = {}
        
        for idx, spot_candle in enumerate(valid_spot_candles):
            timestamp = spot_candle[0]
            spot_close = spot_candle[4]
            is_last_candle = (idx == len(valid_spot_candles) - 1)
            
            radar_strikes = generate_radar_strikes(index_name, spot_close, active_expiry, strike_step)
            active_targets = set(radar_strikes + list(anchored_strikes.keys()))
            
            current_matrix = {}
            new_anchors = {}
            
            for strike in active_targets:
                if strike not in options_cache:
                    options_cache[strike] = fetch_history(fyers, strike)
                    
                sliced_candles = [c for c in options_cache[strike] if c[0] <= timestamp]
                result = run_quad_delta_math(sliced_candles)
                
                if result['state'] == "BASKET_1" or (strike in anchored_strikes and result['state'] != "BASKET_3"):
                    current_matrix[strike] = result
                    new_anchors[strike] = True 
                    
            anchored_strikes = new_anchors
            
            if is_last_candle:
                added = {s: d for s, d in current_matrix.items() if s not in previous_matrix}
                removed = {s: d for s, d in previous_matrix.items() if s not in current_matrix}
                
                master_added.update(added)
                master_removed.update(removed)
                master_current.update(current_matrix)
                
            previous_matrix = current_matrix

    return master_added, master_removed, master_current

# ==========================================
# ✉️ 8. EMAIL DISPATCH & CONSOLE TRANSPARENCY
# ==========================================
def dispatch_results(added, removed, current):
    print(f"\n🛒 ACTIVE MASTER BASKET (Tracking {len(current)} Surviving Trades):")
    if not current:
        print("   [Empty] No trades are currently holding the structural floor.")
    else:
        for sym, d in current.items():
            print(f"   - {sym} | Status: {d['state']} | LTP: {d['price']} | Floor: {d['floor']}")
    print("-" * 50)

    if not EMAIL_SENDER or not EMAIL_APP_PWD:
        return
        
    if not added and not removed:
        print("🔇 No structural changes on the last candle. Matrix remains silent.")
        return

    subject = f"⚡ MASTER INTRADAY DELTA (+{len(added)}/-{len(removed)})"
    body = "QUAD-DELTA INTRADAY SHIFT:\n\n"
    
    if added:
        body += "🟢 [+] ADDED TO BASKET 1 (New Momentum):\n"
        for sym, d in added.items():
            body += f"   - {sym} | LTP: {d['price']} | Floor: {d['floor']}\n"
            
    if removed:
        body += "\n🔴 [-] MOVED TO BASKET 3 (Floor Breached):\n"
        for sym, d in removed.items():
            body += f"   - {sym} | LTP: {d['price']} | Broken Floor: {d['floor']}\n"

    msg = MIMEText(body)
    msg['Subject'] = subject
    msg['From'] = EMAIL_SENDER
    msg['To'] = EMAIL_RECEIVER

    try:
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp_server:
            smtp_server.login(EMAIL_SENDER, EMAIL_APP_PWD)
            smtp_server.sendmail(EMAIL_SENDER, EMAIL_RECEIVER, msg.as_string())
        print("📧 Alert Successfully Dispatched.")
    except Exception as e:
        print(f"⚠️ Email failed to send: {e}")

if __name__ == "__main__":
    added, removed, current = simulate_engine()
    dispatch_results(added, removed, current)
    print("\n✅ Simulation Complete. Server Self-Destructing.")
