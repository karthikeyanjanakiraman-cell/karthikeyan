import os
import smtplib
from email.mime.text import MIMEText
from datetime import datetime, date, timedelta
from fyers_apiv3 import fyersModel

# ==========================================
# ⚙️ 1. GLOBAL COMMAND DIAL & MULTI-INDEX LEDGER
# ==========================================
GLOBAL_START_TIME = "09:30"

ACTIVE_INDICES = {
    "NIFTY": {
        "symbol": "NSE:NIFTY50-INDEX",
        "step": 50
    },
    "BANKNIFTY": {
        "symbol": "NSE:NIFTYBANK-INDEX",
        "step": 100
    },
    "FINNIFTY": {
        "symbol": "NSE:NIFTY FIN SERVICE-INDEX",
        "step": 50
    },
    "SENSEX": {
        "symbol": "BSE:SENSEX-INDEX",
        "step": 100
    }
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
print(f"EMAIL_SENDER Loaded: {'YES' if EMAIL_SENDER else 'NO (Empty)'}")
print(f"EMAIL_APP_PWD Loaded: {'YES' if EMAIL_APP_PWD else 'NO (Empty)'}")
print(f"EMAIL_RECEIVER Loaded: {'YES' if EMAIL_RECEIVER else 'NO (Empty)'}")
print("-" * 50)

if not CLIENT_ID or not ACCESS_TOKEN:
    raise ValueError("🚨 CRITICAL FAILURE: API credentials missing. Halting engine.")

def get_fyers_instance():
    return fyersModel.FyersModel(client_id=CLIENT_ID, is_async=False, token=ACCESS_TOKEN, log_path="")

# ==========================================
# 🛡️ 3. THE WEEKEND OVERRIDE & API EXPIRY TRUTH
# ==========================================
def get_target_date():
    """Shifts internal clock back to Friday if running on the weekend."""
    target_date = date.today()
    if target_date.weekday() == 5: # Saturday
        target_date -= timedelta(days=1)
    elif target_date.weekday() == 6: # Sunday
        target_date -= timedelta(days=2)
    return target_date

TARGET_DATE = get_target_date()
TARGET_DATE_STR = TARGET_DATE.strftime("%Y-%m-%d")

def format_fyers_symbol(index_name, expiry_date, strike, option_type):
    """Accurately formats NSE/BSE option symbols for Fyers."""
    year_str = str(expiry_date.year)[-2:]
    month = expiry_date.month
    day_str = f"{expiry_date.day:02d}"
    
    # Fyers uses 1-9 for Jan-Sep, O, N, D for Oct, Nov, Dec
    month_str = str(month) if month < 10 else {10: 'O', 11: 'N', 12: 'D'}[month]
    
    prefix = "BSE" if index_name == "SENSEX" else "NSE"
    return f"{prefix}:{index_name}{year_str}{month_str}{day_str}{strike}{option_type}"

def get_real_dynamic_expiry(fyers, index_symbol):
    """Interrogates live Option Chain for actual expiries. 0DTE Shield applied."""
    try:
        response = fyers.optionChain(data={"symbol": index_symbol})
        if 'data' not in response or 'expiryData' not in response['data']:
            raise ValueError(f"Empty expiry data from API for {index_symbol}")
            
        raw_expiries = response['data']['expiryData']
        valid_dates = []
        for exp_data in raw_expiries:
            date_str = exp_data['date'] if 'date' in exp_data else exp_data.get('expiryDate', '')
            try:
                date_obj = datetime.strptime(date_str, "%Y-%m-%d").date()
            except ValueError:
                date_obj = datetime.strptime(date_str, "%d-%b-%Y").date()
            valid_dates.append(date_obj)
            
        valid_dates.sort()
        
        # 0DTE Shield: Compare against the target execution date
        closest_expiry = valid_dates[0]
        if TARGET_DATE == closest_expiry:
            next_expiry = valid_dates[1] if len(valid_dates) > 1 else closest_expiry
            print(f"🚨 0DTE DETECTED for {index_symbol}. Shifting to next real expiry: {next_expiry}")
            return next_expiry
            
        return closest_expiry
    except Exception as e:
        print(f"⚠️ API Option Chain fetch failed for {index_symbol}. Fallback triggered. Error: {e}")
        return TARGET_DATE + timedelta(days=7) # Safety fallback

# ==========================================
# 🧠 4. PROPRIETARY QUAD-DELTA MATH
# ==========================================
def run_quad_delta_math(symbol_candles):
    """Analyzes sliced candle array up to the current simulated minute."""
    if not symbol_candles or len(symbol_candles) < 2:
        return {"state": "DEAD", "price": 0, "floor": 0}
        
    latest_close = symbol_candles[-1][4]  
    micro_floor = symbol_candles[-2][3]   
    
    # -----------------------------------------------------
    # REPLACE THIS BOOL WITH YOUR TRUE VOLUME/SPREAD ALGO
    # -----------------------------------------------------
    is_basket_1_birth = True  # Proxy for testing
    
    if latest_close < micro_floor:
        return {"state": "BASKET_3", "price": latest_close, "floor": micro_floor}
    elif is_basket_1_birth:
        return {"state": "BASKET_1", "price": latest_close, "floor": micro_floor}
    else:
        return {"state": "TRACKING", "price": latest_close, "floor": micro_floor}

# ==========================================
# ⏱️ 5. THE STATELESS SIMULATOR
# ==========================================
def fetch_history(fyers, symbol):
    """Fetches the day's 5-minute history once and caches it."""
    data = {
        "symbol": symbol,
        "resolution": "5",
        "date_format": "1",
        "range_from": TARGET_DATE_STR,
        "range_to": TARGET_DATE_STR,
        "cont_flag": "1"
    }
    res = fyers.history(data=data)
    return res.get('candles', [])

def simulate_engine():
    fyers = get_fyers_instance()
    
    start_time_obj = datetime.strptime(GLOBAL_START_TIME, "%H:%M").time()
    options_cache = {}
    
    # Master Ledgers for Final Consolidation
    master_added = {}
    master_removed = {}
    master_current = {}
    
    print(f"🟢 Initiating Stateless Reconstruction for {TARGET_DATE_STR}...\n")

    for index_name, config in ACTIVE_INDICES.items():
        index_symbol = config['symbol']
        strike_step = config['step']
        
        # 1. Fetch real expiry & Spot history
        active_expiry = get_real_dynamic_expiry(fyers, index_symbol)
        spot_candles = fetch_history(fyers, index_symbol)
        
        # 2. Filter candles to respect GLOBAL_START_TIME
        valid_spot_candles = [c for c in spot_candles if datetime.fromtimestamp(c[0]).time() >= start_time_obj]
        
        if not valid_spot_candles:
            continue

        anchored_strikes = {}
        previous_matrix = {}
        
        # 3. Time-Travel Loop
        for idx, spot_candle in enumerate(valid_spot_candles):
            timestamp = spot_candle[0]
            spot_close = spot_candle[4]
            is_last_candle = (idx == len(valid_spot_candles) - 1)
            
            # The Rolling Radar
            atm_strike = round(spot_close / strike_step) * strike_step
            radar_strikes = []
            for i in range(-RADAR_RANGE, RADAR_RANGE + 1):
                s_val = atm_strike + (i * strike_step)
                radar_strikes.append(format_fyers_symbol(index_name, active_expiry, s_val, "CE"))
                radar_strikes.append(format_fyers_symbol(index_name, active_expiry, s_val, "PE"))
            
            active_targets = set(radar_strikes + list(anchored_strikes.keys()))
            current_matrix = {}
            new_anchors = {}
            
            for strike in active_targets:
                if strike not in options_cache:
                    options_cache[strike] = fetch_history(fyers, strike)
                    
                # Slice history up to 'now' in simulation
                sliced_candles = [c for c in options_cache[strike] if c[0] <= timestamp]
                result = run_quad_delta_math(sliced_candles)
                
                if result['state'] == "BASKET_1" or (strike in anchored_strikes and result['state'] != "BASKET_3"):
                    current_matrix[strike] = result
                    new_anchors[strike] = True 
                    
            anchored_strikes = new_anchors
            
            # Delta logic on the very last candle
            if is_last_candle:
                added = {s: d for s, d in current_matrix.items() if s not in previous_matrix}
                removed = {s: d for s, d in previous_matrix.items() if s not in current_matrix}
                
                master_added.update(added)
                master_removed.update(removed)
                master_current.update(current_matrix)
                
            previous_matrix = current_matrix

    return master_added, master_removed, master_current

# ==========================================
# ✉️ 6. EMAIL DISPATCH & CONSOLE TRANSPARENCY
# ==========================================
def dispatch_results(added, removed, current):
    # 1. Console Transparency (Always Prints)
    print(f"\n🛒 ACTIVE MASTER BASKET (Tracking {len(current)} Surviving Trades):")
    if not current:
        print("   [Empty] No trades are currently holding the structural floor.")
    else:
        for sym, d in current.items():
            print(f"   - {sym} | Status: {d['state']} | LTP: {d['price']} | Floor: {d['floor']}")
    print("-" * 50)

    # 2. Email Dispatch (Only sends on Delta)
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
