import os
import smtplib
from email.mime.text import MIMEText
from datetime import datetime, date, timedelta
from fyers_apiv3 import fyersModel

# ==========================================
# ⚙️ 1. GLOBAL COMMAND DIAL & CONFIG
# ==========================================
GLOBAL_START_TIME = "09:30"
INDEX_NAME = "NIFTY"
INDEX_SYMBOL = "NSE:NIFTY50-INDEX"
STRIKE_STEP = 50  
RADAR_RANGE = 5   

# ==========================================
# 🔐 2. GITHUB SECRETS (ENVIRONMENT VARIABLES)
# ==========================================
CLIENT_ID = os.getenv("FYERS_CLIENT_ID")
ACCESS_TOKEN = os.getenv("FYERS_ACCESS_TOKEN")
EMAIL_SENDER = os.getenv("EMAIL_SENDER")       
EMAIL_APP_PWD = os.getenv("EMAIL_APP_PWD")     
EMAIL_RECEIVER = os.getenv("EMAIL_RECEIVER")   

if not CLIENT_ID or not ACCESS_TOKEN:
    raise ValueError("🚨 CRITICAL FAILURE: API credentials missing. Halting engine.")

# ==========================================
# 🛡️ 3. THE 0DTE SHIELD
# ==========================================
def get_fyers_instance():
    return fyersModel.FyersModel(client_id=CLIENT_ID, is_async=False, token=ACCESS_TOKEN, log_path="")

def get_dynamic_expiry():
    """
    Simulates fetching active expiries from Fyers Symbol Master.
    Rolls to Next Expiry if Today == Closest Expiry.
    """
    days_to_thursday = (3 - date.today().weekday()) % 7
    closest_expiry = date.today() + timedelta(days=days_to_thursday)
    next_expiry = closest_expiry + timedelta(days=7)
    
    if date.today() == closest_expiry:
        print("🚨 0DTE DETECTED: Shifting to Next Expiry to avoid Theta decay.")
        return next_expiry
    
    return closest_expiry

def format_fyers_date(dt):
    # Fyers format example: 26813 (Year: 26, Month: 8, Day: 13)
    return dt.strftime("%y%#m%d")

# ==========================================
# 🎯 4. TARGET ACQUISITION
# ==========================================
def generate_radar_strikes(spot_price, expiry_date):
    atm_strike = round(spot_price / STRIKE_STEP) * STRIKE_STEP
    exp_str = format_fyers_date(expiry_date)
    
    strikes = []
    for i in range(-RADAR_RANGE, RADAR_RANGE + 1):
        strike_val = atm_strike + (i * STRIKE_STEP)
        strikes.append(f"NSE:{INDEX_NAME}{exp_str}{strike_val}CE")
        strikes.append(f"NSE:{INDEX_NAME}{exp_str}{strike_val}PE")
    return strikes

# ==========================================
# 🧠 5. PROPRIETARY QUAD-DELTA MATH
# ==========================================
def run_quad_delta_math(symbol_candles):
    """
    Analyzes the candle array up to the CURRENT simulated minute.
    """
    if not symbol_candles or len(symbol_candles) < 2:
        return {"state": "DEAD", "price": 0, "floor": 0}
        
    latest_close = symbol_candles[-1][4]  # Close price
    micro_floor = symbol_candles[-2][3]   # Low of the previous candle (Proxy)
    
    # PROXY ALGO: Replace with your actual Volume/Spread Quad-Delta formula
    is_basket_1_birth = True 
    
    if latest_close < micro_floor:
        return {"state": "BASKET_3", "price": latest_close, "floor": micro_floor}
    elif is_basket_1_birth:
        return {"state": "BASKET_1", "price": latest_close, "floor": micro_floor}
    else:
        return {"state": "TRACKING", "price": latest_close, "floor": micro_floor}

# ==========================================
# ✉️ 6. EMAIL DISPATCHER
# ==========================================
def send_email(subject, body):
    if not EMAIL_SENDER or not EMAIL_APP_PWD:
        print("🔇 No email credentials found. Outputting to console only.")
        print(f"--- {subject} ---\n{body}")
        return
        
    msg = MIMEText(body)
    msg['Subject'] = subject
    msg['From'] = EMAIL_SENDER
    msg['To'] = EMAIL_RECEIVER

    with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp_server:
        smtp_server.login(EMAIL_SENDER, EMAIL_APP_PWD)
        smtp_server.sendmail(EMAIL_SENDER, EMAIL_RECEIVER, msg.as_string())
    print(f"📧 Alert Sent: {subject}")

def format_email_body(added, removed, matrix=None):
    body = ""
    if matrix:
        body += "🌅 MORNING BASELINE (Active Matrix):\n"
        for sym, d in matrix.items():
            body += f"   - {sym} | LTP: {d['price']} | Floor: {d['floor']}\n"
        return body

    body += "⚡ QUAD-DELTA INTRADAY SHIFT:\n\n"
    if added:
        body += "🟢 [+] ADDED TO BASKET 1 (New Momentum):\n"
        for sym, d in added.items():
            body += f"   - {sym} | LTP: {d['price']} | Floor: {d['floor']}\n"
    if removed:
        body += "\n🔴 [-] MOVED TO BASKET 3 (Floor Breached):\n"
        for sym, d in removed.items():
            body += f"   - {sym} | LTP: {d['price']} | Broken Floor: {d['floor']}\n"
    return body

# ==========================================
# ⏱️ 7. THE STATELESS SIMULATOR (CORE ENGINE)
# ==========================================
def fetch_day_history(fyers, symbol):
    """Fetches today's 5-minute history for a given symbol."""
    data = {
        "symbol": symbol,
        "resolution": "5",
        "date_format": "1",
        "range_from": date.today().strftime("%Y-%m-%d"),
        "range_to": date.today().strftime("%Y-%m-%d"),
        "cont_flag": "1"
    }
    res = fyers.history(data=data)
    return res.get('candles', [])

def simulate_day_and_dispatch():
    fyers = get_fyers_instance()
    active_expiry = get_dynamic_expiry()
    
    # 1. Fetch entire day's 5-min Spot Index history
    spot_candles = fetch_day_history(fyers, INDEX_SYMBOL)
    if not spot_candles:
        print("Market data unavailable or market is closed.")
        return

    # Convert GLOBAL_START_TIME to a comparable timestamp
    start_time_obj = datetime.strptime(GLOBAL_START_TIME, "%H:%M").time()
    
    # Filter out candles before the Command Dial start time
    valid_spot_candles = []
    for c in spot_candles:
        candle_dt = datetime.fromtimestamp(c[0])
        if candle_dt.time() >= start_time_obj:
            valid_spot_candles.append(c)
            
    if not valid_spot_candles:
        print(f"⏳ Engine on Standby. Time has not reached {GLOBAL_START_TIME}.")
        return

    # 2. Historical Reconstruction Loop
    # We maintain memory ONLY during this fraction-of-a-second simulation
    anchored_strikes = {}
    previous_matrix = {}
    options_data_cache = {}
    
    for idx, spot_candle in enumerate(valid_spot_candles):
        timestamp = spot_candle[0]
        spot_close = spot_candle[4]
        is_last_candle = (idx == len(valid_spot_candles) - 1)
        
        # Determine the battlefield radar at THIS exact historical minute
        radar_strikes = generate_radar_strikes(spot_close, active_expiry)
        active_targets = set(radar_strikes + list(anchored_strikes.keys()))
        
        current_matrix = {}
        
        for strike in active_targets:
            # Lazy load option history into cache to minimize API calls
            if strike not in options_data_cache:
                options_data_cache[strike] = fetch_day_history(fyers, strike)
            
            # Slice the data so the math doesn't look into the future
            sliced_candles = [c for c in options_data_cache[strike] if c[0] <= timestamp]
            
            result = run_quad_delta_math(sliced_candles)
            
            # State Machine rules
            if result['state'] == "BASKET_1" or (strike in anchored_strikes and result['state'] != "BASKET_3"):
                current_matrix[strike] = result
                anchored_strikes[strike] = True # Pin it
            elif result['state'] == "BASKET_3":
                anchored_strikes.pop(strike, None) # Unpin it

        # 3. Delta Detection on the Final Candle
        if is_last_candle:
            # If this is the very first candle of the allowed day, send the Morning Baseline
            if idx == 0:
                subject = f"🌅 MORNING ANCHOR SET: {len(current_matrix)} Active Trades"
                send_email(subject, format_email_body(None, None, current_matrix))
            else:
                # Compare current vs previous to find what JUST happened
                added = {sym: data for sym, data in current_matrix.items() if sym not in previous_matrix}
                removed = {sym: data for sym, data in previous_matrix.items() if sym not in current_matrix}
                
                if added or removed:
                    subject = f"⚡ INTRADAY DELTA (+{len(added)}/-{len(removed)})"
                    send_email(subject, format_email_body(added, removed))
                else:
                    print("🔇 Tape is quiet on the latest candle. Matrix remains silent.")

        # Move forward in time
        previous_matrix = current_matrix

if __name__ == "__main__":
    print("🟢 Initializing Stateless Quad-Delta Reconstruction...")
    simulate_day_and_dispatch()
