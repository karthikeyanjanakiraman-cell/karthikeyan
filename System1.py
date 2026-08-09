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
LOOKBACK_DAYS = 3       # How many days back to reconstruct the Baskets
TOP_N_STRIKES = 5       # Max apex trades to display per Basket

ACTIVE_INDICES = {
    "NIFTY": {"symbol": "NSE:NIFTY50-INDEX", "step": 50},
    "BANKNIFTY": {"symbol": "NSE:NIFTYBANK-INDEX", "step": 100},
    "FINNIFTY": {"symbol": "NSE:NIFTY FIN SERVICE-INDEX", "step": 50},
    "SENSEX": {"symbol": "BSE:SENSEX-INDEX", "step": 100}
}

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

def get_fyers_instance():
    return fyersModel.FyersModel(client_id=CLIENT_ID, is_async=False, token=ACCESS_TOKEN, log_path="")

# ==========================================
# 🛡️ 3. TIMEZONE LOCK & EPOCH GENERATOR
# ==========================================
IST = timezone(timedelta(hours=5, minutes=30))

def get_target_dates():
    """Calculates the end date (Today) and the start date (Lookback) avoiding weekends."""
    ist_now = datetime.now(IST)
    end_date = ist_now.date()
    
    # Weekend override for the End Date
    if end_date.weekday() == 5: end_date -= timedelta(days=1)
    elif end_date.weekday() == 6: end_date -= timedelta(days=2)
        
    # Calculate Epoch Start Date (Skipping weekends)
    start_date = end_date
    days_subtracted = 0
    while days_subtracted < LOOKBACK_DAYS:
        start_date -= timedelta(days=1)
        if start_date.weekday() < 5: # Monday to Friday
            days_subtracted += 1
            
    return start_date, end_date

EPOCH_START, TARGET_END = get_target_dates()

# ==========================================
# 📜 4. THE MASTER FILE LOADER
# ==========================================
MASTER_SYMBOLS = {}
INDEX_EXPIRIES = {"NIFTY": set(), "BANKNIFTY": set(), "FINNIFTY": set(), "SENSEX": set()}

def load_symbol_master():
    """Downloads Fyers global CSVs for 100% accurate symbol generation."""
    print("📡 Downloading Exchange Symbol Master...")
    urls = ["https://public.fyers.in/sym_details/NSE_FO.csv", "https://public.fyers.in/sym_details/BSE_FO.csv"]
    
    for url in urls:
        try:
            res = requests.get(url, timeout=15)
            for line in res.text.split('\n'):
                parts = line.split(',')
                if len(parts) < 17: continue
                sym_ticker, opt_type, strike_str, expiry_val = parts[9], parts[16], parts[15], parts[8]
                
                if opt_type not in ["CE", "PE"]: continue
                    
                idx = next((i for i in ACTIVE_INDICES if sym_ticker.startswith(ACTIVE_INDICES[i]["symbol"].split("-")[0])), None)
                if not idx: continue
                
                try:
                    if expiry_val.isdigit():
                        expiry_date = datetime.fromtimestamp(int(expiry_val), tz=timezone.utc).astimezone(IST).date()
                    else:
                        expiry_date = datetime.strptime(expiry_val, "%Y-%m-%d").date()
                    
                    INDEX_EXPIRIES[idx].add(expiry_date)
                    MASTER_SYMBOLS[(idx, expiry_date, int(float(strike_str)), opt_type)] = sym_ticker
                except:
                    continue
        except Exception as e:
            print(f"⚠️ Master file download failed: {e}")

    for k in INDEX_EXPIRIES:
        INDEX_EXPIRIES[k] = sorted(list(INDEX_EXPIRIES[k]))

# ==========================================
# 🎯 5. THE LIQUIDITY FILTER (API SHIELD)
# ==========================================
def get_liquid_strikes(fyers, index_name, expiry_date):
    """Gathers all chain strikes, checks live quotes, and purges illiquid junk."""
    # 1. Grab all possible symbols for this index & expiry
    all_symbols = [sym for (idx, exp, strike, opt), sym in MASTER_SYMBOLS.items() if idx == index_name and exp == expiry_date]
    
    liquid_symbols = []
    # 2. Batch API calls to 'quotes' (Max 50 per call)
    for i in range(0, len(all_symbols), 50):
        batch = all_symbols[i:i+50]
        try:
            res = fyers.quotes({"symbols": ",".join(batch)})
            if 'd' in res:
                for data in res['d']:
                    sym = data['n']
                    lp = data['v'].get('lp', 0)
                    vol = data['v'].get('volume', 0)
                    
                    # 3. THE GUILLOTINE: Discard dead strikes
                    if lp > 15.0 and vol > 0:
                        liquid_symbols.append(sym)
        except Exception as e:
            pass
            
    print(f"   🛡️ Liquidity Filter: Purged dead strikes. Tracking {len(liquid_symbols)} high-probability targets.")
    return liquid_symbols

# ==========================================
# 🧠 6. PROPRIETARY QUAD-DELTA MATH (MOCKED)
# ==========================================
def run_quad_delta_math(symbol, symbol_candles):
    """
    Analyzes historical array.
    Returns structured data for the UI scoreboard.
    """
    if not symbol_candles or len(symbol_candles) < 2:
        return None
        
    latest_close = symbol_candles[-1][4]
    
    # -----------------------------------------------------
    # 🧠 PROPRIETARY MATH INJECTION POINT
    # Replace this dummy logic with your actual V, P, M, E formulas
    # -----------------------------------------------------
    # MOCK DATA FOR THE UI:
    is_basket_1 = True
    is_basket_2 = False # Toggle this in your math to route to Reloads
    
    base_points = 150
    sentiment_mult = 1 if symbol.endswith("CE") else -1
    
    if is_basket_1:
        return {
            "state": "BASKET_1",
            "points": (base_points + 17) * sentiment_mult,
            "v": 26 * sentiment_mult, "p": 22 * sentiment_mult, "m": 57 * sentiment_mult, "e": 62 * sentiment_mult,
            "launchpad": latest_close * 0.95,
            "birth_time": "2026-08-07 @ 14:30",
            "micro_floor": latest_close * 0.90,
            "price": latest_close
        }
    elif is_basket_2:
        return {
            "state": "BASKET_2",
            "points": (base_points + 118) * sentiment_mult,
            "v": 23 * sentiment_mult, "p": 54 * sentiment_mult, "m": 96 * sentiment_mult, "e": 96 * sentiment_mult,
            "macro_floor": latest_close * 0.80,
            "macro_time": "2026-07-13 @ 11:45",
            "micro_floor": latest_close * 0.95,
            "micro_time": "2026-08-07 @ 09:45",
            "price": latest_close,
            "drift": 5.44
        }
    return None

# ==========================================
# ⏱️ 7. THE ENGINE CORE
# ==========================================
def simulate_engine():
    fyers = get_fyers_instance()
    load_symbol_master()
    
    print(f"\n🟢 Initiating Deep Matrix Scan (Epoch: {EPOCH_START} to {TARGET_END})...")
    master_basket_1 = {}
    master_basket_2 = {}

    for index_name, config in ACTIVE_INDICES.items():
        print(f"\n🔍 Scanning {index_name} Ecosystem...")
        
        valid_dates = [d for d in INDEX_EXPIRIES.get(index_name, []) if d >= TARGET_END]
        if not valid_dates:
            continue
            
        active_expiry = valid_dates[1] if TARGET_END == valid_dates[0] and len(valid_dates) > 1 else valid_dates[0]
        
        # 1. Engage API Shield
        liquid_targets = get_liquid_strikes(fyers, index_name, active_expiry)
        
        b1_trades = []
        b2_trades = []
        
        # 2. Evaluate Surviving Targets
        for sym in liquid_targets:
            res = fyers.history({
                "symbol": sym, "resolution": "5", "date_format": "1",
                "range_from": EPOCH_START.strftime("%Y-%m-%d"), 
                "range_to": TARGET_END.strftime("%Y-%m-%d"), "cont_flag": "1"
            })
            candles = res.get('candles', [])
            
            math_result = run_quad_delta_math(sym, candles)
            if math_result:
                if math_result["state"] == "BASKET_1":
                    b1_trades.append((sym, math_result))
                elif math_result["state"] == "BASKET_2":
                    b2_trades.append((sym, math_result))

        # 3. Sort & Truncate (The Guillotine)
        # Sort by absolute points (highest momentum)
        b1_trades.sort(key=lambda x: abs(x[1]['points']), reverse=True)
        b2_trades.sort(key=lambda x: abs(x[1]['points']), reverse=True)
        
        master_basket_1[index_name] = b1_trades[:TOP_N_STRIKES]
        master_basket_2[index_name] = b2_trades[:TOP_N_STRIKES]

    return master_basket_1, master_basket_2

# ==========================================
# ✉️ 8. INSTITUTIONAL TERMINAL UI
# ==========================================
def render_terminal(b1_master, b2_master):
    print("\n" + "="*80)
    print(" 🦅 INSTITUTIONAL QUAD-DELTA SCOREBOARD ".center(80, "="))
    print(f" [ EPOCH WINDOW: {EPOCH_START} to {TARGET_END} ] ".center(80, " "))
    print("="*80)

    for index_name in ACTIVE_INDICES.keys():
        b1_trades = b1_master.get(index_name, [])
        b2_trades = b2_master.get(index_name, [])
        
        if not b1_trades and not b2_trades:
            continue
            
        print(f"\n🌐 {index_name} APEX TARGETS")
        print("-" * 80)
        
        # --- BASKET 1 RENDER ---
        if b1_trades:
            print("🔥 BASKET 1: FRESH INTRUSIONS (Phase 1 - Day-1 Births)")
            for sym, d in b1_trades:
                sent = "BULLISH" if sym.endswith("CE") else "BEARISH"
                icon = "🚨" if sent == "BULLISH" else "⚠️" # Differentiator
                sign = "+" if d['points'] > 0 else ""
                
                print(f"  {icon} {sym:<20} {sign}{d['points']} pts [V:{d['v']:+d} P:{d['p']:+d} M:{d['m']:+d} E:{d['e']:+d}] ({sent})")
                print(f"      └─ 🧱 Launchpad (Kinetic Base) : Price: ₹{d.get('launchpad', 0):.2f}")
                print(f"      └─ ⚓ Breakout Anchor (Birth)  : {d.get('birth_time', '')} | Price: ₹{d.get('micro_floor', 0):.2f}")
                print(f"      └─ 🎯 Latest LTP               : {TARGET_END} @ EOD   | Price: ₹{d['price']:.2f}\n")

        # --- BASKET 2 RENDER ---
        if b2_trades:
            print("🔄 BASKET 2: ALGORITHMIC RELOADS (Phase 2 - Institutional Continuations)")
            for sym, d in b2_trades:
                sent = "BULLISH" if sym.endswith("CE") else "BEARISH"
                sign = "+" if d['points'] > 0 else ""
                
                print(f"  🔄 {sym:<20} {sign}{d['points']} pts [V:{d['v']:+d} P:{d['p']:+d} M:{d['m']:+d} E:{d['e']:+d}] ({sent})")
                print(f"      └─ ⚓ Macro Floor (Origin) : {d.get('macro_time', '')} | Price: ₹{d.get('macro_floor', 0):.2f}")
                print(f"      └─ ⚡ Micro Floor (Reload) : {d.get('micro_time', '')} | Price: ₹{d.get('micro_floor', 0):.2f}")
                print(f"      └─ 🎯 Latest LTP           : {TARGET_END} @ EOD   | Price: ₹{d['price']:.2f} (Trend Drift: {d.get('drift', 0):+.2f}%)\n")

if __name__ == "__main__":
    b1, b2 = simulate_engine()
    render_terminal(b1, b2)
    print("✅ System Core Shutting Down.")
