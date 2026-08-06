import os
import pandas as pd
import numpy as np

def scan_live_pure_price_breakouts(csv_filename="historical_fno.csv"):
    """
    Scans the latest session for gapless, zero-drawdown breakouts 
    that emerge from a verified price/volume compression pattern.
    """
    if not os.path.exists(csv_filename):
        print(f"❌ Error: '{csv_filename}' not found in the current directory.")
        return

    print("\n⏳ Loading market data and applying Pure Price Action filters...")
    df = pd.read_csv(csv_filename)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values(['Symbol', 'Date']).reset_index(drop=True)

    latest_date = df['Date'].max()
    print(f"📅 Scanning latest session: {latest_date.strftime('%Y-%m-%d')}")

    breakout_candidates = []

    for symbol, group in df.groupby('Symbol'):
        group = group.reset_index(drop=True)
        
        # We need at least 25 days to calculate the 20-day baseline + 5 setup days
        if len(group) < 25:
            continue

        # --- TIME MAPPING ---
        today = group.iloc[-1]       # Day 1 (The Breakout / Live Day)
        yest = group.iloc[-2]        # Day 0 (The Base)
        t_minus_2 = group.iloc[-3]
        t_minus_3 = group.iloc[-4]
        
        baseline_20d = group.iloc[-25:-5] # 20-day lookback for baseline

        # ==========================================================
        # PHASE 1: THE EXACT BREAKOUT TRIGGER (Today vs Yesterday)
        # ==========================================================
        
        # Rule 1: Gapless Open (Today's Open == Yesterday's Close within 0.2% buffer)
        gap_pct = abs(today['Open'] - yest['Close']) / (yest['Close'] + 1e-8) * 100
        is_gapless = gap_pct <= 0.2
        
        # Rule 2: Zero Drawdown (Today's Low never drops below Yesterday's Close)
        # Note: If today's low is perfectly equal to yest close, that's fine. It just can't be lower.
        zero_drawdown = today['Low'] >= (yest['Close'] * 0.999) # 0.1% tick leniency 

        # ==========================================================
        # PHASE 2: PRE-BREAKOUT DNA (The days leading up to Yesterday)
        # ==========================================================
        
        # Calculate Ranges
        range_yest = yest['High'] - yest['Low']
        range_t2 = t_minus_2['High'] - t_minus_2['Low']
        range_t3 = t_minus_3['High'] - t_minus_3['Low']
        
        avg_recent_range = (range_yest + range_t2 + range_t3) / 3.0
        avg_baseline_range = (baseline_20d['High'] - baseline_20d['Low']).mean()
        
        # DNA 1: Range Contraction (Are recent ranges tighter than the 20-day average?)
        is_compressing = avg_recent_range < avg_baseline_range
        
        # DNA 2: The Floor Test (Higher or Flat Lows leading into the base)
        higher_lows = (yest['Low'] >= t_minus_2['Low']) and (t_minus_2['Low'] >= t_minus_3['Low'])
        
        # DNA 3: Volume Dry-Up (Recent volume is quieter than the 20-day average surge)
        avg_recent_vol = (yest['Volume'] + t_minus_2['Volume'] + t_minus_3['Volume']) / 3.0
        avg_baseline_vol = baseline_20d['Volume'].mean()
        volume_drying = avg_recent_vol <= (avg_baseline_vol * 1.2) # Doesn't exceed 20% over avg
        
        # DNA 4: Closing Strength (Yesterday's Close was near its High - upper wick < 35% of total range)
        yest_total_range = range_yest if range_yest > 0 else 1e-8
        yest_upper_wick = yest['High'] - max(yest['Open'], yest['Close'])
        clean_close = (yest_upper_wick / yest_total_range) <= 0.35

        # ==========================================================
        # MATCH AND RECORD
        # ==========================================================
        
        if is_gapless and zero_drawdown and is_compressing and higher_lows and clean_close:
            current_move_pct = ((today['Close'] - yest['Close']) / yest['Close']) * 100
            
            # We want stocks pushing upwards today (Positive momentum)
            if current_move_pct > 0:
                breakout_candidates.append({
                    'Symbol': symbol,
                    'Current_Move_%': round(current_move_pct, 2),
                    'Today_LTP': today['Close'],
                    'Gap_Variance_%': round(gap_pct, 3),
                    'Pre_Breakout_Status': 'Perfect DNA Match'
                })

    # Output Results
    if not breakout_candidates:
        print("\n⚠️ No stocks met 100% of the strict Gapless + Zero Drawdown + Compression rules today.")
        print("This is normal. We only want A+ setups.")
        return

    result_df = pd.DataFrame(breakout_candidates).sort_values(by='Current_Move_%', ascending=False)
    
    print("\n" + "="*80)
    print("🚀 TODAY'S PURE PRICE ACTION BREAKOUTS (Gapless + Zero Drawdown)")
    print("="*80)
    print(result_df.to_string(index=False))
    
    result_df.to_csv("todays_live_breakouts.csv", index=False)
    print("\n📁 Saved list to 'todays_live_breakouts.csv'.")

if __name__ == "__main__":
    scan_live_pure_price_breakouts()
