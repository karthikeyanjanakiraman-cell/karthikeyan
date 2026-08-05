import os
import pandas as pd
import numpy as np

def extract_exact_pre_breakout_dna(csv_filename="historical_fno.csv", catalog_filename="pristine_long_breakouts_catalog.csv"):
    """
    Takes the pristine breakout catalog and looks backward (T-3, T-2, T-1) 
    to quantify the exact pre-breakout compression pattern.
    """
    if not os.path.exists(csv_filename) or not os.path.exists(catalog_filename):
        print(f"❌ Error: Missing files. Ensure '{csv_filename}' and '{catalog_filename}' exist.")
        return

    print(f"\n⏳ Analyzing pre-breakout DNA across your verified catalog...")
    df = pd.read_csv(csv_filename)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values(['Symbol', 'Date']).reset_index(drop=True)

    catalog = pd.read_csv(catalog_filename)
    if catalog.empty:
        print("⚠️ Catalog is empty.")
        return

    symbol_groups = {symbol: group.reset_index(drop=True) for symbol, group in df.groupby('Symbol')}
    dna_records = []

    for _, row in catalog.iterrows():
        symbol = row['Symbol']
        breakout_date = pd.to_datetime(row['Breakout_Date'])

        if symbol not in symbol_groups:
            continue

        group = symbol_groups[symbol]
        match_idx = group.index[group['Date'] == breakout_date]
        
        if match_idx.empty:
            continue

        idx = match_idx[0]
        
        # In the catalog, Breakout_Date is T+2 (the final day of the 2-day breakout).
        # idx is T+2. 
        # T+1 is idx - 1
        # T0 (Base Close) is idx - 2
        # T-1 is idx - 3
        # T-2 is idx - 4
        # T-3 is idx - 5
        
        if idx < 5:
            continue  # Not enough history before the breakout

        # Extract pre-breakout candles: T-3, T-2, T-1
        t_minus_3 = group.loc[idx - 5]
        t_minus_2 = group.loc[idx - 4]
        t_minus_1 = group.loc[idx - 3]

        # Calculate Ranges (High - Low)
        range_t3 = t_minus_3['High'] - t_minus_3['Low']
        range_t2 = t_minus_2['High'] - t_minus_2['Low']
        range_t1 = t_minus_1['High'] - t_minus_1['Low']

        # Pattern 1: Volatility Compression (Did ranges shrink into T-1?)
        is_compressing = (range_t1 < range_t2) or (range_t2 < range_t3)

        # Pattern 2: Higher-Low Anchor (Did lows stay flat or rise consecutively?)
        higher_lows = (t_minus_2['Low'] >= t_minus_3['Low']) and (t_minus_1['Low'] >= t_minus_2['Low'])

        # Pattern 3: Close near High on T-1 (Absence of upper wick)
        t1_total_range = range_t1 if range_t1 > 0 else 1e-8
        t1_upper_wick = t_minus_1['High'] - max(t_minus_1['Open'], t_minus_1['Close'])
        upper_wick_pct = (t1_upper_wick / t1_total_range) * 100
        clean_close = upper_wick_pct <= 25.0  // Upper wick takes less than 25% of range

        dna_records.append({
            'Symbol': symbol,
            'Breakout_Date': breakout_date.strftime('%Y-%m-%d'),
            'Range_T3': round(range_t3, 2),
            'Range_T2': round(range_t2, 2),
            'Range_T1': round(range_t1, 2),
            'Is_Compressing': int(is_compressing),
            'Higher_Lows': int(higher_lows),
            'Clean_Close_T1': int(clean_close),
            'Volume_T1': t_minus_1['Volume']
        })

    dna_df = pd.DataFrame(dna_records)
    if not dna_df.empty:
        print(f"\n🎉 Successfully mapped pre-breakout DNA for {len(dna_df)} instances!")
        
        # Calculate strict structural percentages
        compression_rate = dna_df['Is_Compressing'].mean() * 100
        higher_lows_rate = dna_df['Higher_Lows'].mean() * 100
        clean_close_rate = dna_df['Clean_Close_T1'].mean() * 100

        print("\n" + "="*80)
        print("🧬 PRE-BREAKOUT PATTERN STATISTICAL VERIFICATION (T-3 to T-1)")
        print("="*80)
        print(f"1. Volatility Compression Rate: {compression_rate:.1f}% of setups show shrinking ranges")
        print(f"2. Higher-Low Anchor Rate:      {higher_lows_rate:.1f}% of setups hold or raise lows")
        print(f"3. Clean Close Rate (T-1):      {clean_close_rate:.1f}% close near highs with small wicks")
        print("-" * 80)
        print("\nSample Extracted DNA Records:")
        print(dna_df.head(15).to_string(index=False))

        dna_df.to_csv("pre_breakout_dna_verified.csv", index=False)
        print("\n📁 Saved complete pre-breakout profile to 'pre_breakout_dna_verified.csv'.")
    else:
        print("⚠️ No DNA records generated.")

if __name__ == "__main__":
    extract_exact_pre_breakout_dna()
