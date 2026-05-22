#!/usr/bin/env python3
"""
FO_FNO_FYERS_VOL_REL_EMAIL_FIXED_RAW.py
F&O Stock Scanner with Raw Directional Signals
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings("ignore")

# ============================================================
# CONFIGURATION
# ============================================================
SYMBOLS = [
    "APOLLOHOSP", "HINDUNILVR", "BRITANNIA", "M&M", "TORNTPHARM"
]

EMAIL_DISPLAY_COLS = [
    "Symbol", "LTP", "Change%",
    "Directional", "Turning", "Stability", "Balanced",
    "5m_Signal", "15m_Signal", "30m_Signal", "60m_Signal",
    "Bull_Signal", "Bear_Signal", "Overall_Signal",
    "Price_Lead_Status", "IVP", "Volatility State", "Last Iteration Time"
]

# ============================================================
# SIGNAL BUILDER (RAW DIRECTIONAL)
# ============================================================
def build_signals_from_raw_directional(detail_df):
    """
    Build timeframe signals from raw Directional values.

    - 5m_Signal  : latest raw Directional
    - 15m_Signal : avg of last 3 Directional values
    - 30m_Signal : avg of last 6 Directional values
    - 60m_Signal : avg of last 12 Directional values
    - Bull_Signal: avg of the four timeframe signals
    - Bear_Signal: max(Directional) - current 5m Directional
    - Overall_Signal: Bull_Signal - Bear_Signal
    """
    vals = detail_df["Directional"].dropna().tolist()

    if not vals:
        return {
            "5m_Signal": 0.0, "15m_Signal": 0.0, "30m_Signal": 0.0, "60m_Signal": 0.0,
            "Bull_Signal": 0.0, "Bear_Signal": 0.0, "Overall_Signal": 0.0
        }

    # Rolling raw averages over N 5-min iterations
    s5  = float(vals[-1])
    s15 = sum(vals[-3:])  / min(len(vals), 3)
    s30 = sum(vals[-6:])  / min(len(vals), 6)
    s60 = sum(vals[-12:]) / min(len(vals), 12)

    # Bull_Signal = net directional average
    bull = (s5 + s15 + s30 + s60) / 4.0

    # Bear_Signal = max Directional - current 5m Directional
    bear = float(detail_df["Directional"].max()) - s5
    if pd.isna(bear):
        bear = 0.0

    # Overall_Signal = Bull_Signal - Bear_Signal
    overall = bull - bear

    return {
        "5m_Signal":      round(s5, 2),
        "15m_Signal":     round(s15, 2),
        "30m_Signal":     round(s30, 2),
        "60m_Signal":     round(s60, 2),
        "Bull_Signal":    round(bull, 2),
        "Bear_Signal":    round(bear, 2),
        "Overall_Signal": round(overall, 2),
    }

# ============================================================
# ITERATION COMPUTATION (Core Statistics)
# ============================================================
def compute_iteration_volume_profile(symbol, ltp, ohlcv_5m_list):
    """
    Compute raw statistical columns for a single symbol.

    ohlcv_5m_list: list of dicts with keys open, high, low, close, volume,
                   each representing one 5-minute candle.
    Returns: (summary_dict, detail_df)
    """
    df = pd.DataFrame(ohlcv_5m_list)
    if df.empty or len(df) < 2:
        return None, None

    # --- Raw directional calculation: slope + net_return ---
    closes = df["close"].values
    opens  = df["open"].values
    highs  = df["high"].values
    lows   = df["low"].values

    # Simple slope over recent closes (last 3 points)
    n = min(3, len(closes))
    x = np.arange(n)
    y = closes[-n:]
    slope = np.polyfit(x, y, 1)[0] if n >= 2 else 0.0

    # Net return from first open to latest close
    net_return = ((closes[-1] - opens[0]) / opens[0] * 100) if opens[0] != 0 else 0.0

    # Directional = raw statistical strength
    directional = slope + net_return

    # Turning = volatility / choppiness proxy
    turning = float(np.std(np.diff(closes))) if len(closes) > 1 else 0.0

    # Stability = trend consistency
    changes = np.diff(closes)
    stability = sum(1 for c in changes if abs(c) > 0) / len(changes) * 100 if len(changes) > 0 else 0.0

    # Balanced = mean reversion distance from mid
    mid = (highs.max() + lows.min()) / 2.0
    balanced = ((closes[-1] - mid) / mid * 100) if mid != 0 else 0.0

    # Price Lead Status
    if closes[-1] >= highs[-3:].max() * 0.995:
        price_lead = "LEAD"
    elif closes[-1] <= lows[-3:].min() * 1.005:
        price_lead = "LAG"
    else:
        price_lead = "NORMAL"

    # IVP (Implied Volume Percentile) placeholder
    ivp = 50.0

    # Volatility State
    vol_state = "Neutral Vol"
    if turning > np.percentile([turning, 1.0], 75):
        vol_state = "High Vol"
    elif turning < np.percentile([turning, 1.0], 25):
        vol_state = "Low Vol"

    # Build detail DataFrame (per-iteration Directional history)
    detail_df = pd.DataFrame({
        "iteration": range(len(closes)),
        "close": closes,
        "Directional": [directional] * len(closes)  # In real script, this would be per-iteration
    })

    # For a proper rolling system, store per-iteration Directional in the caller.
    # Here we pass the detail_df so signal builder can access the rolling window.

    summary = {
        "Symbol": symbol,
        "LTP": round(ltp, 2),
        "Change%": round(((closes[-1] - opens[0]) / opens[0]) * 100, 2),
        "Directional": round(directional, 2),
        "Turning": round(turning, 2),
        "Stability": round(stability, 2),
        "Balanced": round(balanced, 2),
        "Price_Lead_Status": price_lead,
        "IVP": round(ivp, 2),
        "Volatility State": vol_state,
        "Last Iteration Time": datetime.now().strftime("%H:%M"),
    }

    # Build signals from raw directional values in detail_df
    signals = build_signals_from_raw_directional(detail_df)
    summary.update(signals)

    return summary, detail_df

# ============================================================
# EMAIL / DISPLAY HELPERS
# ============================================================
def signal_color(val):
    """Return HTML color style for numeric signal values."""
    try:
        v = float(val)
        if v > 0:
            return "color:#00aa00;"
        elif v < 0:
            return "color:#cc0000;"
        else:
            return "color:#888888;"
    except (ValueError, TypeError):
        return ""

def df_to_html_table(df, display_cols):
    """Convert DataFrame subset to styled HTML table rows."""
    cols = [c for c in display_cols if c in df.columns]
    html = "<table border="1" cellpadding="6" cellspacing="0" style="border-collapse:collapse;font-family:Arial;font-size:12px;">\n"

    # Header
    html += "<tr style="background:#333;color:#fff;">"
    for c in cols:
        html += f"<th>{c}</th>"
    html += "</tr>\n"

    # Rows
    for _, row in df.iterrows():
        html += "<tr>"
        for c in cols:
            v = row.get(c, "")
            style = signal_color(v) if "Signal" in c else ""
            html += f'<td style="{style}">{v}</td>'
        html += "</tr>\n"

    html += "</table>"
    return html

# ============================================================
# MAIN SCANNER LOOP
# ============================================================
def run_scan(symbols=SYMBOLS):
    """
    Main scan function. In production, this pulls live data from Fyers API,
    computes per-iteration statistics, and stores rolling Directional history.
    """
    results = []

    for sym in symbols:
        # ---- PLACEHOLDER: replace with real Fyers API fetch ----
        # Simulated 5-min OHLCV for demonstration
        np.random.seed(hash(sym) % 10000)
        base = 2000 + np.random.randint(-500, 500)
        n_candles = 12
        ohlcv = []
        for i in range(n_candles):
            o = base + np.random.randn() * 5
            c = o + np.random.randn() * 8
            h = max(o, c) + abs(np.random.randn()) * 3
            l = min(o, c) - abs(np.random.randn()) * 3
            ohlcv.append({
                "open": o, "high": h, "low": l, "close": c,
                "volume": int(np.random.randint(1000, 5000))
            })
        ltp = ohlcv[-1]["close"]
        # -------------------------------------------------------

        summary, detail = compute_iteration_volume_profile(sym, ltp, ohlcv)
        if summary:
            results.append(summary)

    df = pd.DataFrame(results)

    # Sort by high Directional, high Stability, low Turning
    if not df.empty:
        df = df.sort_values(
            by=["Directional", "Stability", "Turning"],
            ascending=[False, False, True]
        ).reset_index(drop=True)

    return df

# ============================================================
# EMAIL SENDER (Stub — wire in your SMTP / SendGrid)
# ============================================================
def send_email(html_body, subject="F&O Scanner Raw Signals"):
    """Placeholder: integrate with your email service."""
    print(f"[EMAIL] Subject: {subject}")
    print(f"[EMAIL] Body length: {len(html_body)} chars")
    # TODO: add smtplib / SendGrid logic here

# ============================================================
# ENTRY POINT
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("F&O Scanner — Raw Directional Signals")
    print(f"Run at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    df = run_scan(SYMBOLS)

    if not df.empty:
        # Console preview
        print("\n" + df[EMAIL_DISPLAY_COLS].to_string(index=False))

        # Email HTML
        html = df_to_html_table(df, EMAIL_DISPLAY_COLS)
        send_email(html)
    else:
        print("No results generated.")
