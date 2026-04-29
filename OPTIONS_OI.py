import os
import smtplib
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from email.mime.base import MIMEBase
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email import encoders

try:
    from fyers_apiv3 import fyersModel
except Exception:
    try:
        import fyersModel
    except Exception:
        from fyersapi import fyersModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

DAILY_LOOKBACK_DAYS = 90
INTRADAY_LOOKBACK_DAYS = 20
IVP_LOOKBACK_DAYS = 252
OPTION_PAIRS_TO_KEEP = 5
SECTORS_DIR = os.environ.get("SECTORS_DIR", "sectors")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", ".")

EMAIL_DISPLAY_COLS = [
    "Symbol",
    "LTP",
    "% Change",
    "5m_Signal",
    "15m_Signal",
    "30m_Signal",
    "60m_Signal",
    "Bull_Signal",
    "Bear_Signal",
    "Overall_Signal",
    "Price_Lead_Status",
    "IVP",
    "Volatility State",
    "Last Iteration Time",
]

OPTION_EMAIL_COLS = [
    "Underlying",
    "Option Type",
    "Option Symbol",
    "Strike",
    "LTP",
    "% Change",
    "5m_Signal",
    "15m_Signal",
    "30m_Signal",
    "60m_Signal",
    "Bull_Signal",
    "Bear_Signal",
    "Overall_Signal",
    "Price_Lead_Status",
    "IVP",
    "Volatility State",
    "Last Iteration Time",
]

fyers = None


def init_fyers() -> Optional[object]:
    global fyers
    client_id = os.environ.get("CLIENT_ID") or os.environ.get("CLIENTID")
    access_token = os.environ.get("ACCESS_TOKEN") or os.environ.get("ACCESSTOKEN")
    if not client_id or not access_token:
        logger.error("Missing CLIENT_ID / ACCESS_TOKEN environment variables.")
        fyers = None
        return None
    try:
        fyers = fyersModel.FyersModel(client_id=client_id, is_async=False, token=access_token, log_path="")
    except Exception:
        fyers = fyersModel.FyersModel(client_id=client_id, token=access_token, is_async=False, log_path="")
    return fyers


def safe_float(value, default=np.nan) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except Exception:
        return default


def discover_sector_csvs(root_dir: str = SECTORS_DIR) -> List[str]:
    paths = []
    if not os.path.isdir(root_dir):
        logger.warning("Sectors folder not found: %s", root_dir)
        return []
    for dirpath, _, filenames in os.walk(root_dir):
        for fname in filenames:
            if fname.lower().endswith(".csv"):
                paths.append(os.path.join(dirpath, fname))
    return sorted(set(paths))


def load_fno_symbols_from_sectors(root_dir: str = SECTORS_DIR) -> List[str]:
    symbols = set()
    for path in discover_sector_csvs(root_dir):
        try:
            df = pd.read_csv(path)
        except Exception:
            continue
        lowered = {str(c).strip().lower(): c for c in df.columns}
        symbol_col = None
        for key in ["symbol", "symbols", "ticker", "tradingsymbol"]:
            if key in lowered:
                symbol_col = lowered[key]
                break
        if symbol_col is None:
            continue
        for raw in df[symbol_col].dropna().astype(str):
            sym = raw.strip().upper()
            if sym and sym not in {"NAN", "NONE"}:
                symbols.add(sym)
    return sorted(symbols)


def format_eq_symbol(symbol: str) -> str:
    symbol = str(symbol).strip().upper()
    if symbol.startswith("NSE:"):
        return symbol if symbol.endswith("-EQ") else f"{symbol}-EQ"
    return f"NSE:{symbol}-EQ"


def get_history(symbol: str, resolution: str, days_back: int) -> pd.DataFrame:
    if fyers is None:
        return pd.DataFrame()
    now = datetime.now()
    start = now - timedelta(days=days_back)
    payload = {
        "symbol": symbol,
        "resolution": resolution,
        "date_format": "1",
        "range_from": start.strftime("%Y-%m-%d"),
        "range_to": now.strftime("%Y-%m-%d"),
        "cont_flag": "1",
    }
    try:
        res = fyers.history(data=payload)
    except Exception as exc:
        logger.warning("History fetch failed for %s: %s", symbol, exc)
        return pd.DataFrame()
    candles = (res or {}).get("candles", [])
    if not candles:
        return pd.DataFrame()
    df = pd.DataFrame(candles, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s", utc=True).dt.tz_convert("Asia/Kolkata").dt.tz_localize(None)
    df = df.sort_values("timestamp").drop_duplicates(subset=["timestamp"], keep="last").reset_index(drop=True)
    return df


def compute_ivp(history_df: pd.DataFrame) -> Tuple[float, str]:
    if history_df is None or history_df.empty or len(history_df) < 30:
        return np.nan, "Neutral Vol"
    close = pd.to_numeric(history_df["close"], errors="coerce")
    high = pd.to_numeric(history_df["high"], errors="coerce")
    low = pd.to_numeric(history_df["low"], errors="coerce")
    proxy = ((high - low) / close.replace(0, np.nan) * 100.0).replace([np.inf, -np.inf], np.nan).dropna()
    if proxy.empty:
        return np.nan, "Neutral Vol"
    lookback = proxy.tail(min(IVP_LOOKBACK_DAYS, len(proxy)))
    current = float(lookback.iloc[-1])
    ivp = round((lookback.lt(current).sum() / len(lookback)) * 100, 2)
    if ivp < 30:
        state = "Buyer Zone"
    elif ivp > 50:
        state = "Avoid Buy Premium"
    else:
        state = "Neutral Vol"
    return ivp, state


def score_label(delta: float) -> str:
    if pd.isna(delta):
        return "Neutral"
    if delta >= 7:
        return "Buy++"
    if delta >= 4:
        return "Buy+"
    if delta >= 1:
        return "Buy"
    if delta <= -7:
        return "Sell++"
    if delta <= -4:
        return "Sell+"
    if delta <= -1:
        return "Sell"
    return "Neutral"


def summarize_intraday(intra_df: pd.DataFrame, reference_df: pd.DataFrame) -> Dict[str, object]:
    if intra_df is None or intra_df.empty:
        return {}
    df = intra_df.copy().sort_values("timestamp").reset_index(drop=True)
    close = pd.to_numeric(df["close"], errors="coerce")
    high = pd.to_numeric(df["high"], errors="coerce")
    low = pd.to_numeric(df["low"], errors="coerce")
    volume = pd.to_numeric(df["volume"], errors="coerce").fillna(0.0)

    delta = close.diff().fillna(0.0)
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = gain.rolling(14, min_periods=14).mean()
    avg_loss = loss.rolling(14, min_periods=14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = (100 - (100 / (1 + rs))).fillna(50)

    tr1 = (high - low).abs()
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    atr = pd.Series(tr).rolling(14, min_periods=14).mean()
    plus_di = 100 * pd.Series(plus_dm).rolling(14, min_periods=14).mean() / atr.replace(0, np.nan)
    minus_di = 100 * pd.Series(minus_dm).rolling(14, min_periods=14).mean() / atr.replace(0, np.nan)
    dx = (100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)).fillna(0)
    adx = dx.rolling(14, min_periods=14).mean().fillna(0)

    typical = (high + low + close) / 3.0
    cum_vol = volume.cumsum().replace(0, np.nan)
    vwap = ((typical * volume).cumsum() / cum_vol).ffill().fillna(close)
    vwap_std = (((typical - vwap) ** 2 * volume).cumsum() / cum_vol).pow(0.5).replace(0, np.nan)
    vwap_z = ((close - vwap) / vwap_std).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    range_now = (high - low).clip(lower=0)
    avg_range5 = range_now.rolling(5, min_periods=3).mean()
    avg_vol5 = volume.rolling(5, min_periods=3).mean()
    price_lead_flag = ((range_now / avg_range5.replace(0, np.nan)) >= 1.5) & ((volume / avg_vol5.replace(0, np.nan)) <= 1.0)
    streak = []
    run = 0
    for flag in price_lead_flag.fillna(False).astype(bool):
        run = run + 1 if flag else 0
        streak.append(run)
    streak = pd.Series(streak)
    lead_status = np.select(
        [price_lead_flag & (streak >= 3), price_lead_flag & (streak >= 2), price_lead_flag],
        ["STRONG_PRICE_LEAD_FADE", "PRICE_LEADING_FADE_RISK", "EARLY_PRICE_LEAD"],
        default="NORMAL",
    )

    prev_close = safe_float(reference_df["close"].iloc[-2]) if reference_df is not None and len(reference_df) >= 2 else np.nan
    ltp = safe_float(close.iloc[-1])
    pct_change = ((ltp - prev_close) / prev_close * 100.0) if pd.notna(prev_close) and prev_close != 0 else 0.0

    bull = 0
    bear = 0
    if pct_change > 0:
        bull += 2
    if pct_change < 0:
        bear += 2
    if safe_float(vwap_z.iloc[-1], 0) >= 0.30:
        bull += 2
    if safe_float(vwap_z.iloc[-1], 0) <= -0.30:
        bear += 2
    if safe_float(plus_di.iloc[-1], 0) > safe_float(minus_di.iloc[-1], 0):
        bull += 2
    if safe_float(minus_di.iloc[-1], 0) > safe_float(plus_di.iloc[-1], 0):
        bear += 2
    if safe_float(adx.iloc[-1], 0) >= 20:
        bull += 1
        bear += 1
    if safe_float(rsi.iloc[-1], 50) >= 55:
        bull += 1
    if safe_float(rsi.iloc[-1], 50) <= 45:
        bear += 1

    rank_delta = bull - bear
    ivp, vol_state = compute_ivp(reference_df)
    last_ts = pd.to_datetime(df["timestamp"].iloc[-1])

    return {
        "LTP": round(ltp, 2),
        "% Change": round(pct_change, 2),
        "5m_Signal": score_label(rank_delta),
        "15m_Signal": score_label(rank_delta * 0.9),
        "30m_Signal": score_label(rank_delta * 0.8),
        "60m_Signal": score_label(rank_delta * 0.7),
        "Bull_Signal": score_label(bull),
        "Bear_Signal": score_label(-bear),
        "Overall_Signal": score_label(rank_delta),
        "Price_Lead_Status": str(lead_status[-1]),
        "IVP": ivp,
        "Volatility State": vol_state,
        "Last Iteration Time": last_ts.strftime("%H:%M"),
        "Bull Rank": bull,
        "Bear Rank": bear,
        "Rank Delta": rank_delta,
        "Cumulative +DI": round(safe_float(plus_di.iloc[-1], np.nan), 2),
        "Cumulative -DI": round(safe_float(minus_di.iloc[-1], np.nan), 2),
        "Cumulative ADX": round(safe_float(adx.iloc[-1], np.nan), 2),
        "Cumulative RSI": round(safe_float(rsi.iloc[-1], np.nan), 2),
        "VWAP Z-Score": round(safe_float(vwap_z.iloc[-1], 0.0), 2),
    }


def choose_top_candidates(summary_df: pd.DataFrame, top_n: int = 10) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if summary_df is None or summary_df.empty:
        return pd.DataFrame(columns=EMAIL_DISPLAY_COLS), pd.DataFrame(columns=EMAIL_DISPLAY_COLS)
    base = summary_df.copy()
    long_df = base[
        (pd.to_numeric(base["% Change"], errors="coerce") > 0) &
        (pd.to_numeric(base["Cumulative +DI"], errors="coerce") > pd.to_numeric(base["Cumulative -DI"], errors="coerce"))
    ].copy()
    short_df = base[
        (pd.to_numeric(base["% Change"], errors="coerce") < 0) &
        (pd.to_numeric(base["Cumulative -DI"], errors="coerce") > pd.to_numeric(base["Cumulative +DI"], errors="coerce"))
    ].copy()

    long_df = long_df.sort_values(["Rank Delta", "Cumulative ADX", "% Change"], ascending=[False, False, False]).head(top_n)
    short_df = short_df.sort_values(["Rank Delta", "Cumulative ADX", "% Change"], ascending=[True, False, True]).head(top_n)

    keep_cols = [c for c in EMAIL_DISPLAY_COLS if c in base.columns]
    return long_df[keep_cols].reset_index(drop=True), short_df[keep_cols].reset_index(drop=True)


def nearest_step(value: float) -> int:
    value = abs(safe_float(value, 0))
    if value >= 20000:
        return 100
    if value >= 10000:
        return 50
    if value >= 2000:
        return 20
    if value >= 500:
        return 10
    if value >= 100:
        return 5
    return 1


def option_type_of(item: dict) -> str:
    raw = str(item.get("option_type") or item.get("optionType") or item.get("type") or "").upper()
    if raw in {"CE", "CALL", "CALLS"}:
        return "CE"
    if raw in {"PE", "PUT", "PUTS"}:
        return "PE"
    symbol = str(item.get("symbol") or item.get("fyToken") or item.get("option_symbol") or "").upper()
    if symbol.endswith("-CE") or symbol.endswith("CE"):
        return "CE"
    if symbol.endswith("-PE") or symbol.endswith("PE"):
        return "PE"
    return ""


def option_symbol_of(item: dict) -> str:
    return str(item.get("symbol") or item.get("symbolName") or item.get("option_symbol") or "")


def option_ltp_of(item: dict) -> float:
    for key in ["ltp", "last_traded_price", "lastPrice", "lp"]:
        if key in item:
            return round(safe_float(item.get(key), 0.0), 2)
    return 0.0


def strike_of(item: dict) -> float:
    for key in ["strike_price", "strikePrice", "strike"]:
        if key in item:
            return safe_float(item.get(key), np.nan)
    return np.nan


def fetch_underlying_quote(symbol: str) -> float:
    eq_symbol = format_eq_symbol(symbol)
    try:
        quote = fyers.quotes({"symbols": eq_symbol})
        return safe_float(quote.get("d", [{}])[0].get("v", {}).get("lp"), np.nan)
    except Exception:
        return np.nan


def fetch_option_pairs(symbol: str, pair_count: int = OPTION_PAIRS_TO_KEEP) -> pd.DataFrame:
    if fyers is None:
        return pd.DataFrame(columns=["Underlying", "Underlying LTP", "ATM Strike", "Strike", "CE Symbol", "CE LTP", "PE Symbol", "PE LTP"])

    eq_symbol = format_eq_symbol(symbol)
    ltp = fetch_underlying_quote(symbol)

    try:
        chain_res = fyers.optionchain(data={"symbol": eq_symbol, "strikecount": 50})
    except Exception as exc:
        logger.warning("Option chain failed for %s: %s", symbol, exc)
        return pd.DataFrame(columns=["Underlying", "Underlying LTP", "ATM Strike", "Strike", "CE Symbol", "CE LTP", "PE Symbol", "PE LTP"])

    chain = ((chain_res or {}).get("data") or {}).get("optionsChain", [])
    if not chain:
        return pd.DataFrame(columns=["Underlying", "Underlying LTP", "ATM Strike", "Strike", "CE Symbol", "CE LTP", "PE Symbol", "PE LTP"])

    rows = []
    for item in chain:
        strike = strike_of(item)
        typ = option_type_of(item)
        if pd.isna(strike) or typ not in {"CE", "PE"}:
            continue
        rows.append({
            "Strike": strike,
            "Type": typ,
            "OptionSymbol": option_symbol_of(item),
            "OptionLTP": option_ltp_of(item),
        })
    if not rows:
        return pd.DataFrame(columns=["Underlying", "Underlying LTP", "ATM Strike", "Strike", "CE Symbol", "CE LTP", "PE Symbol", "PE LTP"])

    oc = pd.DataFrame(rows)
    step = nearest_step(ltp if pd.notna(ltp) else oc["Strike"].median())
    atm = round(ltp / step) * step if pd.notna(ltp) else oc["Strike"].median()
    strikes = sorted(oc["Strike"].dropna().unique(), key=lambda x: abs(x - atm))[:pair_count]
    strikes = sorted(strikes)

    final_rows = []
    for strike in strikes:
        sub = oc[oc["Strike"] == strike]
        ce = sub[sub["Type"] == "CE"].head(1)
        pe = sub[sub["Type"] == "PE"].head(1)
        final_rows.append({
            "Underlying": symbol,
            "Underlying LTP": round(safe_float(ltp, np.nan), 2) if pd.notna(ltp) else np.nan,
            "ATM Strike": atm,
            "Strike": strike,
            "CE Symbol": ce["OptionSymbol"].iloc[0] if not ce.empty else "",
            "CE LTP": round(safe_float(ce["OptionLTP"].iloc[0], 0.0), 2) if not ce.empty else 0.0,
            "PE Symbol": pe["OptionSymbol"].iloc[0] if not pe.empty else "",
            "PE LTP": round(safe_float(pe["OptionLTP"].iloc[0], 0.0), 2) if not pe.empty else 0.0,
        })
    return pd.DataFrame(final_rows)


def format_option_history_symbol(raw_symbol: str) -> str:
    symbol = str(raw_symbol).strip()
    return symbol if symbol.startswith("NSE:") else f"NSE:{symbol}"


def scan_single_option(option_symbol: str, option_type: str, strike: float, underlying: str) -> Optional[Dict[str, object]]:
    hist_symbol = format_option_history_symbol(option_symbol)
    daily_df = get_history(hist_symbol, "D", max(DAILY_LOOKBACK_DAYS, IVP_LOOKBACK_DAYS))
    intra_df = get_history(hist_symbol, "5", INTRADAY_LOOKBACK_DAYS)
    if daily_df.empty or intra_df.empty:
        return None
    summary = summarize_intraday(intra_df, daily_df)
    if not summary:
        return None
    return {
        "Underlying": underlying,
        "Option Type": option_type,
        "Option Symbol": option_symbol,
        "Strike": strike,
        "LTP": summary.get("LTP"),
        "% Change": summary.get("% Change"),
        "5m_Signal": summary.get("5m_Signal"),
        "15m_Signal": summary.get("15m_Signal"),
        "30m_Signal": summary.get("30m_Signal"),
        "60m_Signal": summary.get("60m_Signal"),
        "Bull_Signal": summary.get("Bull_Signal"),
        "Bear_Signal": summary.get("Bear_Signal"),
        "Overall_Signal": summary.get("Overall_Signal"),
        "Price_Lead_Status": summary.get("Price_Lead_Status"),
        "IVP": summary.get("IVP"),
        "Volatility State": summary.get("Volatility State"),
        "Last Iteration Time": summary.get("Last Iteration Time"),
        "Bull Rank": summary.get("Bull Rank"),
        "Bear Rank": summary.get("Bear Rank"),
        "Rank Delta": summary.get("Rank Delta"),
        "Cumulative +DI": summary.get("Cumulative +DI"),
        "Cumulative -DI": summary.get("Cumulative -DI"),
        "Cumulative ADX": summary.get("Cumulative ADX"),
        "Cumulative RSI": summary.get("Cumulative RSI"),
        "VWAP Z-Score": summary.get("VWAP Z-Score"),
    }


def build_option_candidates(candidates_df: pd.DataFrame, side: str) -> pd.DataFrame:
    if candidates_df is None or candidates_df.empty or "Symbol" not in candidates_df.columns:
        return pd.DataFrame(columns=OPTION_EMAIL_COLS)

    wanted_type = "CE" if str(side).lower() == "long" else "PE"
    rows = []
    for underlying in candidates_df["Symbol"].dropna().astype(str):
        pair_df = fetch_option_pairs(underlying)
        if pair_df.empty:
            continue
        for _, row in pair_df.iterrows():
            option_symbol = row.get(f"{wanted_type} Symbol", "")
            strike = row.get("Strike", np.nan)
            if not option_symbol:
                continue
            scanned = scan_single_option(option_symbol, wanted_type, strike, underlying)
            if scanned:
                rows.append(scanned)

    if not rows:
        return pd.DataFrame(columns=OPTION_EMAIL_COLS)

    out = pd.DataFrame(rows)
    if wanted_type == "CE":
        out = out[
            (pd.to_numeric(out["Rank Delta"], errors="coerce") > 0) &
            (pd.to_numeric(out["Cumulative +DI"], errors="coerce") > pd.to_numeric(out["Cumulative -DI"], errors="coerce"))
        ].copy()
        out = out.sort_values(["Rank Delta", "Cumulative ADX", "% Change"], ascending=[False, False, False])
    else:
        out = out[
            (pd.to_numeric(out["Rank Delta"], errors="coerce") < 0) &
            (pd.to_numeric(out["Cumulative -DI"], errors="coerce") > pd.to_numeric(out["Cumulative +DI"], errors="coerce"))
        ].copy()
        out = out.sort_values(["Rank Delta", "Cumulative ADX", "% Change"], ascending=[True, False, True])

    keep_cols = [c for c in OPTION_EMAIL_COLS if c in out.columns]
    return out[keep_cols].reset_index(drop=True)


def format_cell(col: str, val) -> str:
    if pd.isna(val):
        return ""
    if col == "% Change":
        return f"{float(val):.2f}%"
    if isinstance(val, (int, float, np.integer, np.floating)):
        return f"{float(val):.2f}"
    return str(val)


def dataframe_to_html(df: pd.DataFrame, columns: List[str], title: str) -> str:
    html = [f"<h3>{title}</h3>"]
    if df is None or df.empty:
        html.append("<p>No data found.</p>")
        return "\n".join(html)
    view = df[[c for c in columns if c in df.columns]].copy()
    html.append("<table border='1' cellspacing='0' cellpadding='6' style='border-collapse:collapse;font-family:Arial,sans-serif;font-size:12px;'>")
    html.append("<tr>" + "".join([f"<th style='background:#f2f2f2'>{c}</th>" for c in view.columns]) + "</tr>")
    for _, row in view.iterrows():
        html.append("<tr>" + "".join([f"<td>{format_cell(c, row[c])}</td>" for c in view.columns]) + "</tr>")
    html.append("</table>")
    return "\n".join(html)


def send_email(long_df: pd.DataFrame, short_df: pd.DataFrame, ce_df: pd.DataFrame, pe_df: pd.DataFrame, attachments: List[str]) -> bool:
    smtp_host = os.environ.get("SMTP_HOST", "smtp.gmail.com")
    smtp_port = int(os.environ.get("SMTP_PORT", "587"))
    sender_email = os.environ.get("SENDER_EMAIL")
    sender_password = os.environ.get("SENDER_PASSWORD")
    recipient_email = os.environ.get("RECIPIENT_EMAIL")

    if not sender_email or not sender_password or not recipient_email:
        logger.warning("Email env vars missing. Skipping email.")
        return False

    scan_time = datetime.now().strftime("%d %b %Y, %H:%M")
    html = f"""
    <html>
    <body>
        <h2>Intraday Vol Iteration Alert</h2>
        <p>Scan completed at {scan_time}.</p>
        {dataframe_to_html(long_df, EMAIL_DISPLAY_COLS, 'Stock Long Candidates')}
        <br>
        {dataframe_to_html(short_df, EMAIL_DISPLAY_COLS, 'Stock Short Candidates')}
        <br>
        {dataframe_to_html(ce_df, OPTION_EMAIL_COLS, 'CE Candidates from Long Stocks')}
        <br>
        {dataframe_to_html(pe_df, OPTION_EMAIL_COLS, 'PE Candidates from Short Stocks')}
        <p>Attached CSVs: summary, CE candidates, PE candidates.</p>
    </body>
    </html>
    """

    msg = MIMEMultipart()
    msg["From"] = sender_email
    msg["To"] = recipient_email
    msg["Subject"] = f"Intraday Vol Iteration Alert - {datetime.now().strftime('%d %b %H:%M')}"
    msg.attach(MIMEText(html, "html", "utf-8"))

    for path in attachments:
        if path and os.path.exists(path):
            with open(path, "rb") as f:
                part = MIMEBase("application", "octet-stream")
                part.set_payload(f.read())
            encoders.encode_base64(part)
            part.add_header("Content-Disposition", f"attachment; filename={os.path.basename(path)}")
            msg.attach(part)

    try:
        if smtp_port == 465:
            with smtplib.SMTP_SSL(smtp_host, smtp_port, timeout=40) as server:
                server.login(sender_email, sender_password)
                server.send_message(msg)
        else:
            with smtplib.SMTP(smtp_host, smtp_port, timeout=40) as server:
                server.ehlo()
                server.starttls()
                server.ehlo()
                server.login(sender_email, sender_password)
                server.send_message(msg)
        logger.info("Email sent to %s", recipient_email)
        return True
    except Exception as exc:
        logger.error("Email failed: %s", exc)
        return False


def scan_symbol(symbol: str) -> Optional[Dict[str, object]]:
    eq = format_eq_symbol(symbol)
    daily_df = get_history(eq, "D", max(DAILY_LOOKBACK_DAYS, IVP_LOOKBACK_DAYS))
    intra_df = get_history(eq, "5", INTRADAY_LOOKBACK_DAYS)
    if daily_df.empty or intra_df.empty:
        return None
    summary = summarize_intraday(intra_df, daily_df)
    if not summary:
        return None
    summary["Symbol"] = symbol
    return summary


def ensure_output_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def main() -> None:
    ensure_output_dir(OUTPUT_DIR)
    init_fyers()
    if fyers is None:
        raise RuntimeError("Fyers client initialization failed.")

    symbols = load_fno_symbols_from_sectors(SECTORS_DIR)
    if not symbols:
        raise FileNotFoundError(f"No F&O symbols found under '{SECTORS_DIR}'.")

    logger.info("Loaded %s symbols from sectors folder.", len(symbols))
    rows = []
    for i, symbol in enumerate(symbols, start=1):
        logger.info("[%s/%s] Scanning %s", i, len(symbols), symbol)
        row = scan_symbol(symbol)
        if row:
            rows.append(row)

    if not rows:
        raise RuntimeError("No symbols returned usable market data.")

    summary_df = pd.DataFrame(rows)
    ordered_cols = ["Symbol"] + [c for c in EMAIL_DISPLAY_COLS if c != "Symbol"] + [
        "Bull Rank", "Bear Rank", "Rank Delta", "Cumulative +DI", "Cumulative -DI", "Cumulative ADX", "Cumulative RSI", "VWAP Z-Score"
    ]
    ordered_cols = [c for c in ordered_cols if c in summary_df.columns]
    summary_df = summary_df[ordered_cols].sort_values(["Rank Delta", "% Change"], ascending=[False, False]).reset_index(drop=True)

    long_df, short_df = choose_top_candidates(summary_df, top_n=10)
    ce_df = build_option_candidates(long_df, side="long")
    pe_df = build_option_candidates(short_df, side="short")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    summary_csv = os.path.join(OUTPUT_DIR, f"fo_summary_{timestamp}.csv")
    ce_csv = os.path.join(OUTPUT_DIR, f"fo_ce_candidates_{timestamp}.csv")
    pe_csv = os.path.join(OUTPUT_DIR, f"fo_pe_candidates_{timestamp}.csv")

    summary_df.to_csv(summary_csv, index=False)
    ce_df.to_csv(ce_csv, index=False)
    pe_df.to_csv(pe_csv, index=False)

    send_email(long_df, short_df, ce_df, pe_df, [summary_csv, ce_csv, pe_csv])

    logger.info("Completed.")
    logger.info("Summary CSV: %s", summary_csv)
    logger.info("CE candidates CSV: %s", ce_csv)
    logger.info("PE candidates CSV: %s", pe_csv)


if __name__ == "__main__":
    main()
