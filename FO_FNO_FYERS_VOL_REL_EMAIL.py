import os
import sys
import math
import logging
import configparser
from datetime import datetime, timedelta
from typing import List, Dict, Optional

import numpy as np
import pandas as pd
from fyersapiv3 import fyersModel
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders


class UTF8Formatter(logging.Formatter):
    def format(self, record):
        msg = record.getMessage()
        for x in ["✅", "❌", "🟢", "🟡", "🔴", "⚠️", "📊", "🎯"]:
            msg = msg.replace(x, "")
        record.msg = msg
        return super().format(record)


logformat = "%(asctime)s - %(levelname)s - %(message)s"
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.handlers.clear()

consolehandler = logging.StreamHandler(sys.stdout)
consolehandler.setFormatter(UTF8Formatter(logformat))
logger.addHandler(consolehandler)

filehandler = logging.FileHandler("fo_fyers_momentum_email.log", encoding="utf-8")
filehandler.setFormatter(UTF8Formatter(logformat))
logger.addHandler(filehandler)

logger.info("FO Fyers Momentum Email Scanner initialized")

config = configparser.ConfigParser()
config.read("config.ini")


def getcfg(section, key, envname=None, default=None, isint=False):
    if envname:
        val = os.getenv(envname)
        if val is not None and val.strip() != "":
            return int(val) if isint else val
    if section and key and config.has_option(section, key):
        val = config.get(section, key)
        return int(val) if isint else val
    return default


try:
    clientid = getcfg("fyers", "clientid", envname="CLIENTID")
    token = getcfg("fyers", "accesstoken", envname="ACCESSTOKEN") or getcfg("fyers", "token", envname="TOKEN")
    if not clientid or not token:
        raise ValueError("Missing CLIENTID or ACCESSTOKEN")
    fyers = fyersModel.FyersModel(client_id=clientid, token=token)
    logger.info("Fyers API connected successfully")
except Exception as e:
    logger.error(f"Fyers init failed: {e}")
    fyers = None


def load_fno_symbols_from_sectors(rootdir: str = "sectors") -> List[str]:
    symbols = set()
    if not os.path.isdir(rootdir):
        logger.warning(f"FNO sectors folder {rootdir} not found")
        return []
    for dirpath, _, filenames in os.walk(rootdir):
        for fname in filenames:
            if not fname.lower().endswith(".csv"):
                continue
            fpath = os.path.join(dirpath, fname)
            try:
                df = pd.read_csv(fpath)
                col = None
                for c in df.columns:
                    if c.lower() in ["symbol", "symbols", "ticker"]:
                        col = c
                        break
                if col is None:
                    continue
                for s in df[col].dropna().astype(str):
                    s = s.strip().upper()
                    if s:
                        symbols.add(s)
            except Exception as e:
                logger.warning(f"Error reading {fpath}: {e}")
    out = sorted(symbols)
    logger.info(f"Loaded {len(out)} unique FO symbols")
    return out


def format_fyers_symbol(sym: str) -> str:
    s = sym.strip().upper()
    if s.startswith("NSE:") and ("-EQ" in s or "-INDEX" in s):
        return s
    s = s.replace("NSE:", "").replace("-EQ", "")
    if s in ["NIFTY", "NIFTY50"]:
        return "NSE:NIFTY50-INDEX"
    if s == "BANKNIFTY":
        return "NSE:NIFTYBANK-INDEX"
    if s == "FINNIFTY":
        return "NSE:FINNIFTY-INDEX"
    if s == "MIDCPNIFTY":
        return "NSE:MIDCPNIFTY-INDEX"
    return f"NSE:{s}-EQ"


def get_fyers_history(symbol: str, resolution: str, daysback: int) -> Optional[pd.DataFrame]:
    if fyers is None or daysback <= 0:
        return None

    per_call_limit = 366 if resolution == "D" else 100
    all_chunks = []
    enddate = datetime.now()
    remaining = daysback

    while remaining > 0:
        chunkdays = min(per_call_limit, remaining)
        startdate = enddate - timedelta(days=chunkdays - 1)

        data = {
            "symbol": symbol,
            "resolution": resolution,
            "date_format": 1,
            "range_from": startdate.strftime("%Y-%m-%d"),
            "range_to": enddate.strftime("%Y-%m-%d"),
            "cont_flag": 1,
        }

        try:
            resp = fyers.history(data=data)
            if resp.get("s") != "ok" or "candles" not in resp:
                break
            candles = resp["candles"]
            if not candles:
                break

            dfchunk = pd.DataFrame(
                candles,
                columns=["timestamp", "open", "high", "low", "close", "volume"]
            )
            dfchunk["timestamp"] = pd.to_datetime(dfchunk["timestamp"], unit="s")
            all_chunks.append(dfchunk)
        except Exception:
            break

        remaining -= chunkdays
        enddate = startdate - timedelta(days=1)

    if not all_chunks:
        return None

    df = pd.concat(all_chunks, ignore_index=True)
    df = df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    return df


def compute_annualized_volatility_from_daily(df: pd.DataFrame) -> Optional[float]:
    if df is None or df.empty or len(df) < 10:
        return None
    closes = df["close"].astype(float)
    returns = np.log(closes / closes.shift(1)).dropna()
    if len(returns) < 5:
        return None
    vol = returns.std() * math.sqrt(252) * 100.0
    return float(vol)


def compute_volatility_pair(dfdaily: pd.DataFrame, shortdays: int = 20, longdays: int = 252) -> Dict[str, Optional[float]]:
    if dfdaily is None or dfdaily.empty:
        return {"DailyCurrVolPct": None, "DailyAvgVolPct": None, "VolExpansion": None}

    dfsorted = dfdaily.sort_values("timestamp").reset_index(drop=True)
    currvolann = compute_annualized_volatility_from_daily(dfsorted.tail(shortdays))
    avgvolann = compute_annualized_volatility_from_daily(dfsorted.tail(longdays))

    sqrt252 = math.sqrt(252)
    dailycurr = currvolann / sqrt252 if currvolann else None
    dailyavg = avgvolann / sqrt252 if avgvolann else None
    volexpansion = dailycurr / dailyavg if dailycurr and dailyavg and dailyavg != 0 else None

    return {
        "DailyCurrVolPct": dailycurr,
        "DailyAvgVolPct": dailyavg,
        "VolExpansion": volexpansion
    }


def compute_cumulative_relative_volume(df: pd.DataFrame, lookbackshort: int = 10, lookbacklong: int = 20) -> Dict[str, Optional[float]]:
    if df is None or df.empty or len(df) < 10:
        return {"RelVolume10": None, "RelVolume20": None, "CurrentVolume": None, "LTP": None}

    s = df.copy()
    s["datetime"] = pd.to_datetime(s["timestamp"])
    s["date"] = s["datetime"].dt.date
    s["time"] = s["datetime"].dt.time

    latest = s.iloc[-1]
    latestdate = latest["date"]
    latesttime = latest["time"]
    ltp = float(latest["close"])

    todaymask = (s["date"] == latestdate) & (s["time"] <= latesttime)
    currentcumvol = float(s.loc[todaymask, "volume"].sum())

    def avgcumvol(lastndays: int) -> Optional[float]:
        startdate = latestdate - timedelta(days=lastndays * 2 + 5)
        maskprior = (s["date"] < latestdate) & (s["date"] >= startdate) & (s["time"] <= latesttime)
        priordata = s.loc[maskprior]
        if priordata.empty:
            return None
        dailysums = priordata.groupby("date")["volume"].sum().tail(lastndays)
        if dailysums.empty:
            return None
        return float(dailysums.mean())

    avg10 = avgcumvol(lookbackshort)
    avg20 = avgcumvol(lookbacklong)

    rel10 = currentcumvol / avg10 if avg10 and avg10 != 0 else None
    rel20 = currentcumvol / avg20 if avg20 and avg20 != 0 else None

    return {
        "RelVolume10": rel10,
        "RelVolume20": rel20,
        "CurrentVolume": currentcumvol,
        "LTP": ltp
    }


def scan_fno_universe() -> pd.DataFrame:
    symbols = load_fno_symbols_from_sectors("sectors")
    if not symbols:
        logger.error("No FO symbols found")
        return pd.DataFrame()

    rows = []
    total = len(symbols)

    for idx, sym in enumerate(symbols, start=1):
        logger.info(f"{idx}/{total} Processing {sym}")
        fyerssym = format_fyers_symbol(sym)

        dailydf = get_fyers_history(fyerssym, resolution="D", daysback=365)
        volinfo = compute_volatility_pair(dailydf, shortdays=20, longdays=252)

        prevclose = float(dailydf["close"].iloc[-2]) if dailydf is not None and len(dailydf) >= 2 else None

        intradf = get_fyers_history(fyerssym, resolution="5", daysback=40)
        if intradf is not None:
            rvolinfo = compute_cumulative_relative_volume(intradf, lookbackshort=10, lookbacklong=20)
        else:
            rvolinfo = {
                "RelVolume10": None,
                "RelVolume20": None,
                "CurrentVolume": None,
                "LTP": float(dailydf["close"].iloc[-1]) if dailydf is not None and len(dailydf) else None
            }

        pctchange = ((rvolinfo["LTP"] - prevclose) / prevclose * 100) if prevclose and rvolinfo["LTP"] else 0.0

        daily_volume_expansion = max(
            [x for x in [rvolinfo.get("RelVolume10"), rvolinfo.get("RelVolume20")] if x is not None],
            default=None
        )

        momentum_score = (
            pctchange * volinfo["VolExpansion"] * rvolinfo["RelVolume10"]
            if volinfo.get("VolExpansion") and rvolinfo.get("RelVolume10") and pctchange > 0
            else None
        )

        short_momentum_score = (
            (-pctchange) * volinfo["VolExpansion"] * rvolinfo["RelVolume10"]
            if volinfo.get("VolExpansion") and rvolinfo.get("RelVolume10") and pctchange < 0
            else None
        )

        trade_side = (
            "LONG" if momentum_score and momentum_score > 0
            else "SHORT" if short_momentum_score and short_momentum_score > 0
            else "NONE"
        )

        rows.append({
            "Symbol": sym,
            "LTP": rvolinfo["LTP"],
            "% Change": pctchange,
            "Current Daily Volatility": volinfo["DailyCurrVolPct"],
            "Avg Daily Volatility": volinfo["DailyAvgVolPct"],
            "Daily Volatility Expansion": volinfo["VolExpansion"],
            "Current Volume": rvolinfo["CurrentVolume"],
            "Today Volume / 10 Day Relative Volume": rvolinfo["RelVolume10"],
            "Today Volume / 20 Day Relative Volume": rvolinfo["RelVolume20"],
            "Daily Volume Expansion": daily_volume_expansion,
            "Momentum Score": momentum_score,
            "Short Momentum Score": short_momentum_score,
            "Trade Side": trade_side,
        })

    return pd.DataFrame(rows)


def format_display_table(df: pd.DataFrame, kind: str, maxrows: int = 15) -> pd.DataFrame:
    if df is None or df.empty:
        if kind == "long":
            return pd.DataFrame(columns=[
                "Symbol", "LTP", "% Change",
                "Daily Volatility Expansion",
                "Today Volume / 10 Day Relative Volume",
                "Momentum Score"
            ])
        return pd.DataFrame(columns=[
            "Symbol", "LTP", "% Change",
            "Daily Volatility Expansion",
            "Today Volume / 10 Day Relative Volume",
            "Short Momentum Score"
        ])

    if kind == "long":
        out = df[
            (df["% Change"] > 0) &
            (df["Daily Volatility Expansion"] > 1) &
            (df["Today Volume / 10 Day Relative Volume"] > 1)
        ].copy()

        out = out.sort_values(
            ["Momentum Score", "% Change"],
            ascending=[False, False]
        ).head(maxrows)

        cols = [
            "Symbol", "LTP", "% Change",
            "Daily Volatility Expansion",
            "Today Volume / 10 Day Relative Volume",
            "Momentum Score"
        ]
    else:
        out = df[
            (df["% Change"] < 0) &
            (df["Daily Volatility Expansion"] > 1) &
            (df["Today Volume / 10 Day Relative Volume"] > 1)
        ].copy()

        out = out.sort_values(
            ["Short Momentum Score", "% Change"],
            ascending=[False, True]
        ).head(maxrows)

        cols = [
            "Symbol", "LTP", "% Change",
            "Daily Volatility Expansion",
            "Today Volume / 10 Day Relative Volume",
            "Short Momentum Score"
        ]

    return out[cols].copy()


def dftohtmltable(df: pd.DataFrame, maxrows: int = 15) -> str:
    if df is None or df.empty:
        return "<p>No data available.</p>"

    dfdisp = df.copy().head(maxrows)

    for col in dfdisp.columns:
        if col == "Symbol":
            continue
        elif col == "LTP":
            dfdisp[col] = dfdisp[col].map(lambda x: f"{x:,.2f}" if pd.notna(x) else "")
        elif col == "Current Volume":
            dfdisp[col] = dfdisp[col].map(lambda x: f"{x:,.0f}" if pd.notna(x) else "")
        else:
            dfdisp[col] = dfdisp[col].map(lambda x: f"{x:.2f}" if pd.notna(x) else "")

    return dfdisp.to_html(index=False, border=1, justify="center", escape=False)


def send_email_with_tables(dfall: pd.DataFrame, csvfilename: str) -> bool:
    senderemail = getcfg("email", "senderemail", envname="SENDEREMAIL")
    senderpassword = getcfg("email", "senderpassword", envname="SENDERPASSWORD")
    recipientemail = getcfg("email", "recipientemail", envname="RECIPIENTEMAIL")
    smtpserver = getcfg("email", "smtpserver", envname="SMTPSERVER", default="smtp.gmail.com")
    smtpport = getcfg("email", "smtpport", envname="SMTPPORT", default=587, isint=True)

    if not all([senderemail, senderpassword, recipientemail]):
        logger.warning("Missing email credentials")
        return False

    now = datetime.now()
    subject = f"FO Momentum Scan - {now.strftime('%Y-%m-%d %H:%M IST')}"

    long_df = format_display_table(dfall, "long", maxrows=15)
    short_df = format_display_table(dfall, "short", maxrows=15)

    vol_df = dfall.sort_values(
        by="Daily Volatility Expansion",
        ascending=False,
        na_position="last"
    ).head(15).copy()

    vol_df = vol_df[
        [
            "Symbol", "LTP", "% Change",
            "Daily Volatility Expansion",
            "Today Volume / 10 Day Relative Volume",
            "Momentum Score"
        ]
    ]

    long_html = dftohtmltable(long_df, maxrows=15)
    short_html = dftohtmltable(short_df, maxrows=15)
    vol_html = dftohtmltable(vol_df, maxrows=15)

    bodyhtml = f"""
    <html>
    <head>
    <style>
        body {{ font-family: Arial, sans-serif; font-size: 13px; color: #222; }}
        table {{ border-collapse: collapse; width: 100%; margin-bottom: 18px; }}
        th, td {{ border: 1px solid #dddddd; padding: 6px; text-align: right; }}
        th {{ background-color: #f2f2f2; text-align: center; }}
        td:first-child {{ text-align: left; font-weight: bold; }}
        h2, h3 {{ margin-bottom: 6px; }}
        .note {{ color: #555; font-size: 12px; }}
    </style>
    </head>
    <body>
        <p>Hello,</p>
        <p>Below is the latest FO scan using <b>Daily Volatility Expansion</b>, <b>Relative Volume</b>, and <b>Momentum Score</b>.</p>

        <h3>LONG Candidates (Top 15)</h3>
        <p class="note">Rule: % Change &gt; 0, VolExp &gt; 1, VolRel &gt; 1, sorted by Momentum Score.</p>
        {long_html}

        <h3>SHORT Candidates (Top 15)</h3>
        <p class="note">Rule: % Change &lt; 0, VolExp &gt; 1, VolRel &gt; 1, sorted by Short Momentum Score.</p>
        {short_html}

        <h3>Top Daily Volatility Expansion</h3>
        <p class="note">Reference table sorted by Daily Volatility Expansion.</p>
        {vol_html}

        <h3>Formula Guide</h3>
        <ul>
            <li><b>Momentum Score</b> = % Change × Daily Volatility Expansion × Today Volume / 10 Day Relative Volume</li>
            <li><b>Short Momentum Score</b> = -(% Change) × Daily Volatility Expansion × Today Volume / 10 Day Relative Volume</li>
            <li><b>Daily Volume Expansion</b> = max(10 Day Relative Volume, 20 Day Relative Volume)</li>
        </ul>

        <p>Attached CSV: {os.path.basename(csvfilename)}</p>
        <p>Generated at {now.strftime('%Y-%m-%d %H:%M:%S')}</p>
    </body>
    </html>
    """

    msg = MIMEMultipart()
    msg["From"] = senderemail
    msg["To"] = recipientemail
    msg["Subject"] = subject
    msg.attach(MIMEText(bodyhtml, "html"))

    try:
        with open(csvfilename, "rb") as f:
            part = MIMEBase("application", "octet-stream")
            part.set_payload(f.read())
        encoders.encode_base64(part)
        part.add_header("Content-Disposition", f"attachment; filename={os.path.basename(csvfilename)}")
        msg.attach(part)
    except Exception as e:
        logger.warning(f"Failed to attach CSV: {e}")

    try:
        with smtplib.SMTP(smtpserver, smtpport) as server:
            server.starttls()
            server.login(senderemail, senderpassword)
            server.send_message(msg)
        logger.info("Email sent successfully")
        return True
    except Exception as e:
        logger.error(f"Email sending failed: {e}")
        return False


def main():
    dfall = scan_fno_universe()
    if dfall is None or dfall.empty:
        logger.error("No data to email. Exiting.")
        return

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    csvfilename = f"fo_fyers_expansion_scan_{ts}.csv"

    ordered_cols = [
        "Symbol",
        "LTP",
        "% Change",
        "Current Daily Volatility",
        "Avg Daily Volatility",
        "Daily Volatility Expansion",
        "Current Volume",
        "Today Volume / 10 Day Relative Volume",
        "Today Volume / 20 Day Relative Volume",
        "Daily Volume Expansion",
        "Momentum Score",
        "Short Momentum Score",
        "Trade Side",
    ]

    dfall = dfall[ordered_cols].copy()
    dfall.to_csv(csvfilename, index=False)
    logger.info(f"Saved scan to {csvfilename}")

    send_email_with_tables(dfall, csvfilename)


if __name__ == "__main__":
    main()
