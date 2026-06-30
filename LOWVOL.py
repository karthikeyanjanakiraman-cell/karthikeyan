#!/usr/bin/env python3
"""
FO_FNO_FYERS_CONFLUENCE_EMAIL.py - Index & Stock Confluence Screener (No Thresholds)
"""

import os
import sys
import logging
import warnings
from datetime import datetime, timedelta
from typing import Dict

import numpy as np
import pandas as pd
from fyers_apiv3 import fyersModel

import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders

class Config:
    def __init__(self):
        self.client_id = os.environ.get("CLIENT_ID") or os.environ.get("CLIENTID")
        self.access_token = os.environ.get("ACCESS_TOKEN") or os.environ.get("ACCESSTOKEN")
        self.smtp_host = os.environ.get("SMTP_HOST", "smtp.gmail.com")
        self.smtp_port = int(os.environ.get("SMTP_PORT", "587"))
        self.sender_email = os.environ.get("SENDER_EMAIL", "you@example.com")
        self.sender_password = os.environ.get("SENDER_PASSWORD", "password")
        self.recipient_email = os.environ.get("RECIPIENT_EMAIL", "you@example.com")
        
        # 1. The Indices (Always shown in the dashboard)
        self.index_symbols = ["NSE:NIFTY50-INDEX", "NSE:NIFTYBANK-INDEX", "BSE:SENSEX-INDEX"]
        
        # 2. Add your F&O Stocks here
        self.stock_symbols = [
            "NSE:RELIANCE-EQ", "NSE:HDFCBANK-EQ", "NSE:ICICIBANK-EQ",
            "NSE:INFY-EQ", "NSE:TCS-EQ", "NSE:SBIN-EQ", 
            "NSE:ITC-EQ", "NSE:LT-EQ", "NSE:AXISBANK-EQ", "NSE:KOTAKBANK-EQ"
        ]

cfg = Config()

# The clean layout you requested
EMAIL_DISPLAY_COLS = [
    "Symbol", "% Change",
    "Conf_Below-3", "Conf_Below-2", "Conf_Below-1", 
    "LTP", 
    "Conf_Above-1", "Conf_Above-2", "Conf_Above-3"
]

logger = logging.getLogger()
logger.setLevel(logging.INFO)
if logger.hasHandlers():
    logger.handlers.clear()
ch = logging.StreamHandler(sys.stdout)
ch.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
logger.addHandler(ch)
warnings.filterwarnings("ignore")

def format_value(val):
    if pd.isna(val) or val in [float("inf"), float("-inf")]:
        return "-"
    if isinstance(val, (int, float, np.integer, np.floating)):
        return f"{float(val):.2f}"
    return str(val)

def format_change(val):
    try:
        return f"{float(val):.2f}%"
    except (TypeError, ValueError):
        return ""

def init_fyers():
    try:
        return fyersModel.FyersModel(client_id=cfg.client_id, is_async=False, token=cfg.access_token, log_path="")
    except Exception as e:
        logger.warning(f"INIT Failed: {e}")
        return None

def get_history(fyers, symbol, res, days):
    try:
        now = datetime.now()
        start, end = (now - timedelta(days=days)).strftime("%Y-%m-%d"), now.strftime("%Y-%m-%d")
        res_data = fyers.history(data={
            "symbol": symbol,
            "resolution": res,
            "date_format": "1",
            "range_from": start,
            "range_to": end,
            "cont_flag": "1",
        })
        if res_data and "candles" in res_data:
            df = pd.DataFrame(res_data["candles"], columns=["timestamp", "open", "high", "low", "close", "volume"])
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")
            return df
        return None
    except Exception:
        return None

def get_opening_anchor(fyers, symbol):
    """Fetches the 9:15 AM price to act as the day's structural anchor."""
    today = datetime.now().strftime("%Y-%m-%d")
    data = fyers.history(data={
        "symbol": symbol, "resolution": "5", "date_format": "1",
        "range_from": today, "range_to": today, "cont_flag": "1"
    })
    if data and "candles" in data:
        # First 5-min candle is index 0
        return float(data['candles'][0][4]) # Close price of 9:15 candle
    return None

def scan_universe(fyers, symbol_list, is_stock=False):
    lookback_days = 150
    dedupe_pct = 0.005
    match_tolerance = 0.005
    rows = []

    for sym in symbol_list:
        # 1. Get the 9:15 AM Anchor
        anchor_price = get_opening_anchor(fyers, sym)
        if not anchor_price: continue
            
        daily = get_history(fyers, sym, "D", lookback_days)
        # 2. Lock to historical data based on opening context
        valid_daily = daily[daily["volume"] > 0]
        
        # Confluence logic (Standardized search)
        hv_work = valid_daily.sort_values(["volume", "high"], ascending=[False, False]).head(60)
        lv_work = valid_daily.sort_values(["volume", "high"], ascending=[True, False]).head(60)
        
        # ... [Confluence extraction logic remains same] ...
        
        # 3. Final row assembly using anchor_price as reference
        row = {
            "Symbol": sym, 
            "Anchor_915": anchor_price,
            "LTP": float(daily["close"].iloc[-1]),
            # ... rest of your columns ...
        }
        rows.append(row)
    return pd.DataFrame(rows)
                
    
def build_html_table(df, title, cols):
    if df.empty:
        return f"<h3 style='color:#fbbf24; font-family:sans-serif; margin-top: 25px;'>{title}</h3><p style='color:#94a3b8; font-family:sans-serif;'>No candidates found today.</p>"
    table_html = f"<h3 style='color:#fbbf24; font-family:sans-serif; margin-top: 25px;'>{title}</h3><table style='border-collapse: collapse; width: 100%; font-family: sans-serif; font-size: 13px; text-align: left; background-color: #0f172a;'>"
    table_html += "<tr style='background-color: #1e293b; color: #f1f5f9;'>" + "".join([f"<th style='padding: 10px; border: 1px solid #334155;'>{c}</th>" for c in cols]) + "</tr>"
    
    for i, (_, row) in enumerate(df.iterrows()):
        bg_row = "#0f172a" if i % 2 == 0 else "#1e293b"
        table_html += f"<tr style='background-color: {bg_row}; color: #e2e8f0;'>"
        for c in cols:
            val = row.get(c, "-")
            style = "padding: 8px; border: 1px solid #334155;"
            if c == "% Change":
                val_str = format_change(val)
                style += " color: #4ade80; font-weight: bold;" if float(row["% Change"]) > 0 else " color: #f87171; font-weight: bold;"
                table_html += f"<td style='{style}'>{val_str}</td>"
            else:
                table_html += f"<td style='{style}'>{val}</td>"
        table_html += "</tr>"
    return table_html + "</table>"

def send_email(index_df, long_df, short_df):
    try:
        msg = MIMEMultipart()
        msg["From"] = cfg.sender_email
        msg["To"] = cfg.recipient_email
        msg["Subject"] = f"Confluence Trade Setups - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        
        html = (
            "<body style='background-color: #030712; padding: 20px; font-family: sans-serif;'>"
            + build_html_table(index_df, "Market Index Confluence Dashboard", EMAIL_DISPLAY_COLS)
            + build_html_table(long_df, "F&O Long Candidates (Ordered by Proximity to Support)", EMAIL_DISPLAY_COLS)
            + build_html_table(short_df, "F&O Short Candidates (Ordered by Proximity to Resistance)", EMAIL_DISPLAY_COLS)
            + "</body>"
        )
        msg.attach(MIMEText(html, "html"))
        with smtplib.SMTP(cfg.smtp_host, cfg.smtp_port) as s:
            s.starttls()
            s.login(cfg.sender_email, cfg.sender_password)
            s.sendmail(cfg.sender_email, cfg.recipient_email, msg.as_string())
        logger.info("Confluence Email sent successfully.")
    except Exception as e:
        logger.error(f"Email Failed: {e}")

def main():
    fyers = init_fyers()
    if not fyers:
        return
    
    # 1. Scan the Indices (Always display these)
    logger.info("Starting Index Scan...")
    index_df = scan_universe(fyers, cfg.index_symbols, is_stock=False)
    
    # 2. Scan the F&O Stocks 
    logger.info("Starting F&O Stock Scan...")
    stock_df = scan_universe(fyers, cfg.stock_symbols, is_stock=True)

    long_stocks = pd.DataFrame()
    short_stocks = pd.DataFrame()

    # 3. Split the stocks into Long and Short (No filtering, just sorting)
    if not stock_df.empty:
        long_stocks = stock_df[stock_df["Signal"] == "Long"]
        short_stocks = stock_df[stock_df["Signal"] == "Short"]

    send_email(index_df, long_stocks, short_stocks)

if __name__ == "__main__":
    main()
