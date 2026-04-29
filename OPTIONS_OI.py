import os
import smtplib
import logging
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from email.mime.base import MIMEBase
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email import encoders

try:
    from fyers_apiv3 import fyersModel
except:
    from fyersapiv3 import fyersModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

def get_fyers():
    return fyersModel.FyersModel(client_id=os.environ.get("CLIENT_ID"), is_async=False, 
                                 token=os.environ.get("ACCESS_TOKEN"), log_path="")

def get_data(fyers, symbol, res, days):
    try:
        data = fyers.history(data={"symbol": symbol, "resolution": res, "date_format": "1", 
                                   "range_from": (datetime.now()-timedelta(days=days)).strftime("%Y-%m-%d"), 
                                   "range_to": datetime.now().strftime("%Y-%m-%d"), "cont_flag": "1"})
        return pd.DataFrame(data["candles"], columns=["ts", "o", "h", "l", "c", "v"])
    except: return pd.DataFrame()

def summarize(intra):
    if intra.empty: return {"LTP": 0, "Rank": 0}
    c = intra["c"]
    delta = c.diff().fillna(0)
    rsi = 100 - (100 / (1 + (delta.clip(lower=0).rolling(14).mean() / (-delta).clip(lower=0).rolling(14).mean()).replace(0, np.nan))).fillna(50)
    rank = int((c.iloc[-1] > c.iloc[0]) + (rsi.iloc[-1] > 50))
    return {"LTP": c.iloc[-1], "Rank": rank}

def main():
    fyers = get_fyers()
    # Scans symbols from 'sectors' folder
    symbols = []
    if os.path.exists("sectors"):
        for root, _, files in os.walk("sectors"):
            for f in files:
                if f.endswith(".csv"):
                    df = pd.read_csv(os.path.join(root, f))
                    symbols.extend(df.iloc[:, 0].dropna().astype(str).tolist())

    # Analyze
    results = []
    for sym in symbols[:10]:
        intra = get_data(fyers, f"NSE:{sym}-EQ", "5", 5)
        res = summarize(intra)
        res.update({"Symbol": sym})
        results.append(res)

    summary = pd.DataFrame(results)
    print("Top Candidates:")
    print(summary.sort_values("Rank", ascending=False))

if __name__ == "__main__":
    main()
