import os
import json
import time
import smtplib
import logging
from datetime import datetime, timedelta, time as dtime
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
        try:
            from fyersapi import fyersModel
        except Exception:
            fyersModel = None

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

DAILYLOOKBACKDAYS = 252
INTRADAYLOOKBACKDAYS = 20
IVPLOOKBACKDAYS = 252
OPTIONPAIRSTOKEEP = 5
SIGNALWINDOWMINUTES = 5
ITERATIONSTOKEEP = 75
SECTORSDIR = os.environ.get("SECTORSDIR", "sectors")
OUTPUTDIR = os.environ.get("OUTPUTDIR", ".")
MINOPTIONLTP = 10.0
MINATMCHAINVOLUME = int(os.environ.get("MINATMCHAINVOLUME", "100000"))
PERSYMBOLSLEEPSEC = float(os.environ.get("PERSYMBOLSLEEPSEC", "0.25"))
EMAILMAXROWSLONG = int(os.environ.get("EMAILMAXROWSLONG", "25"))
EMAILMAXROWSSHORT = int(os.environ.get("EMAILMAXROWSSHORT", "25"))
EMAILSAFEWIDTH = int(os.environ.get("EMAILSAFEWIDTH", "900"))
TOPNUNDERLYINGS = int(os.environ.get("TOPNUNDERLYINGS", "60"))
OBVBREAKOUTWINDOW = int(os.environ.get("OBVBREAKOUTWINDOW", "5"))
T30MIN = int(os.environ.get("T30MIN", "2"))
DAILYSTATEFILE = os.path.join(OUTPUTDIR, "chainsignalstate.json")

OPTIONEMAILCOLS = [
    "Underlying", "Option Type", "Option Symbol", "Strike", "LTP", "Change", "OI", "Volume", "OBV",
    "OIVolumeOBV Score", "EMAILRANKSCORE", "Rank Delta", "Cumulative ADX", "5mSignal", "15mSignal",
    "30mSignal", "60mSignal", "BullSignal", "BearSignal", "OverallSignal", "PriceLeadStatus", "IVP",
    "Volatility State", "Last Iteration Time",
]
OPTIONEMAILCOLRENAME = {
    "OIVolumeOBV Score": "Liq Score",
    "EMAILRANKSCORE": "Rank",
    "Change": "Chg",
    "Last Iteration Time": "Time",
    "PriceLeadStatus": "Lead",
    "Volatility State": "Vol State",
    "Option Symbol": "Opt Symbol",
}

fyers = None


def initfyers() -> Optional[object]:
    global fyers
    if fyersModel is None:
        logger.warning("Fyers SDK not available; running in dry mode")
        fyers = None
        return None
    clientid = os.environ.get("CLIENTID") or os.environ.get("CLIENT_ID")
    accesstoken = os.environ.get("ACCESSTOKEN") or os.environ.get("ACCESS_TOKEN")
    if not clientid or not accesstoken:
        logger.error("Missing CLIENTID / ACCESSTOKEN environment variables.")
        fyers = None
        return None
    try:
        fyers = fyersModel.FyersModel(client_id=clientid, token=accesstoken, is_async=False, log_path="")
        return fyers
    except Exception as e:
        logger.exception("Failed to initialize Fyers: %s", e)
        fyers = None
        return None


def safefloat(value, default=np.nan) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except Exception:
        return default


def safeseries(frame: pd.DataFrame, col: str, default: float = 0.0) -> pd.Series:
    if frame is not None and col in frame.columns:
        return pd.to_numeric(frame[col], errors="coerce").fillna(default)
    if frame is None:
        return pd.Series(dtype=float)
    return pd.Series(default, index=frame.index, dtype=float)


def discoversectorcsvs(rootdir: str = SECTORSDIR) -> List[str]:
    if not os.path.isdir(rootdir):
        return []
    paths = []
    for dirpath, _, filenames in os.walk(rootdir):
        for fname in filenames:
            if fname.lower().endswith(".csv"):
                paths.append(os.path.join(dirpath, fname))
    return sorted(set(paths))


def loadfnosymbolsfromsectors(rootdir: str = SECTORSDIR) -> List[str]:
    symbols = set()
    for path in discoversectorcsvs(rootdir):
        try:
            df = pd.read_csv(path)
        except Exception:
            continue
        lowered = {str(c).strip().lower(): c for c in df.columns}
        symbolcol = next((lowered[k] for k in ["symbol", "symbols", "ticker", "tradingsymbol"] if k in lowered), None)
        if symbolcol is None:
            continue
        for raw in df[symbolcol].dropna().astype(str):
            sym = raw.strip().upper()
            if sym and sym not in ("NAN", "NONE"):
                symbols.add(sym)
    return sorted(symbols)


def formateqsymbol(symbol: str) -> str:
    symbol = str(symbol).strip().upper()
    if symbol.startswith("NSE:"):
        return symbol
    if symbol.endswith("-EQ"):
        return f"NSE:{symbol}"
    return f"NSE:{symbol}-EQ"


def gethistory(symbol: str, resolution: str, daysback: int) -> pd.DataFrame:
    if fyers is None:
        return pd.DataFrame()
    now = datetime.now()
    start = now - timedelta(days=daysback)
    payload = {
        "symbol": symbol,
        "resolution": resolution,
        "date_format": "1",
        "range_from": start.strftime("%Y-%m-%d"),
        "range_to": now.strftime("%Y-%m-%d"),
        "cont_flag": "1",
    }
    try:
        res = fyers.history(payload)
    except Exception:
        return pd.DataFrame()
    candles = (res or {}).get("candles", [])
    if not candles:
        return pd.DataFrame()
    df = pd.DataFrame(candles, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s", utc=True).dt.tz_convert("Asia/Kolkata").dt.tz_localize(None)
    return df.sort_values("timestamp").drop_duplicates(subset=["timestamp"], keep="last").reset_index(drop=True)


def neareststep(val: float) -> int:
    val = abs(safefloat(val, 0))
    if val >= 20000:
        return 100
    if val >= 10000:
        return 50
    if val >= 2000:
        return 20
    if val >= 500:
        return 10
    if val >= 100:
        return 5
    return 1


def computeobv(df: pd.DataFrame) -> float:
    if df is None or df.empty or len(df) < 2:
        return np.nan
    close = pd.to_numeric(df["close"], errors="coerce")
    vol = pd.to_numeric(df["volume"], errors="coerce").fillna(0.0)
    direction = np.sign(close.diff())
    obv = (vol * direction).cumsum()
    return round(safefloat(obv.iloc[-1], np.nan), 2)


def computetodayobv(df: pd.DataFrame) -> float:
    if df is None or df.empty or len(df) < 2:
        return np.nan
    d = df.copy().sort_values("timestamp")
    d["timestamp"] = pd.to_datetime(d["timestamp"])
    today = pd.Timestamp.now().tz_localize(None).date()
    todaydf = d[d["timestamp"].dt.date == today].copy()
    if todaydf.empty:
        todaydf = d.copy()
    todaydf["timestamp"] = pd.Timestamp.combine(today, dtime(9, 15))
    if len(todaydf) < 2:
        return computeobv(todaydf)
    return computeobv(todaydf)


def computeivp(historydf: pd.DataFrame, minbars: int = 10) -> Tuple[float, str]:
    if historydf is None or historydf.empty or len(historydf) < minbars:
        return np.nan, "Neutral Vol"
    close = pd.to_numeric(historydf["close"], errors="coerce")
    high = pd.to_numeric(historydf["high"], errors="coerce")
    low = pd.to_numeric(historydf["low"], errors="coerce")
    proxy = ((high - low) / close.replace(0, np.nan) * 100.0).replace([np.inf, -np.inf], np.nan).dropna()
    if proxy.empty:
        return np.nan, "Neutral Vol"
    lookback = proxy.tail(min(IVPLOOKBACKDAYS, len(proxy)))
    current = float(lookback.iloc[-1])
    ivp = round((lookback.lt(current).sum() / len(lookback)) * 100, 2)
    if ivp < 30:
        return ivp, "Buyer Zone"
    if ivp > 50:
        return ivp, "Avoid Buy Premium"
    return ivp, "Neutral Vol"


def scorelabel(delta: float) -> str:
    if pd.isna(delta):
        return "Neutral"
    if delta >= 7:
        return "Buy"
    if delta >= 4:
        return "Buy"
    if delta >= 1:
        return "Buy"
    if delta <= -7:
        return "Sell"
    if delta <= -4:
        return "Sell"
    if delta <= -1:
        return "Sell"
    return "Neutral"


def directionallabel(rawlabel: str, side: str) -> str:
    raw = str(rawlabel).strip().upper()
    if side.lower() == "long":
        return {"BUY": "BUY LONG", "SELL": "SELL SHORT"}.get(raw, raw)
    return {"BUY": "SELL SHORT", "SELL": "BUY LONG"}.get(raw, raw)


def applydisplaylabels(df: pd.DataFrame, side: str) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    out = df.copy()
    for col in ["5mSignal", "15mSignal", "30mSignal", "60mSignal", "BullSignal", "BearSignal", "OverallSignal"]:
        if col in out.columns:
            out[col] = out[col].apply(lambda x: directionallabel(str(x), side))
    return out


def intradaywindowscore(df: pd.DataFrame, windowminutes: int = SIGNALWINDOWMINUTES) -> float:
    if df is None or df.empty or len(df) < 2:
        return np.nan
    d = df.copy().sort_values("timestamp")
    endts = pd.to_datetime(d["timestamp"].iloc[-1])
    startts = endts - timedelta(minutes=windowminutes)
    cur = d[(d["timestamp"] >= startts) & (d["timestamp"] <= endts)]
    if cur.empty or len(cur) < 2:
        return np.nan
    fc = safefloat(cur["close"].iloc[0])
    lc = safefloat(cur["close"].iloc[-1])
    if pd.isna(fc) or fc == 0:
        return np.nan
    return round(((lc - fc) / fc) * 100.0, 2)


def previoustradingdaysametimescore(fulldf: pd.DataFrame, endts: Optional[pd.Timestamp] = None, windowminutes: int = SIGNALWINDOWMINUTES) -> float:
    if fulldf is None or fulldf.empty or len(fulldf) < 4:
        return np.nan
    d = fulldf.copy().sort_values("timestamp").reset_index(drop=True)
    d["timestamp"] = pd.to_datetime(d["timestamp"])
    endts = pd.to_datetime(endts) if endts is not None else pd.to_datetime(d["timestamp"].iloc[-1])
    tradingdays = sorted(d["timestamp"].dt.date.unique())
    prevdays = [day for day in tradingdays if day < endts.date()]
    if not prevdays:
        return np.nan
    prevday = prevdays[-1]
    prevdaydata = d[d["timestamp"].dt.date == prevday].copy()
    if prevdaydata.empty:
        return np.nan
    sametimerows = prevdaydata[(prevdaydata["timestamp"].dt.hour == endts.hour) & (prevdaydata["timestamp"].dt.minute == endts.minute)]
    prevend = pd.to_datetime(sametimerows["timestamp"].iloc[-1] if not sametimerows.empty else prevdaydata["timestamp"].iloc[-1])
    prevstart = prevend - timedelta(minutes=windowminutes)
    prevwindow = prevdaydata[(prevdaydata["timestamp"] >= prevstart) & (prevdaydata["timestamp"] <= prevend)]
    if prevwindow.empty or len(prevwindow) < 2:
        return np.nan
    fc = safefloat(prevwindow["close"].iloc[0])
    lc = safefloat(prevwindow["close"].iloc[-1])
    if pd.isna(fc) or fc == 0:
        return np.nan
    return round(((lc - fc) / fc) * 100.0, 2)


def comparewindowsignal(currentscore: float, previousscore: float) -> Tuple[float, str]:
    if pd.isna(currentscore) or pd.isna(previousscore):
        return np.nan, "Neutral"
    delta = round(currentscore - previousscore, 2)
    if delta >= 0.50:
        return delta, "Buy"
    if delta >= 0.20:
        return delta, "Buy"
    if delta >= 0.05:
        return delta, "Buy"
    if delta <= -0.50:
        return delta, "Sell"
    if delta <= -0.20:
        return delta, "Sell"
    if delta <= -0.05:
        return delta, "Sell"
    return delta, "Neutral"


def builditerationhistory(intradf: pd.DataFrame, windowminutes: int = SIGNALWINDOWMINUTES, iterations: int = ITERATIONSTOKEEP) -> pd.DataFrame:
    if intradf is None or intradf.empty:
        return pd.DataFrame()
    fulldf = intradf.copy().sort_values("timestamp").reset_index(drop=True)
    fulldf["timestamp"] = pd.to_datetime(fulldf["timestamp"])
    lastday = fulldf["timestamp"].dt.date.max()
    d = fulldf[fulldf["timestamp"].dt.date == lastday].copy().reset_index(drop=True)
    if d.empty:
        return pd.DataFrame()
    startanchor = pd.Timestamp.combine(pd.Timestamp(lastday).date(), dtime(9, 15))
    endanchor = pd.Timestamp.combine(pd.Timestamp(lastday).date(), dtime(15, 30))
    d = d[(d["timestamp"] >= startanchor) & (d["timestamp"] <= endanchor)].copy().reset_index(drop=True)
    if d.empty:
        return pd.DataFrame()
    rows = []
    for i in range(len(d)):
        endts = pd.to_datetime(d.loc[i, "timestamp"])
        startts = endts - timedelta(minutes=windowminutes)
        cur = d[(d["timestamp"] >= startts) & (d["timestamp"] <= endts)]
        if cur.empty or len(cur) < 2:
            continue
        fc = safefloat(cur["close"].iloc[0])
        lc = safefloat(cur["close"].iloc[-1])
        if pd.isna(fc) or fc == 0:
            continue
        currentscore = round(((lc - fc) / fc) * 100.0, 2)
        prevscore = previoustradingdaysametimescore(fulldf, endts, windowminutes)
        delta, signal = comparewindowsignal(currentscore, prevscore)
        rows.append({
            "iteration": len(rows) + 1,
            "timestamp": endts.strftime("%H:%M"),
            "windowminutes": windowminutes,
            "windowstart": startts.strftime("%H:%M"),
            "windowend": endts.strftime("%H:%M"),
            "currentwindowscore": currentscore,
            "previoustradingdaysametimescore": prevscore,
            "windowdelta": delta,
            "windowsignal": signal,
            "close": lc,
        })
        if len(rows) >= iterations:
            break
    return pd.DataFrame(rows)


def summarizeintraday(intradf: pd.DataFrame, referencedf: pd.DataFrame) -> Dict[str, object]:
    if intradf is None or intradf.empty:
        return {}
    df = intradf.copy().sort_values("timestamp").reset_index(drop=True)
    close = pd.to_numeric(df["close"], errors="coerce")
    high = pd.to_numeric(df["high"], errors="coerce")
    low = pd.to_numeric(df["low"], errors="coerce")
    volume = pd.to_numeric(df["volume"], errors="coerce").fillna(0.0)
    delta = close.diff().fillna(0.0)
    avggain = delta.clip(lower=0.0).rolling(14, min_periods=14).mean()
    avgloss = (-delta.clip(upper=0.0)).rolling(14, min_periods=14).mean()
    rs = avggain / avgloss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    tr = pd.concat([(high - low).abs(), (high - close.shift(1)).abs(), (low - close.shift(1)).abs()], axis=1).max(axis=1)
    upmove = high.diff()
    downmove = -low.diff()
    plusdm = np.where((upmove > downmove) & (upmove > 0), upmove, 0.0)
    minusdm = np.where((downmove > upmove) & (downmove > 0), downmove, 0.0)
    atr = pd.Series(tr).rolling(14, min_periods=14).mean()
    plusdi = 100 * pd.Series(plusdm).rolling(14, min_periods=14).mean() / atr.replace(0, np.nan)
    minusdi = 100 * pd.Series(minusdm).rolling(14, min_periods=14).mean() / atr.replace(0, np.nan)
    dx = 100 * (plusdi - minusdi).abs() / (plusdi + minusdi).replace(0, np.nan)
    adx = dx.rolling(14, min_periods=14).mean().fillna(0)
    typical = (high + low + close) / 3.0
    cumvol = volume.cumsum().replace(0, np.nan)
    vwap = (typical * volume).cumsum() / cumvol.ffill().fillna(close)
    vwapstd = (typical - vwap).pow(2) * volume.cumsum() / cumvol.pow(0.5).replace(0, np.nan)
    vwapz = (close - vwap) / vwapstd.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    rangenow = (high - low).clip(lower=0)
    avgrange5 = rangenow.rolling(5, min_periods=3).mean()
    avgvol5 = volume.rolling(5, min_periods=3).mean()
    priceleadflag = (rangenow > avgrange5.replace(0, np.nan) * 1.5) & (volume > avgvol5.replace(0, np.nan) * 1.0)
    streak = []
    run = 0
    for flag in priceleadflag.fillna(False).astype(bool):
        run = run + 1 if flag else 0
        streak.append(run)
    streak = pd.Series(streak, index=df.index)
    leadstatus = pd.Series(np.select([priceleadflag & (streak >= 3), priceleadflag & (streak == 2), priceleadflag], ["STRONG_PRICE_LEAD_FADE", "PRICE_LEADING_FADE_RISK", "EARLY_PRICE_LEAD"], default="NORMAL"), index=df.index)
    prevclose = safefloat(referencedf["close"].iloc[-2] if referencedf is not None and len(referencedf) >= 2 else np.nan)
    ltp = safefloat(close.iloc[-1])
    pctchange = ((ltp - prevclose) / prevclose * 100.0) if pd.notna(prevclose) and prevclose != 0 else 0.0
    currentwin = intradaywindowscore(df)
    prevwin = previoustradingdaysametimescore(df)
    winsignal = comparewindowsignal(currentwin, prevwin)[1]
    iterationhistory = builditerationhistory(df)
    bull = 0 if pctchange < 0 else 1
    bear = 0 if pctchange > 0 else 1
    bull += 1 if safefloat(vwapz.iloc[-1], 0) > 0.30 else 0
    bear += 1 if safefloat(vwapz.iloc[-1], 0) < -0.30 else 0
    bull += 1 if safefloat(plusdi.iloc[-1], 0) > safefloat(minusdi.iloc[-1], 0) else 0
    bear += 1 if safefloat(minusdi.iloc[-1], 0) > safefloat(plusdi.iloc[-1], 0) else 0
    bull += 1 if safefloat(adx.iloc[-1], 0) > 20 and safefloat(plusdi.iloc[-1], 0) > safefloat(minusdi.iloc[-1], 0) else 0
    bear += 1 if safefloat(adx.iloc[-1], 0) > 20 and safefloat(minusdi.iloc[-1], 0) > safefloat(plusdi.iloc[-1], 0) else 0
    bull += 1 if safefloat(rsi.iloc[-1], 50) > 55 else 0
    bear += 1 if safefloat(rsi.iloc[-1], 50) < 45 else 0
    bull += 2 if winsignal.startswith("Buy") else 0
    bear += 2 if winsignal.startswith("Sell") else 0
    rankdelta = bull - bear
    ivp, volstate = computeivp(referencedf, minbars=10)
    lastts = pd.to_datetime(df["timestamp"].iloc[-1])
    return {
        "LTP": round(ltp, 2),
        "Change": round(pctchange, 2),
        "5mSignal": scorelabel(rankdelta),
        "15mSignal": winsignal,
        "30mSignal": scorelabel(rankdelta * 0.8),
        "60mSignal": scorelabel(rankdelta * 0.7),
        "BullSignal": scorelabel(bull),
        "BearSignal": scorelabel(-bear),
        "OverallSignal": scorelabel(rankdelta),
        "PriceLeadStatus": str(leadstatus.iloc[-1]),
        "IVP": ivp,
        "Volatility State": volstate,
        "Last Iteration Time": lastts.strftime("%H:%M"),
        "Bull Rank": bull,
        "Bear Rank": bear,
        "Rank Delta": rankdelta,
        "Cumulative ADX": round(safefloat(adx.iloc[-1], np.nan), 2),
        "Iteration History": iterationhistory,
    }


def choosetopcandidates(summarydf: pd.DataFrame, topn: int = TOPNUNDERLYINGS) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if summarydf is None or summarydf.empty:
        return pd.DataFrame(), pd.DataFrame()
    rankdelta = safeseries(summarydf, "Rank Delta", 0)
    longdf = summarydf[rankdelta > 0].copy()
    shortdf = summarydf[rankdelta < 0].copy()
    longdf = longdf.sort_values(["Rank Delta", "Cumulative ADX", "Change"], ascending=[False, False, False]).head(topn)
    shortdf = shortdf.sort_values(["Rank Delta", "Cumulative ADX", "Change"], ascending=[True, False, True]).head(topn)
    return longdf.reset_index(drop=True), shortdf.reset_index(drop=True)


def fetchoptionpairs(symbol: str, paircount: int = OPTIONPAIRSTOKEEP) -> pd.DataFrame:
    if fyers is None:
        return pd.DataFrame()
    eqsymbol = formateqsymbol(symbol)
    try:
        quote = fyers.quotes({"symbols": [eqsymbol]})
        ltp = safefloat((quote or {}).get("d", [{}])[0].get("v", {}).get("lp"), np.nan)
        chainres = fyers.optionchain({"symbol": eqsymbol, "strikecount": 50})
    except Exception:
        return pd.DataFrame()
    chain = (chainres or {}).get("data") or (chainres or {}).get("optionsChain") or []
    rows = []
    for item in chain:
        strike = safefloat(item.get("strikePrice") or item.get("strike"), np.nan)
        typ = str(item.get("optionType") or item.get("type") or "").upper()
        if pd.isna(strike) or typ not in ("CE", "PE"):
            continue
        rows.append({
            "Strike": strike,
            "Option Type": typ,
            "Option Symbol": str(item.get("symbol") or item.get("tradingSymbol") or "").strip(),
            "OI": safefloat(item.get("oi") or item.get("openInterest"), np.nan),
            "Chain Volume": safefloat(item.get("volume"), np.nan),
            "LTP": safefloat(item.get("ltp") or item.get("lp"), np.nan),
        })
    if not rows:
        return pd.DataFrame()
    oc = pd.DataFrame(rows)
    step = neareststep(ltp if pd.notna(ltp) else oc["Strike"].median())
    atm = round(ltp / step) * step if pd.notna(ltp) else oc["Strike"].median()
    strikes = sorted(oc["Strike"].dropna().unique(), key=lambda x: abs(x - atm))[:paircount]
    out = []
    for strike in sorted(strikes):
        for opttype in ["CE", "PE"]:
            leg = oc[(oc["Strike"] == strike) & (oc["Option Type"] == opttype)]
            if leg.empty:
                continue
            out.append({
                "Strike": strike,
                "Option Type": opttype,
                "Option Symbol": leg["Option Symbol"].iloc[0],
                "OI": safefloat(leg["OI"].iloc[0], 0),
                "Chain Volume": safefloat(leg["Chain Volume"].iloc[0], 0),
            })
    return pd.DataFrame(out)


def scansingleoption(optionsymbol: str, optiontype: str, strike: float, underlying: str) -> Optional[Dict]:
    histsymbol = optionsymbol if optionsymbol.startswith("NSE:") else f"NSE:{optionsymbol}"
    dailydf = gethistory(histsymbol, "D", DAILYLOOKBACKDAYS)
    intradf = gethistory(histsymbol, "5", INTRADAYLOOKBACKDAYS)
    if dailydf.empty or intradf.empty:
        return None
    summary = summarizeintraday(intradf, dailydf)
    if not summary:
        return None
    summary.update({
        "Underlying": underlying,
        "Option Type": optiontype,
        "Option Symbol": optionsymbol,
        "Strike": strike,
        "OBV": computetodayobv(intradf),
        "OI": np.nan,
        "Volume": intradf["volume"].sum() if "volume" in intradf.columns else 0,
    })
    return summary


def optionliquidityscore(oi, volume, obv) -> float:
    return round(np.log1p(max(safefloat(oi, 0), 0)) * 0.45 + np.log1p(max(safefloat(volume, 0), 0)) * 0.35 + np.log1p(max(abs(safefloat(obv, 0)), 0)) * 0.20, 4)


def rankoptioncandidates(df: pd.DataFrame, side: str) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    out = df.copy()
    out["Liq"] = safeseries(out, "OIVolumeOBV Score", 0)
    out["RD"] = safeseries(out, "Rank Delta", 0)
    out["ADX"] = safeseries(out, "Cumulative ADX", 0)
    out["PCT"] = safeseries(out, "Change", 0)
    otyp = out["Option Type"].astype(str).str.upper() if "Option Type" in out.columns else pd.Series("", index=out.index)
    if side == "long":
        typebonus = np.where(otyp.eq("CE"), 0.30, 0.10)
        out["EMAILRANKSCORE"] = out["Liq"] * 0.40 + out["RD"] * 0.30 + out["ADX"] * 0.18 + out["PCT"] * 0.10 + typebonus
        out = out.sort_values(["EMAILRANKSCORE", "Liq", "RD", "ADX", "PCT"], ascending=[False, False, False, False, False])
    else:
        typebonus = np.where(otyp.eq("PE"), 0.30, 0.10)
        out["EMAILRANKSCORE"] = out["Liq"] * 0.40 + (-out["RD"]) * 0.30 + out["ADX"] * 0.18 + (-out["PCT"]) * 0.10 + typebonus
        out = out.sort_values(["EMAILRANKSCORE", "Liq", "RD", "ADX", "PCT"], ascending=[False, False, True, False, True])
    return out.reset_index(drop=True)


def buildoptioncandidates(candidatesdf: pd.DataFrame, side: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if candidatesdf is None or candidatesdf.empty or "Underlying" not in candidatesdf.columns:
        return pd.DataFrame(), pd.DataFrame()
    rows, iterrows = [], []
    for underlying in candidatesdf["Underlying"].dropna().astype(str):
        pairdf = fetchoptionpairs(underlying)
        if pairdf.empty:
            continue
        atmstrike = pairdf["Strike"].iloc[0]
        reqtype = "CE" if side == "long" else "PE"
        atmrows = pairdf[(pairdf["Strike"] == atmstrike) & (pairdf["Option Type"] == reqtype)]
        atmvol = safefloat(atmrows["Chain Volume"].iloc[0] if not atmrows.empty else 0, 0)
        if atmvol < MINATMCHAINVOLUME:
            logger.debug("SKIP %s ATM %s vol %.0f < %d", underlying, reqtype, atmvol, MINATMCHAINVOLUME)
            continue
        for _, row in pairdf.iterrows():
            strike = safefloat(row.get("Strike"), np.nan)
            opttype = str(row.get("Option Type", "")).upper()
            sym = str(row.get("Option Symbol", "")).strip()
            if not sym or opttype not in ("CE", "PE"):
                continue
            scanned = scansingleoption(sym, opttype, strike, underlying)
            if not scanned:
                continue
            scanned["OIVolumeOBV Score"] = optionliquidityscore(scanned.get("OI", 0), scanned.get("Volume", 0), scanned.get("OBV", 0))
            if safefloat(scanned.get("LTP", 0.0)) < MINOPTIONLTP:
                continue
            if safefloat(scanned.get("Volume", 0)) < MINATMCHAINVOLUME:
                continue
            rows.append(scanned)
            hist = scanned.get("Iteration History")
            if isinstance(hist, pd.DataFrame) and not hist.empty:
                tmp = hist.copy()
                tmp.insert(0, "Option Symbol", sym)
                tmp.insert(1, "Underlying", underlying)
                tmp.insert(2, "Strike", strike)
                tmp.insert(3, "Option Type", opttype)
                iterrows.append(tmp)
    if not rows:
        return pd.DataFrame(), pd.DataFrame()
    out = pd.DataFrame(rows)
    out = rankoptioncandidates(out, side)
    finalcols = [c for c in OPTIONEMAILCOLS if c in out.columns]
    finalout = out[finalcols].reset_index(drop=True)
    iterdf = pd.DataFrame()
    if iterrows and not finalout.empty:
        alliters = pd.concat(iterrows, ignore_index=True)
        alliters = alliters[alliters["Option Symbol"].isin(finalout["Option Symbol"])].copy()
        sortcols = [c for c in ["Underlying", "Option Type", "Strike", "Option Symbol", "iteration"] if c in alliters.columns]
        if sortcols:
            alliters = alliters.sort_values(sortcols).reset_index(drop=True)
        groupcols = [c for c in ["Underlying", "Option Type", "Strike", "Option Symbol"] if c in alliters.columns]
        if groupcols and not alliters.empty:
            alliters["iteration"] = alliters.groupby(groupcols).cumcount() + 1
        if "iteration" in alliters.columns:
            alliters["iteration"] = pd.to_numeric(alliters["iteration"], errors="coerce").astype("Int64")
            alliters = alliters[alliters["iteration"].between(1, ITERATIONSTOKEEP)]
        iterdf = alliters.reset_index(drop=True)
    return finalout, iterdf


def computechainstatus(windowsignals: list) -> Tuple[str, int, int, int]:
    b = sum(1 for s in windowsignals if str(s).startswith("Buy"))
    s = sum(1 for s in windowsignals if str(s).startswith("Sell"))
    t = b + s
    if t == 0:
        return "MIXED", 0, 0, 0
    ratio = b / t
    if ratio >= 0.65:
        status = "CONFIRMED"
    elif ratio <= 0.35:
        status = "BROKEN"
    else:
        status = "MIXED"
    return status, b, s, t


def latestblockchain(iterdf: pd.DataFrame, blockminutes: int) -> Tuple[str, int, int, int]:
    if iterdf is None or iterdf.empty or "iteration" not in iterdf.columns:
        return "MIXED", 0, 0, 0
    itersperblock = max(1, blockminutes // SIGNALWINDOWMINUTES)
    lastit = int(iterdf["iteration"].max())
    blocknum = (lastit - 1) // itersperblock + 1
    bstart = (blocknum - 1) * itersperblock + 1
    bend = blocknum * itersperblock
    blockrows = iterdf[(iterdf["iteration"] >= bstart) & (iterdf["iteration"] <= bend)]
    sigs = blockrows["windowsignal"].tolist() if not blockrows.empty and "windowsignal" in blockrows.columns else []
    return computechainstatus(sigs)


def scanoptionchainsignals(optionsymbol: str, optiontype: str, strike: float, underlying: str, iterdf: pd.DataFrame) -> Optional[Dict]:
    c5, b5, s5, t5 = latestblockchain(iterdf, 5)
    c15, b15, s15, t15 = latestblockchain(iterdf, 15)
    c30, b30, s30, t30 = latestblockchain(iterdf, 30)
    entrysignal = c5 == "CONFIRMED" and c30 == "CONFIRMED" and t30 >= T30MIN
    exitsignal = c30 == "BROKEN" or t30 < T30MIN
    chainsignal = "ENTER" if entrysignal else "EXIT" if exitsignal else "WAIT"
    exitlabel = "X EXIT NOW" if exitsignal else "OK HOLD"
    return {
        "Underlying": underlying,
        "Option Type": optiontype,
        "Strike": strike,
        "5m": c5,
        "T5": t5,
        "15m": c15,
        "T15": t15,
        "30m": c30,
        "T30": t30,
        "Chain Signal": chainsignal,
        "Exit Signal": exitlabel,
    }


def loaddailystate() -> Dict:
    today = datetime.now().strftime("%Y-%m-%d")
    if os.path.exists(DAILYSTATEFILE):
        try:
            with open(DAILYSTATEFILE) as f:
                state = json.load(f)
            if state.get("date") == today:
                return state
        except Exception:
            pass
    return {"date": today, "rows": []}


def savedailystatestate(state: Dict):
    try:
        os.makedirs(os.path.dirname(DAILYSTATEFILE) or ".", exist_ok=True)
        with open(DAILYSTATEFILE, "w") as f:
            json.dump(state, f, default=str)
    except Exception as e:
        logger.warning("Could not save chain state: %s", e)


def updatestickyrows(state: Dict, newrows: List[Dict]) -> List[Dict]:
    existing = state.get("rows", {})
    for row in newrows:
        key = f"{row.get('Underlying')}_{row.get('Option Type')}_{row.get('Strike')}"
        if key not in existing:
            if row.get("Chain Signal") == "ENTER":
                row["Entry Time"] = datetime.now().strftime("%H:%M")
            existing[key] = {k: v for k, v in row.items()}
        else:
            for col in ["LTP", "5m", "T5", "15m", "T15", "30m", "T30", "Chain Signal", "Exit Signal"]:
                if col in row:
                    existing[key][col] = row[col]
    state["rows"] = existing
    savedailystatestate(state)
    return sorted(existing.values(), key=lambda r: r.get("Entry Time", "9999"))


def sendsingleemail(subject: str, htmlbody: str, attachments: list = None) -> bool:
    sender = os.environ.get("SENDEREMAIL")
    password = os.environ.get("SENDERPASSWORD")
    recipientsraw = os.environ.get("RECIPIENTEMAIL", "")
    recipients = [r.strip() for r in recipientsraw.split(",") if r.strip()]
    if not sender or not password or not recipients:
        logger.warning("Email credentials/recipients not configured; skipping.")
        return False
    msg = MIMEMultipart("mixed")
    msg["Subject"] = subject
    msg["From"] = sender
    msg["To"] = ", ".join(recipients)
    msg.attach(MIMEText(htmlbody, "html", "utf-8"))
    for path in attachments or []:
        if not path or not os.path.isfile(path):
            continue
        try:
            with open(path, "rb") as fh:
                part = MIMEBase("application", "octet-stream")
                part.set_payload(fh.read())
                encoders.encode_base64(part)
                part.add_header("Content-Disposition", f'attachment; filename="{os.path.basename(path)}"')
                msg.attach(part)
        except Exception as e:
            logger.warning("Could not attach %s: %s", path, e)
    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(sender, password)
            server.sendmail(sender, recipients, msg.as_string())
        logger.info("Email sent: %s", subject)
        return True
    except Exception as e:
        logger.error("Failed to send email %s: %s", subject, e)
        return False


def prepareoptionemailview(df: pd.DataFrame, side: str, maxrows: int = 25) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    view = applydisplaylabels(df.copy(), side)
    view = view.rename(columns=OPTIONEMAILCOLRENAME)
    return view.head(maxrows).reset_index(drop=True)


def compacttablehtml(df: pd.DataFrame, title: str, maxrows: int) -> str:
    cols = [c for c in ["Underlying", "Option Type", "Strike", "LTP", "Chg", "OI", "Volume", "OBV", "Liq Score", "Rank", "Rank Delta", "Cumulative ADX", "5mSignal", "15mSignal", "OverallSignal", "IVP", "Time"] if c in df.columns]
    display = df[cols].head(maxrows).copy()
    headercells = "".join(f"<th>{c}</th>" for c in cols)
    rowshtml = []
    for _, row in display.iterrows():
        cells = []
        for c in cols:
            val = row[c]
            if pd.isna(val):
                sval = ""
            elif c in ["Chg", "Liq Score", "Rank", "Rank Delta", "Cumulative ADX", "IVP", "LTP", "Strike"]:
                sval = f"{float(val):.2f}"
            elif c in ["OI", "Volume", "OBV"]:
                sval = f"{int(float(val))}"
            else:
                sval = str(val)
            cells.append(f"<td>{sval}</td>")
        rowshtml.append(f"<tr>{''.join(cells)}</tr>")
    body = ''.join(rowshtml) if rowshtml else f'<tr><td colspan="{len(cols)}" style="color:#999;text-align:center">No rows</td></tr>'
    return f"<h3>{title}</h3><table border='1' cellspacing='0' cellpadding='4'><thead><tr>{headercells}</tr></thead><tbody>{body}</tbody></table>"


def buildchainemailhtml(longrows: List[Dict], shortrows: List[Dict]) -> str:
    ts = datetime.now().strftime("%d %b %Y, %H:%M")
    longtable = pd.DataFrame(longrows)
    shorttable = pd.DataFrame(shortrows)
    return f"<html><body><p>Chain Signal Report - {ts}</p>{compacttablehtml(longtable,'LONG CANDIDATES CE',EMAILMAXROWSLONG)}{compacttablehtml(shorttable,'SHORT CANDIDATES PE',EMAILMAXROWSSHORT)}</body></html>"


def buildcepebuyrows(optionsdf: pd.DataFrame, iterationdf: pd.DataFrame) -> Tuple[List[Dict], List[Dict]]:
    if optionsdf is None or optionsdf.empty or iterationdf is None or iterationdf.empty:
        return [], []
    if "Option Symbol" not in iterationdf.columns:
        return [], []
    signalcol = "windowsignal" if "windowsignal" in iterationdf.columns else None
    if signalcol is None:
        return [], []
    itermap = {sym: grp for sym, grp in iterationdf.groupby("Option Symbol")}
    cerows, perows = [], []
    for _, row in optionsdf.iterrows():
        sym = str(row.get("Option Symbol", "")).strip()
        if not sym or sym not in itermap:
            continue
        grp = itermap[sym]
        signals = [str(v).strip().title() for v in grp[signalcol].tolist()]
        last11 = signals[-11:]
        if len(last11) < 11:
            continue
        buycount = sum(s.startswith("Buy") for s in last11)
        sellcount = sum(s.startswith("Sell") for s in last11)
        optype = str(row.get("Option Type", "")).upper()
        tradesignal = "CE BUY" if optype == "CE" and buycount >= 8 else "PE BUY" if optype == "PE" and sellcount >= 8 else ""
        if not tradesignal:
            continue
        out = {
            "Underlying": row.get("Underlying", ""),
            "Option Type": row.get("Option Type", ""),
            "Option Symbol": sym,
            "Strike": row.get("Strike", np.nan),
            "LTP": row.get("LTP", np.nan),
            "Change": row.get("Change", np.nan),
            "Rank Delta": row.get("Rank Delta", np.nan),
            "Last Iteration Time": row.get("Last Iteration Time", ""),
            "Trade Signal": tradesignal,
            "Buy Count": buycount,
            "Sell Count": sellcount,
        }
        if tradesignal == "CE BUY":
            cerows.append(out)
        else:
            perows.append(out)
    return cerows, perows


def sendcepebuyemail(cerows: List[Dict], perows: List[Dict], attachments: list = None) -> bool:
    logger.info("CE BUY %d PE BUY %d", len(cerows), len(perows))
    if not cerows and not perows:
        return False
    subject = f"CE PE Momentum Buy Report - {datetime.now().strftime('%d %b %H:%M')}"
    html = buildcepebuyemailhtml(cerows, perows)
    return sendsingleemail(subject, html, attachments)


def cepetablehtml(rows: List[Dict], title: str) -> str:
    df = pd.DataFrame(rows)
    if df.empty:
        return f"<h3>{title}</h3><p>No momentum buy signals.</p>"
    cols = [c for c in ["Underlying", "Option Type", "Option Symbol", "Strike", "LTP", "Change", "Rank Delta", "Last Iteration Time", "Trade Signal", "Buy Count", "Sell Count"] if c in df.columns]
    header = ''.join(f"<th>{c}</th>" for c in cols)
    bodyrows = []
    for _, row in df[cols].iterrows():
        bodyrows.append('<tr>' + ''.join(f'<td>{row[c]}</td>' for c in cols) + '</tr>')
    return f"<h3>{title}</h3><table border='1' cellspacing='0' cellpadding='4'><thead><tr>{header}</tr></thead><tbody>{''.join(bodyrows)}</tbody></table>"


def buildcepebuyemailhtml(cerows: List[Dict], perows: List[Dict]) -> str:
    ts = datetime.now().strftime("%d %b %Y, %H:%M")
    return f"<html><body><p>CE PE Momentum Buy Report - {ts}</p><p>Rule: 5 same-direction signals to start, 8 of last 11 to confirm.</p>{cepetablehtml(cerows,'CE BUY MOMENTUM SIGNALS')}{cepetablehtml(perows,'PE BUY MOMENTUM SIGNALS')}</body></html>"


def scansymbol(symbol: str) -> Optional[Dict]:
    eq = formateqsymbol(symbol)
    dailydf = gethistory(eq, "D", DAILYLOOKBACKDAYS)
    intradf = gethistory(eq, "5", INTRADAYLOOKBACKDAYS)
    if dailydf.empty or intradf.empty:
        return None
    summary = summarizeintraday(intradf, dailydf)
    if not summary:
        return None
    summary["Symbol"] = symbol
    return summary


def sendchainsignalemail(longrows: List[Dict], shortrows: List[Dict], attachments: list = None) -> bool:
    subject = f"Chain Signal Report - {datetime.now().strftime('%d %b %H:%M')}"
    html = buildchainemailhtml(longrows, shortrows)
    return sendsingleemail(subject, html, attachments)


def main():
    os.makedirs(OUTPUTDIR, exist_ok=True)
    initfyers()
    symbols = loadfnosymbolsfromsectors(SECTORSDIR)
    rows = []
    for i, symbol in enumerate(symbols, start=1):
        logger.info("%s/%s Scanning %s", i, len(symbols), symbol)
        row = scansymbol(symbol)
        if row:
            rows.append(row)
        time.sleep(PERSYMBOLSLEEPSEC)
    summarydf = pd.DataFrame(rows)
    if summarydf.empty:
        raise RuntimeError("No symbols returned usable market data.")
    summarydf = summarydf.sort_values(["Rank Delta", "Change"], ascending=[False, False]).reset_index(drop=True)
    longseeddf, shortseeddf = choosetopcandidates(summarydf, topn=TOPNUNDERLYINGS)
    longdf, longiterdf = buildoptioncandidates(longseeddf, side="long")
    shortdf, shortiterdf = buildoptioncandidates(shortseeddf, side="short")
    iterationdf = pd.concat([longiterdf, shortiterdf], ignore_index=True) if not longiterdf.empty or not shortiterdf.empty else pd.DataFrame()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    summarycsv = os.path.join(OUTPUTDIR, f"fo_summary_{timestamp}.csv")
    longcsv = os.path.join(OUTPUTDIR, f"fo_long_candidates_{timestamp}.csv")
    shortcsv = os.path.join(OUTPUTDIR, f"fo_short_candidates_{timestamp}.csv")
    itercsv = os.path.join(OUTPUTDIR, f"fo_iteration_history_{timestamp}.csv")
    summarydf.to_csv(summarycsv, index=False)
    longdf.to_csv(longcsv, index=False)
    shortdf.to_csv(shortcsv, index=False)
    if iterationdf.empty:
        iterationdf = pd.DataFrame(columns=["iteration", "Underlying", "Option Type", "Strike", "Option Symbol", "timestamp", "windowminutes", "windowstart", "windowend", "currentwindowscore", "previoustradingdaysametimescore", "windowdelta", "windowsignal", "close"])
    iterationdf.to_csv(itercsv, index=False)
    logger.info("LONG %d rows SHORT %d rows Iter %d rows", len(longdf), len(shortdf), len(iterationdf))
    longmerged = []
    shortmerged = []
    if not longiterdf.empty:
        for _, grp in longiterdf.groupby([c for c in ["Underlying", "Option Type", "Strike", "Option Symbol"] if c in longiterdf.columns]):
            keymap = {c: v for c, v in zip([c for c in ["Underlying", "Option Type", "Strike", "Option Symbol"] if c in longiterdf.columns], grp.iloc[0].tolist())}
            sig = scanoptionchainsignals(keymap.get("Option Symbol", ""), keymap.get("Option Type", ""), safefloat(keymap.get("Strike", np.nan)), keymap.get("Underlying", ""), grp)
            if sig:
                longmerged.append(sig)
    if not shortiterdf.empty:
        for _, grp in shortiterdf.groupby([c for c in ["Underlying", "Option Type", "Strike", "Option Symbol"] if c in shortiterdf.columns]):
            keymap = {c: v for c, v in zip([c for c in ["Underlying", "Option Type", "Strike", "Option Symbol"] if c in shortiterdf.columns], grp.iloc[0].tolist())}
            sig = scanoptionchainsignals(keymap.get("Option Symbol", ""), keymap.get("Option Type", ""), safefloat(keymap.get("Strike", np.nan)), keymap.get("Underlying", ""), grp)
            if sig:
                shortmerged.append(sig)
    state = loaddailystate()
    state["rows"] = updatestickyrows(state, longmerged + shortmerged)
    savedailystatestate(state)
    longcsv2 = os.path.join(OUTPUTDIR, f"chain_long_{timestamp}.csv")
    shortcsv2 = os.path.join(OUTPUTDIR, f"chain_short_{timestamp}.csv")
    pd.DataFrame(longmerged).to_csv(longcsv2, index=False)
    pd.DataFrame(shortmerged).to_csv(shortcsv2, index=False)
    sendchainsignalemail(longmerged, shortmerged, attachments=[summarycsv, longcsv, shortcsv, itercsv])
    combinedoptionsdf = pd.concat([longdf, shortdf], ignore_index=True) if not longdf.empty or not shortdf.empty else pd.DataFrame()
    cebuyrows, pebuyrows = buildcepebuyrows(combinedoptionsdf, iterationdf)
    sendcepebuyemail(cebuyrows, pebuyrows, attachments=[longcsv2, shortcsv2])


if __name__ == "__main__":
    main()
