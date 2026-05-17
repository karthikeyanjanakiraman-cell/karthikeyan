import os
import logging
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from OPTIONS_OI import (
    initfyers,
    gethistory,
    builditerationhistory,
    SIGNALWINDOWMINUTES,
    ITERATIONSTOKEEP,
    INTRADAYLOOKBACKDAYS,
    formateqsymbol,
    safefloat,
    loaddailystate,
    savedailystatestate,
    sendcepebuyemail,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR = Path(os.environ.get("DATA_DIR", "."))
OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", "."))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
ASIT_GLOB = os.environ.get("ASIT_GLOB", "asit*.csv")
TOP_N = int(os.environ.get("TOP_N", "20"))
CHAIN_UP = int(os.environ.get("CHAIN_UP", "5"))
CHAIN_DOWN = int(os.environ.get("CHAIN_DOWN", "5"))
MIN_OPTION_LTP = float(os.environ.get("MIN_OPTION_LTP", "10"))
MIN_ATM_CHAIN_VOLUME = int(os.environ.get("MIN_ATM_CHAIN_VOLUME", "100000"))


def pick_latest_file(pattern: str, base_dir: Path = BASE_DIR) -> Optional[Path]:
    files = sorted(base_dir.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    return files[0] if files else None


def load_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [str(c).strip() for c in df.columns]
    return df


def find_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    cols = {str(c).strip().lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in cols:
            return cols[c.lower()]
    return None


def clean_text(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.upper().replace({"NAN": np.nan, "NONE": np.nan, "NULL": np.nan, "": np.nan})


def score_frame(df: pd.DataFrame):
    bull_col = find_col(df, ["BullMultiTFScore", "bull", "bullish", "buy", "long", "BullScore", "bull_score", "5min_BullScore", "15min_BullScore", "1hour_BullScore", "4hour_BullScore", "1day_BullScore"])
    bear_col = find_col(df, ["BearMultiTFScore", "bear", "bearish", "sell", "short", "BearScore", "bear_score", "5min_BearScore", "15min_BearScore", "1hour_BearScore", "4hour_BearScore", "1day_BearScore"])
    rank_col = find_col(df, ["RankScore15Tier", "score", "rank", "Score15Tier", "5min_Score15Tier", "15min_Score15Tier", "1hour_Score15Tier", "4hour_Score15Tier", "1day_Score15Tier"])
    sym_col = find_col(df, ["Symbol", "symbol", "underlying", "ticker", "tradingsymbol", "name"])
    if sym_col is None:
        raise ValueError("No symbol column found in ASIT CSV")
    for c in [bull_col, bear_col, rank_col]:
        if c is not None:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df[sym_col] = clean_text(df[sym_col])
    return df, sym_col, bull_col, bear_col, rank_col


def top_rows(df: pd.DataFrame, sort_col: Optional[str], n: int) -> pd.DataFrame:
    out = df.copy()
    if sort_col is not None:
        out = out.sort_values(sort_col, ascending=False, na_position="last")
    return out.head(n).reset_index(drop=True)


def build_top_lists(asit_path: Path, top_n: int = TOP_N) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = load_csv(asit_path)
    df, sym_col, bull_col, bear_col, rank_col = score_frame(df)
    bull_df = top_rows(df, bull_col or rank_col, top_n).copy()
    bear_df = top_rows(df, bear_col or rank_col, top_n).copy()
    bull_df["Side"] = "BULLISH"
    bear_df["Side"] = "BEARISH"
    bull_df["Underlying"] = bull_df[sym_col]
    bear_df["Underlying"] = bear_df[sym_col]
    return bull_df, bear_df, pd.concat([bull_df, bear_df], ignore_index=True)


def fetch_option_chain(underlying: str) -> pd.DataFrame:
    fy = initfyers()
    if fy is None:
        return pd.DataFrame()
    eqsymbol = formateqsymbol(underlying)
    try:
        res = fy.optionchain({"symbol": eqsymbol, "strikecount": max(CHAIN_UP, CHAIN_DOWN) * 2 + 10})
    except Exception as e:
        logger.warning("optionchain failed for %s: %s", underlying, e)
        return pd.DataFrame()
    chain = (res or {}).get("data") or (res or {}).get("optionsChain") or []
    rows = []
    for item in chain:
        strike = safefloat(item.get("strikePrice") or item.get("strike"), np.nan)
        typ = str(item.get("optionType") or item.get("type") or "").upper()
        sym = str(item.get("symbol") or item.get("tradingSymbol") or item.get("optionSymbol") or "").strip()
        if pd.isna(strike) or typ not in ("CE", "PE"):
            continue
        rows.append({
            "Underlying": underlying,
            "Option Type": typ,
            "Strike": strike,
            "Option Symbol": sym,
            "OptionLTP": safefloat(item.get("ltp") or item.get("lp"), np.nan),
            "OI": safefloat(item.get("oi") or item.get("openInterest"), np.nan),
            "Volume": safefloat(item.get("volume"), np.nan),
        })
    return pd.DataFrame(rows)


def infer_atm_from_spot(underlying: str, chain: pd.DataFrame) -> float:
    if chain.empty:
        return np.nan
    fy = initfyers()
    spot = np.nan
    if fy is not None:
        try:
            q = fy.quotes({"symbols": [formateqsymbol(underlying)]})
            data = (q or {}).get("d") or (q or {}).get("data") or []
            if data:
                spot = safefloat(data[0].get("v", {}).get("lp") or data[0].get("lp"), np.nan)
        except Exception:
            spot = np.nan
    strikes = pd.to_numeric(chain["Strike"], errors="coerce").dropna().unique()
    if len(strikes) == 0:
        return np.nan
    strikes = sorted(map(float, strikes))
    if pd.isna(spot):
        return float(strikes[len(strikes) // 2])
    return float(min(strikes, key=lambda x: abs(x - spot)))


def chain_window(chain: pd.DataFrame, atm: float, up: int = CHAIN_UP, down: int = CHAIN_DOWN) -> pd.DataFrame:
    if chain.empty or pd.isna(atm):
        return pd.DataFrame()
    chain = chain.copy()
    chain["Strike"] = pd.to_numeric(chain["Strike"], errors="coerce")
    chain = chain.dropna(subset=["Strike"])
    strikes = sorted(chain["Strike"].unique())
    if not strikes:
        return pd.DataFrame()
    nearest = min(strikes, key=lambda x: abs(x - atm))
    idx = strikes.index(nearest)
    lo = max(0, idx - down)
    hi = min(len(strikes) - 1, idx + up)
    keep = strikes[lo:hi + 1]
    return chain[chain["Strike"].isin(keep)].sort_values(["Strike", "Option Type"]).reset_index(drop=True)


def build_chain_rows(seed_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    iter_rows = []
    if seed_df is None or seed_df.empty:
        return pd.DataFrame(), pd.DataFrame()
    for _, row in seed_df.iterrows():
        underlying = str(row.get("Underlying", "")).strip().upper()
        side = str(row.get("Side", "")).upper()
        chain = fetch_option_chain(underlying)
        if chain.empty:
            continue
        atm = infer_atm_from_spot(underlying, chain)
        win = chain_window(chain, atm, CHAIN_UP, CHAIN_DOWN)
        if win.empty:
            continue
        req_type = "CE" if side.startswith("BULL") else "PE"
        leg = win[win["Option Type"] == req_type].copy()
        if leg.empty:
            continue
        leg = leg.sort_values(["Strike", "OptionLTP"], ascending=[True, False])
        atm_leg = leg.iloc[(leg["Strike"] - atm).abs().argsort().iloc[0]] if not leg.empty else None
        if atm_leg is None:
            continue
        ltp = safefloat(atm_leg.get("OptionLTP"), np.nan)
        oi = safefloat(atm_leg.get("OI"), np.nan)
        vol = safefloat(atm_leg.get("Volume"), np.nan)
        if pd.isna(ltp) or ltp < MIN_OPTION_LTP:
            continue
        if pd.notna(vol) and vol < MIN_ATM_CHAIN_VOLUME:
            continue
        hist = builditerationhistory(gethistory(atm_leg.get("Option Symbol"), "5", INTRADAYLOOKBACKDAYS), SIGNALWINDOWMINUTES, ITERATIONSTOKEEP) if gethistory is not None and builditerationhistory is not None else pd.DataFrame()
        rows.append({
            "Underlying": underlying,
            "Option Type": req_type,
            "Option Symbol": atm_leg.get("Option Symbol", ""),
            "Strike": safefloat(atm_leg.get("Strike"), np.nan),
            "LTP": ltp,
            "OI": oi,
            "Volume": vol,
            "ATMStrike": atm,
        })
        if hist is not None and not hist.empty:
            tmp = hist.copy()
            tmp.insert(0, "Option Symbol", atm_leg.get("Option Symbol", ""))
            tmp.insert(0, "Underlying", underlying)
            iter_rows.append(tmp)
    return pd.DataFrame(rows), (pd.concat(iter_rows, ignore_index=True) if iter_rows else pd.DataFrame())


def build_ce_pe_buy_rows(options_df: pd.DataFrame, iteration_df: pd.DataFrame) -> Tuple[List[dict], List[dict]]:
    if options_df is None or options_df.empty or iteration_df is None or iteration_df.empty:
        return [], []
    if "Option Symbol" not in iteration_df.columns:
        return [], []
    signal_col = "windowsignal" if "windowsignal" in iteration_df.columns else "WindowSignal" if "WindowSignal" in iteration_df.columns else None
    if signal_col is None:
        return [], []
    itermap = {sym: grp for sym, grp in iteration_df.groupby("Option Symbol")}
    cerows, perows = [], []
    for _, row in options_df.iterrows():
        sym = str(row.get("Option Symbol", "")).strip()
        if not sym or sym not in itermap:
            continue
        grp = itermap[sym]
        signals = [str(v).strip().title() for v in grp[signal_col].tolist()]
        last11 = signals[-11:]
        if len(last11) < 11:
            continue
        buycount = sum(s.startswith("Buy") for s in last11)
        sellcount = sum(s.startswith("Sell") for s in last11)
        optype = str(row.get("Option Type", "")).upper()
        trade_signal = "CE BUY" if optype == "CE" and buycount >= 8 else "PE BUY" if optype == "PE" and sellcount >= 8 else ""
        if not trade_signal:
            continue
        out = {
            "Underlying": row.get("Underlying", ""),
            "Option Type": row.get("Option Type", ""),
            "Option Symbol": sym,
            "Strike": row.get("Strike", np.nan),
            "LTP": row.get("LTP", np.nan),
            "OI": row.get("OI", np.nan),
            "Volume": row.get("Volume", np.nan),
            "Trade Signal": trade_signal,
            "Buy Count": buycount,
            "Sell Count": sellcount,
            "Rank Delta": np.nan,
            "Last Iteration Time": grp.iloc[-1].get("timestamp", ""),
        }
        if trade_signal == "CE BUY":
            cerows.append(out)
        else:
            perows.append(out)
    return cerows, perows


def main():
    asit_file = pick_latest_file(ASIT_GLOB)
    if asit_file is None:
        raise FileNotFoundError(f"No file found for pattern {ASIT_GLOB} in {BASE_DIR.resolve()}")
    logger.info("Using ASIT file: %s", asit_file.resolve())
    bull_df, bear_df, combined = build_top_lists(asit_file, top_n=TOP_N)
    options_df, iteration_df = build_chain_rows(combined)
    cebuyrows, pebuyrows = build_ce_pe_buy_rows(options_df, iteration_df)
    state = loaddailystate()
    state["rows"] = cebuyrows + pebuyrows
    savedailystatestate(state)
    ts = pd.Timestamp.now().strftime("%Y%m%d_%H%M")
    options_df.to_csv(OUTPUT_DIR / f"cepebuy_options_{ts}.csv", index=False)
    iteration_df.to_csv(OUTPUT_DIR / f"cepebuy_iterations_{ts}.csv", index=False)
    pd.DataFrame(cebuyrows).to_csv(OUTPUT_DIR / f"ce_buy_{ts}.csv", index=False)
    pd.DataFrame(pebuyrows).to_csv(OUTPUT_DIR / f"pe_buy_{ts}.csv", index=False)
    if cebuyrows or pebuyrows:
        sendcepebuyemail(cebuyrows, pebuyrows, [str(OUTPUT_DIR / f"ce_buy_{ts}.csv"), str(OUTPUT_DIR / f"pe_buy_{ts}.csv")])
    print(f"ASIT file: {asit_file.name}")
    print(f"Top bullish: {len(bull_df)}")
    print(f"Top bearish: {len(bear_df)}")
    print(f"Option rows: {len(options_df)}")
    print(f"Iteration rows: {len(iteration_df)}")
    print(f"CE BUY rows: {len(cebuyrows)}")
    print(f"PE BUY rows: {len(pebuyrows)}")


if __name__ == "__main__":
    main()
