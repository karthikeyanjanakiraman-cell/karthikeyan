#!/usr/bin/env python3
"""
FO_FNO_FYERS_VOL_REL_EMAIL.py

Intraday F&O scanner via Fyers API with email alerts.
Complete standalone file - no external email.py dependency.
SORTS CANDIDATES BY DIRECTIONAL COLUMN.
ALL STATISTICAL SCORES ARE RAW (ORIGINAL FORMULAS).
Now powered by Zero-Lag Dual Engine Row-vs-Row State Matrices.
"""

import os
import re
import sys
import logging
import warnings
from datetime import datetime, timedelta, time
from typing import List, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from fyers_apiv3 import fyersModel

import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders


class UTF8Formatter(logging.Formatter):
    def format(self, record):
        msg = record.getMessage()
        record.msg = msg.encode("ascii", "ignore").decode("ascii")
        return super().format(record)


logger = logging.getLogger()
logger.setLevel(logging.INFO)
if logger.hasHandlers():
    logger.handlers.clear()

ch = logging.StreamHandler(sys.stdout)
ch.setLevel(logging.INFO)
formatter = UTF8Formatter(
    "%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
ch.setFormatter(formatter)
logger.addHandler(ch)

warnings.filterwarnings("ignore")

DAILY_LOOKBACK_DAYS = 60
INTRADAY_LOOKBACK_DAYS = 20
IVP_LOOKBACK_DAYS = 252
INDEX_SOFT_BOOST_WEIGHT = 0.25

fyers: Optional[fyersModel.FyersModel] = None

smtp_host = os.environ.get("SMTP_HOST", "smtp.gmail.com")
smtp_port = int(os.environ.get("SMTP_PORT", "587"))
sender_email = os.environ.get("SENDER_EMAIL", "you@example.com")
sender_password = os.environ.get("SENDER_PASSWORD", "password")
recipient_email = os.environ.get("RECIPIENT_EMAIL", "you@example.com")

EMAIL_DISPLAY_COLS = [
    "Symbol",
    "LTP",
    "% Change",
    "Iteration Change",
    "10 Day Relative Volume",
    "VWAP Z-Score",
    "Turning Regime",
    "Trade Action",
    "Dual Engine State",
    "MTF_ALIGN",
]


def build_signals_from_raw_directional(detail_df) -> dict:
    nan = float("nan")
    out = {
        k: nan for k in (
            "5m_Signal",
            "15m_Signal",
            "30m_Signal",
            "60m_Signal",
            "Bull_Signal",
            "Bear_Signal",
            "Overall_Signal",
        )
    }
    if detail_df is None or detail_df.empty:
        return out

    df = detail_df.copy()
    if "Iteration No" in df.columns:
        df = df.sort_values("Iteration No")

    vals = pd.to_numeric(df["Directional"], errors="coerce").dropna().to_numpy(dtype=float)
    if vals.size == 0:
        return out

    last = vals.size - 1

    def raw_at(offset: int) -> float:
        i = last - offset
        if i < 0:
            i = 0
        return float(vals[i])

    out["5m_Signal"] = round(raw_at(0), 4)
    out["15m_Signal"] = round(raw_at(3) if last >= 3 else raw_at(0), 4)
    out["30m_Signal"] = round(raw_at(6) if last >= 6 else raw_at(0), 4)
    out["60m_Signal"] = round(raw_at(12) if last >= 12 else raw_at(0), 4)
    out["Bull_Signal"] = round(float(vals[vals > 0].max()) if (vals > 0).any() else 0.0, 4)
    out["Bear_Signal"] = round(abs(float(vals[vals < 0].min())) if (vals < 0).any() else 0.0, 4)
    out["Overall_Signal"] = round(raw_at(0), 4)
    return out


def classify_mtf_from_window(win, eps: float = 1e-9) -> float:
    s = pd.Series(win).dropna().astype(float)
    if len(s) < 3:
        return float("nan")
    diff1 = s.diff().dropna()
    if len(diff1) < 2:
        return float("nan")
    x = np.arange(len(s), dtype=float)
    slope = float(np.polyfit(x, s.values, 1)[0])
    net_move = float(s.iloc[-1] - s.iloc[0])
    turning = float(np.mean(np.abs(np.diff(s.values, n=2)))) if len(s) >= 3 else 0.0
    stability = float(np.std(s.values))
    cumsum_plus = float(np.clip(diff1, 0, None).sum())
    score = (slope + net_move) + (0.25 * cumsum_plus) - (0.50 * turning) - (0.10 * stability)
    if score > eps:
        return 1.0
    if score < -eps:
        return -1.0
    return 0.0


def build_mtf_alignment(detail_df: pd.DataFrame) -> Dict[str, object]:
    out = {
        "MTF_5m": float("nan"),
        "MTF_15m": float("nan"),
        "MTF_30m": float("nan"),
        "MTF_60m": float("nan"),
        "MTF_SCORE": float("nan"),
        "MTF_ALIGN": "NA",
    }
    if detail_df is None or detail_df.empty or "Iteration Change" not in detail_df.columns:
        return out
    df = detail_df.copy()
    if "Iteration No" in df.columns:
        df = df.sort_values("Iteration No")
    series = pd.to_numeric(df["Iteration Change"], errors="coerce").dropna().astype(float)
    n = len(series)
    if n < 3:
        return out

    mtf_5 = classify_mtf_from_window(series.tail(min(3, n)))
    mtf_15 = classify_mtf_from_window(series.tail(3)) if n >= 3 else float("nan")
    mtf_30 = classify_mtf_from_window(series.tail(6)) if n >= 6 else float("nan")
    mtf_60 = classify_mtf_from_window(series.tail(12)) if n >= 12 else float("nan")

    available = [v for v in [mtf_5, mtf_15, mtf_30, mtf_60] if pd.notna(v)]
    if not available:
        return out

    score = float(np.nansum([mtf_5, mtf_15, mtf_30, mtf_60]))
    align = "MIXED"
    if all(v == 1.0 for v in available):
        align = "LONG"
    elif all(v == -1.0 for v in available):
        align = "SHORT"
    elif len(available) == 1 and available[0] == 0.0:
        align = "NA"

    out.update({
        "MTF_5m": mtf_5,
        "MTF_15m": mtf_15,
        "MTF_30m": mtf_30,
        "MTF_60m": mtf_60,
        "MTF_SCORE": score,
        "MTF_ALIGN": align,
    })
    return out


def classify_diff_status(cumsum_diff: float, turning_diff: float, prior_cumsum: float = 0.0, eps: float = 1e-4) -> str:
    c = 0.0 if pd.isna(cumsum_diff) else float(cumsum_diff)
    t = 0.0 if pd.isna(turning_diff) else float(turning_diff)
    p = 0.0 if pd.isna(prior_cumsum) else float(prior_cumsum)
    friction_expanding = t > eps
    if c > eps and p <= eps and not friction_expanding:
        return "PRISTINE_BREAKOUT"
    if c > eps and p > eps and not friction_expanding:
        return "ACTIVE_CONTINUATION"
    if c <= eps and friction_expanding:
        return "TRUE_EXHAUSTION"
    if c > eps and friction_expanding:
        return "CHURNING_FAKEOUT"
    if abs(c) <= eps and not friction_expanding:
        return "HEALTHY_PAUSE"
    return "TRANSITION"


def add_dual_engine_matrix(detail_df: pd.DataFrame, eps: float = 1e-4) -> pd.DataFrame:
    if detail_df is None or detail_df.empty:
        return pd.DataFrame() if detail_df is None else detail_df.copy()
    required = {"Symbol", "Iteration No", "Turning", "CumsumPlus"}
    if not required.issubset(detail_df.columns):
        return detail_df.copy()
    out = detail_df.copy()
    out["Turning"] = pd.to_numeric(out["Turning"], errors="coerce").fillna(0.0)
    out["CumsumPlus"] = pd.to_numeric(out["CumsumPlus"], errors="coerce").fillna(0.0)
    out["Iteration No"] = pd.to_numeric(out["Iteration No"], errors="coerce")
    out = out.dropna(subset=["Iteration No"]).sort_values(["Symbol", "Iteration No"]).reset_index(drop=True)
    grouped = out.groupby("Symbol", group_keys=False)
    out["Current_Step"] = grouped["CumsumPlus"].diff().fillna(0.0)
    out["Prior_Step"] = grouped["Current_Step"].shift(1).fillna(0.0)
    out["CumsumDiff"] = out["Current_Step"]
    out["Prior_Turning"] = grouped["Turning"].shift(1).fillna(0.0)
    out["TurningDiff"] = out["Turning"] - out["Prior_Turning"]
    out["Friction_Expanding"] = (out["Turning"] > out["Prior_Turning"]) & (out["Turning"] > eps)
    cond_pristine = (out["Current_Step"] > eps) & (out["Prior_Step"] <= eps) & (~out["Friction_Expanding"])
    cond_exhaustion = (out["Current_Step"] <= eps) & out["Friction_Expanding"]
    cond_trap = (out["Current_Step"] > eps) & out["Friction_Expanding"]
    cond_pause = (out["Current_Step"].abs() <= eps) & (~out["Friction_Expanding"])
    cond_active = (out["Current_Step"] > eps) & (out["Prior_Step"] > eps) & (~out["Friction_Expanding"])
    out["Dual Engine State"] = np.select(
        [cond_pristine, cond_exhaustion, cond_trap, cond_pause, cond_active],
        ["PRISTINE_BREAKOUT", "TRUE_EXHAUSTION", "CHURNING_FAKEOUT", "HEALTHY_PAUSE", "ACTIVE_CONTINUATION"],
        default="TRANSITION",
    )
    out["Trade Action"] = np.select(
        [
            out["Dual Engine State"] == "PRISTINE_BREAKOUT",
            out["Dual Engine State"] == "TRUE_EXHAUSTION",
            out["Dual Engine State"] == "CHURNING_FAKEOUT",
            out["Dual Engine State"] == "HEALTHY_PAUSE",
            out["Dual Engine State"] == "ACTIVE_CONTINUATION",
        ],
        ["ENTRY", "EXIT", "BLOCK_ENTRY", "HOLD", "HOLD"],
        default="WAIT",
    )
    out["Turning Regime"] = np.where(out["Friction_Expanding"], "EXPANDING_FRICTION", "LOW_FRICTION")
    out["Entry Allowed"] = out["Trade Action"].eq("ENTRY")
    out["Hold Allowed"] = out["Trade Action"].eq("HOLD")
    out["Exit Now"] = out["Trade Action"].eq("EXIT")
    out["Diff Status"] = out["Dual Engine State"]
    return out


def merge_dual_engine_latest(summary_df: pd.DataFrame, detail_df: pd.DataFrame) -> pd.DataFrame:
    if summary_df is None or summary_df.empty:
        return pd.DataFrame() if summary_df is None else summary_df.copy()
    if detail_df is None or detail_df.empty:
        return summary_df.copy()
    needed = {
        "Symbol", "Iteration No", "Iteration Change", "CumsumDiff", "TurningDiff",
        "Turning Regime", "Dual Engine State", "Trade Action",
    }
    if not needed.issubset(detail_df.columns):
        return summary_df.copy()
    latest = (
        detail_df.sort_values(["Symbol", "Iteration No"])
        .groupby("Symbol", as_index=False)
        .tail(1)[[
            "Symbol", "Iteration Change", "CumsumDiff", "TurningDiff",
            "Turning Regime", "Dual Engine State", "Trade Action",
        ]]
        .copy()
    )
    return summary_df.merge(latest, on="Symbol", how="left")


def build_candidate_tables(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if df is None or df.empty:
        return (
            pd.DataFrame(columns=EMAIL_DISPLAY_COLS),
            pd.DataFrame(columns=EMAIL_DISPLAY_COLS),
        )

    base = df.copy()

    numeric_cols = [
        "Directional",
        "Turning",
        "Stability",
        "Balanced",
        "CumsumPlus",
        "10 Day Relative Volume",
        "Last5mVolume",
        "OBV30mDelta",
    ]
    for c in numeric_cols:
        if c in base.columns:
            base[c] = pd.to_numeric(base[c], errors="coerce")

    if "Last Iteration Time" in base.columns:
        base["time"] = pd.to_datetime(
            base["Last Iteration Time"],
            format="%H:%M",
            errors="coerce"
        ).dt.time
    else:
        base["time"] = pd.NaT

    if "MTF_ALIGN" not in base.columns:
        base["MTF_ALIGN"] = ""
    base["MTF_ALIGN"] = base["MTF_ALIGN"].astype(str).str.upper().str.strip()

    if "Turning Regime" in base.columns:
        base["Turning Regime"] = (
            base["Turning Regime"].astype(str).str.upper().str.strip()
        )

    if "Price_Leading_Flag" in base.columns:
        base["Price_Leading_Flag"] = (
            base["Price_Leading_Flag"]
            .astype(str)
            .str.upper()
            .str.strip()
            .map({"TRUE": True, "FALSE": False})
            .fillna(False)
        )

    if "Price_Lead_Status" in base.columns:
        base["Price_Lead_Status"] = (
            base["Price_Lead_Status"].astype(str).str.upper().str.strip()
        )

    start_count = len(base)

    if "10 Day Relative Volume" in base.columns:
        pre = len(base)
        base = base[base["10 Day Relative Volume"].fillna(0) >= 1.0]
        logger.info(f"CANDIDATES RVOL gate kept {len(base)}/{pre}")

    if "Last5mVolume" in base.columns:
        pre = len(base)
        base = base[base["Last5mVolume"].fillna(0) > 0]
        logger.info(f"CANDIDATES Last5mVolume gate kept {len(base)}/{pre}")

    gate_time = pd.to_datetime("09:45", format="%H:%M").time()
    bearish_status = {"STRONG_PRICE_LEAD_FADE", "PRICE_LEADING_FADE_RISK"}

    def _apply_long_gate(frame: pd.DataFrame, allow_mixed: bool, relaxed: bool) -> pd.DataFrame:
        out = frame.copy()
        if "time" in out.columns:
            pre = len(out)
            out = out[out["time"] >= gate_time]
            logger.info(f"LONG time gate kept {len(out)}/{pre}")
        if "OBV30mDelta" in out.columns:
            pre = len(out)
            out = out[out["OBV30mDelta"].fillna(0) >= 0] if relaxed else out[out["OBV30mDelta"].fillna(0) > 0]
            logger.info(f"LONG OBV gate kept {len(out)}/{pre}")
        if "Turning Regime" in out.columns:
            pre = len(out)
            out = out[out["Turning Regime"].isin(["LOW_FRICTION", "EXPANDING_FRICTION"])] if relaxed else out[out["Turning Regime"] == "LOW_FRICTION"]
            logger.info(f"LONG friction gate kept {len(out)}/{pre}")
        if "Price_Leading_Flag" in out.columns:
            pre = len(out)
            if not relaxed:
                out = out[out["Price_Leading_Flag"] == True]
            logger.info(f"LONG lead gate kept {len(out)}/{pre}")
        if "Directional" in out.columns:
            pre = len(out)
            out = out[out["Directional"] > 0]
            logger.info(f"LONG directional gate kept {len(out)}/{pre}")
        if "MTF_ALIGN" in out.columns:
            allowed = {"LONG", "MIXED"} if allow_mixed else {"LONG"}
            pre = len(out)
            out = out[out["MTF_ALIGN"].isin(allowed)]
            logger.info(f"LONG MTF gate kept {len(out)}/{pre} using {sorted(allowed)}")
        return out

    def _apply_short_gate(frame: pd.DataFrame, allow_mixed: bool, relaxed: bool) -> pd.DataFrame:
        out = frame.copy()
        if "time" in out.columns:
            pre = len(out)
            out = out[out["time"] >= gate_time]
            logger.info(f"SHORT time gate kept {len(out)}/{pre}")
        if "OBV30mDelta" in out.columns:
            pre = len(out)
            out = out[out["OBV30mDelta"].fillna(0) <= 0] if relaxed else out[out["OBV30mDelta"].fillna(0) < 0]
            logger.info(f"SHORT OBV gate kept {len(out)}/{pre}")
        if "Turning Regime" in out.columns:
            pre = len(out)
            out = out[out["Turning Regime"].isin(["LOW_FRICTION", "EXPANDING_FRICTION"])] if relaxed else out[out["Turning Regime"] == "LOW_FRICTION"]
            logger.info(f"SHORT friction gate kept {len(out)}/{pre}")
        if "Price_Lead_Status" in out.columns:
            pre = len(out)
            if not relaxed:
                out = out[out["Price_Lead_Status"].isin(bearish_status)]
            logger.info(f"SHORT lead-status gate kept {len(out)}/{pre}")
        if "Directional" in out.columns:
            pre = len(out)
            out = out[out["Directional"] < 0]
            logger.info(f"SHORT directional gate kept {len(out)}/{pre}")
        if "MTF_ALIGN" in out.columns:
            allowed = {"SHORT", "MIXED"} if allow_mixed else {"SHORT"}
            pre = len(out)
            out = out[out["MTF_ALIGN"].isin(allowed)]
            logger.info(f"SHORT MTF gate kept {len(out)}/{pre} using {sorted(allowed)}")
        return out

    def _build_side(frame: pd.DataFrame, side: str) -> pd.DataFrame:
        attempts = [
            {"allow_mixed": False, "relaxed": False, "label": "strict"},
            {"allow_mixed": True,  "relaxed": False, "label": "strict+mixed"},
            {"allow_mixed": True,  "relaxed": True,  "label": "relaxed+mixed"},
        ]
        final = pd.DataFrame()
        for attempt in attempts:
            logger.info(f"{side.upper()} attempting mode={attempt['label']}")
            if side == "long":
                candidate = _apply_long_gate(frame, attempt["allow_mixed"], attempt["relaxed"])
                candidate = candidate.sort_values(
                    ["Directional", "Turning", "CumsumPlus", "Stability"],
                    ascending=[False, True, False, False],
                    na_position="last",
                )
            else:
                candidate = _apply_short_gate(frame, attempt["allow_mixed"], attempt["relaxed"])
                candidate = candidate.sort_values(
                    ["Directional", "Turning", "CumsumPlus", "Stability"],
                    ascending=[True, True, True, False],
                    na_position="last",
                )
            candidate = candidate.drop_duplicates(subset=["Symbol"])
            logger.info(f"{side.upper()} mode={attempt['label']} produced {len(candidate)} rows")
            if not candidate.empty:
                final = candidate
                break
        return final.head(15)

    logger.info(f"CANDIDATES starting universe size {start_count}")
    long_df = _build_side(base, "long")
    short_df = _build_side(base, "short")
    cols = [c for c in EMAIL_DISPLAY_COLS if c in base.columns]
    long_df = long_df[cols] if not long_df.empty else pd.DataFrame(columns=EMAIL_DISPLAY_COLS)
    short_df = short_df[cols] if not short_df.empty else pd.DataFrame(columns=EMAIL_DISPLAY_COLS)
    logger.info(f"CANDIDATES final long={len(long_df)} short={len(short_df)}")
    return long_df, short_df
