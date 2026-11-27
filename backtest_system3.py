#!/usr/bin/env python3
"""
Historical Backtest & Win-Rate Runner for System3.py
Uses System3.py's exact 3-Lock Kinetic + Dual-Band + IV engine across a date range
using your existing UPSTOX_ACCESS_TOKEN environment variable (no emails sent).

Usage:
  python3 backtest_system3.py --mode INDEX_OPTIONS --start-date 2026-11-10 --end-date 2026-11-20
  python3 backtest_system3.py --mode STOCK_FNO --start-date 2026-11-17 --end-date 2026-11-20 --win-r 0.10
  python3 backtest_system3.py --mode CASH_EQUITY --start-date 2026-11-01 --end-date 2026-11-20 --csv-out eq_backtest.csv
"""
import os
import sys
import csv
import time
import argparse
import concurrent.futures
from datetime import datetime, timedelta, time as dtime

# Ensure System3.py in the same directory is importable
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

import System4 as s3


def _parse_date(s):
    return datetime.strptime(s.strip(), "%Y-%m-%d").date()


def _parse_hhmm(s):
    h, m = str(s).strip().split(":")[:2]
    return dtime(int(h), int(m))


def _iter_weekdays(start_dt, end_dt):
    cur = start_dt
    while cur <= end_dt:
        if cur.weekday() < 5:
            yield cur
        cur += timedelta(days=1)


def _bucket_exit_reason(reason, survived=False):
    if survived:
        return "SURVIVED_EOD"
    r = str(reason or "").strip()
    if "Profit Lock" in r:
        return "Profit Lock"
    if "Hard stop" in r:
        return "Hard Stop"
    if "Dual-Band Reversal" in r:
        return "Dual-Band Reversal"
    if "IV Crush" in r:
        return "IV Crush"
    if "Kinetic SL" in r:
        return "Kinetic SL"
    return r[:32] or "Other Exit"


def extract_trades_from_row(trade_date, row, include_open_eod=True):
    """Extract every completed (killed) attempt and optional EOD-surviving anchor from an EOD row."""
    trades = []
    sym = str(row.get("Symbol") or "")
    base_atr = float(row.get("BaseAtr", 0.0) or 0.0)
    if base_atr <= 0:
        return trades

    is_pe = s3._is_pe_symbol(sym)
    is_ce = s3._is_ce_symbol(sym)
    is_opt = is_ce or is_pe
    opt_type = "CE" if is_ce else ("PE" if is_pe else "EQ")
    sector = str(row.get("Sector") or s3.DEFAULT_UNMAPPED_SECTOR)

    def _market_dir(raw_dir):
        if raw_dir == "BULL" and is_pe:
            return "BEAR"
        if raw_dir == "BEAR" and is_pe:
            return "BULL"
        return raw_dir

    kills = list(row.get("Kills") or [])
    kills.sort(key=lambda k: str(k.get("anchor_time", "")))

    attempt_counters = {"BULL": 0, "BEAR": 0}
    prior_r_by_dir = {"BULL": [], "BEAR": []}

    for k in kills:
        raw_dir = str(k.get("dir") or "").upper()
        if raw_dir not in ("BULL", "BEAR"):
            continue
        if s3.OPTIONS_BUYING_ONLY and is_opt and raw_dir != "BULL":
            continue
        ap = float(k.get("anchor_price", 0.0) or 0.0)
        kp = float(k.get("killed_price", 0.0) or 0.0)
        pk = float(k.get("peak_price", 0.0) or 0.0)
        if ap <= 0 or kp <= 0:
            continue

        attempt_counters[raw_dir] += 1
        att_num = attempt_counters[raw_dir]
        r_mult = float(s3._kill_r_multiple(k, base_atr))
        peak_r = ((pk - ap) / base_atr if raw_dir == "BULL" else (ap - pk) / base_atr) if pk > 0 else r_mult
        move_pct = ((kp - ap) / ap) * 100.0 if raw_dir == "BULL" else ((ap - kp) / ap) * 100.0

        prior_rs = prior_r_by_dir[raw_dir]
        if att_num == 1:
            popup_type = "1ST-TIME"
        elif s3._prior_attempts_qualify(prior_rs):
            popup_type = "RE-ANCHOR+"
        else:
            popup_type = "RE-ANCHOR*"
        prior_rs.append(r_mult)

        trades.append({
            "date": trade_date.strftime("%Y-%m-%d"),
            "symbol": sym,
            "sector": sector,
            "opt_type": opt_type,
            "direction": _market_dir(raw_dir),
            "premium_dir": raw_dir,
            "attempt": att_num,
            "popup_type": popup_type,
            "anchor_time": str(k.get("anchor_time") or "-"),
            "anchor_price": round(ap, 2),
            "exit_time": str(k.get("killed_time") or "-"),
            "exit_price": round(kp, 2),
            "status": "CLOSED",
            "exit_bucket": _bucket_exit_reason(k.get("reason"), survived=False),
            "exit_reason": str(k.get("reason") or ""),
            "r_mult": round(r_mult, 3),
            "peak_r": round(peak_r, 3),
            "move_pct": round(move_pct, 2),
        })

    # Surviving EOD anchor
    if include_open_eod and row.get("State") != "[STOPPED]" and float(row.get("AnchorPrice", 0.0) or 0.0) > 0:
        raw_dir = str(row.get("PremiumDir") or row.get("AnchorDir") or "").upper()
        if raw_dir in ("BULL", "BEAR") and not (s3.OPTIONS_BUYING_ONLY and is_opt and raw_dir != "BULL"):
            ap = float(row["AnchorPrice"])
            ltp = float(row.get("LTP", 0.0) or 0.0)
            pk = float(row.get("PeakPrice", 0.0) or 0.0)
            if ltp > 0:
                attempt_counters[raw_dir] += 1
                att_num = attempt_counters[raw_dir]
                r_mult = (ltp - ap) / base_atr if raw_dir == "BULL" else (ap - ltp) / base_atr
                peak_r = ((pk - ap) / base_atr if raw_dir == "BULL" else (ap - pk) / base_atr) if pk > 0 else r_mult
                move_pct = ((ltp - ap) / ap) * 100.0 if raw_dir == "BULL" else ((ap - ltp) / ap) * 100.0

                prior_rs = prior_r_by_dir[raw_dir]
                if att_num == 1:
                    popup_type = "1ST-TIME"
                elif s3._prior_attempts_qualify(prior_rs):
                    popup_type = "RE-ANCHOR+"
                else:
                    popup_type = "RE-ANCHOR*"

                cp_dt = row.get("CheckpointTime")
                exit_t = cp_dt.strftime("%H:%M") if hasattr(cp_dt, "strftime") else "15:30"
                trades.append({
                    "date": trade_date.strftime("%Y-%m-%d"),
                    "symbol": sym,
                    "sector": sector,
                    "opt_type": opt_type,
                    "direction": _market_dir(raw_dir),
                    "premium_dir": raw_dir,
                    "attempt": att_num,
                    "popup_type": popup_type,
                    "anchor_time": str(row.get("ActiveAnchor") or "-").replace("*", ""),
                    "anchor_price": round(ap, 2),
                    "exit_time": exit_t,
                    "exit_price": round(ltp, 2),
                    "status": "SURVIVED_EOD",
                    "exit_bucket": "SURVIVED_EOD",
                    "exit_reason": f"Survived to {exit_t} ({row.get('State', '')})",
                    "r_mult": round(r_mult, 3),
                    "peak_r": round(peak_r, 3),
                    "move_pct": round(move_pct, 2),
                })

    return trades


def compute_stats(trades, win_r=0.10):
    n = len(trades)
    if n == 0:
        return {
            "count": 0, "wins": 0, "losses": 0, "scratches": 0,
            "win_rate": 0.0, "net_r": 0.0, "avg_r": 0.0, "avg_peak_r": 0.0,
            "profit_factor": 0.0, "max_win_r": 0.0, "max_loss_r": 0.0,
        }
    rs = [float(t["r_mult"]) for t in trades]
    peaks = [float(t["peak_r"]) for t in trades]
    wins = [r for r in rs if r >= win_r - 1e-9]
    losses = [r for r in rs if r < 0.0]
    scratches = [r for r in rs if 0.0 <= r < win_r - 1e-9]
    gross_win = sum(r for r in rs if r > 0)
    gross_loss = abs(sum(losses))
    pf = (gross_win / gross_loss) if gross_loss > 1e-9 else (99.99 if gross_win > 0 else 0.0)
    return {
        "count": n,
        "wins": len(wins),
        "losses": len(losses),
        "scratches": len(scratches),
        "win_rate": (len(wins) / n) * 100.0,
        "net_r": sum(rs),
        "avg_r": sum(rs) / n,
        "avg_peak_r": sum(peaks) / n,
        "profit_factor": pf,
        "max_win_r": max(rs),
        "max_loss_r": min(rs),
    }


def _print_stats_table(title, grouped_items, win_r):
    print(f"\n=== {title} ===")
    hdr = f"{'Segment':<22} {'Trades':>6} {'Wins':>5} {'Loss':>5} {'Scr':>4} {'Win%':>7} {'Net R':>8} {'Avg R':>7} {'AvgPkR':>7} {'PF':>6}"
    print(hdr)
    print("-" * len(hdr))
    for label, subset in grouped_items:
        st = compute_stats(subset, win_r=win_r)
        if st["count"] == 0:
            continue
        print(
            f"{label:<22} {st['count']:>6} {st['wins']:>5} {st['losses']:>5} {st['scratches']:>4} "
            f"{st['win_rate']:>6.1f}% {st['net_r']:>+8.2f} {st['avg_r']:>+7.2f} {st['avg_peak_r']:>+7.2f} {st['profit_factor']:>6.2f}"
        )


def print_backtest_report(trades, mode, start_dt, end_dt, trading_days, win_r):
    overall = compute_stats(trades, win_r=win_r)
    print("\n" + "=" * 92)
    print(f"SYSTEM 3.0 HISTORICAL BACKTEST REPORT [{mode}] | {start_dt} to {end_dt} ({len(trading_days)} trading days)")
    print(f"Win Threshold: R >= {win_r:+.2f}R | Dual-Band: +/-{s3.DUAL_BAND_STD_MULT:g} sigma, max stale {s3.DUAL_BAND_TWIN_MAX_STALE_MIN:g}m | ATR: {s3.ATR_BASIS_TF}")
    print("=" * 92)
    if overall["count"] == 0:
        print("No qualifying trades found across the selected date range.")
        return

    print(
        f"Total Trades: {overall['count']} | Wins: {overall['wins']} | Losses: {overall['losses']} | Scratches: {overall['scratches']}\n"
        f"Win Rate:     {overall['win_rate']:.2f}% | Net R: {overall['net_r']:+.2f}R | Avg R: {overall['avg_r']:+.2f}R | "
        f"Avg Peak R: {overall['avg_peak_r']:+.2f}R | Profit Factor: {overall['profit_factor']:.2f}\n"
        f"Best Trade:   {overall['max_win_r']:+.2f}R | Worst Trade: {overall['max_loss_r']:+.2f}R"
    )

    _print_stats_table(
        "BREAKDOWN BY POPUP TYPE",
        [
            ("1ST-TIME", [t for t in trades if t["popup_type"] == "1ST-TIME"]),
            ("RE-ANCHOR+", [t for t in trades if t["popup_type"] == "RE-ANCHOR+"]),
            ("RE-ANCHOR*", [t for t in trades if t["popup_type"] == "RE-ANCHOR*"]),
        ],
        win_r,
    )

    _print_stats_table(
        "BREAKDOWN BY DIRECTION & INSTRUMENT TYPE",
        [
            ("BULL", [t for t in trades if t["direction"] == "BULL"]),
            ("BEAR", [t for t in trades if t["direction"] == "BEAR"]),
            ("CE Options", [t for t in trades if t["opt_type"] == "CE"]),
            ("PE Options", [t for t in trades if t["opt_type"] == "PE"]),
            ("Cash Equity", [t for t in trades if t["opt_type"] == "EQ"]),
        ],
        win_r,
    )

    buckets = sorted({t["exit_bucket"] for t in trades})
    _print_stats_table(
        "BREAKDOWN BY EXIT REASON",
        [(b, [t for t in trades if t["exit_bucket"] == b]) for b in buckets],
        win_r,
    )

    dates = sorted({t["date"] for t in trades})
    _print_stats_table(
        "DAY-BY-DAY SUMMARY",
        [(d, [t for t in trades if t["date"] == d]) for d in dates],
        win_r,
    )


def save_trades_csv(trades, csv_path):
    if not csv_path:
        return
    fieldnames = [
        "date", "symbol", "sector", "opt_type", "direction", "premium_dir",
        "attempt", "popup_type", "anchor_time", "anchor_price", "exit_time",
        "exit_price", "status", "exit_bucket", "exit_reason", "r_mult", "peak_r", "move_pct"
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(trades)
    print(f"\nSaved {len(trades)} trades to: {csv_path}")


def run_backtest(args):
    if not s3.get_upstox_token():
        print(f"[!] Missing {s3.ENV_UPSTOX_TOKEN} / {s3.ENV_FALLBACK_TOKEN} environment variable.")
        return 1

    # Never send emails during a backtest
    s3.SEND_DASHBOARD_EMAIL = False
    s3.SEND_STILL_HOLDING_EMAIL = False
    s3.SAVE_LOCAL_HTML_FILE = False

    if args.dual_band_std is not None:
        s3.DUAL_BAND_STD_MULT = args.dual_band_std
    if args.dual_band_max_stale is not None:
        s3.DUAL_BAND_TWIN_MAX_STALE_MIN = args.dual_band_max_stale
    if args.strikes_up is not None:
        s3.STRIKES_ABOVE_ATM = args.strikes_up
        s3.STOCK_OPT_STRIKES_ABOVE_ATM = args.strikes_up
    if args.strikes_down is not None:
        s3.STRIKES_BELOW_ATM = args.strikes_down
        s3.STOCK_OPT_STRIKES_BELOW_ATM = args.strikes_down
    if args.atr_tf is not None:
        s3.ATR_BASIS_TF = args.atr_tf

    s3._RUN_MODE = args.mode
    today = s3.now_ist().date()
    end_dt = _parse_date(args.end_date) if args.end_date else today
    start_dt = _parse_date(args.start_date) if args.start_date else (end_dt - timedelta(days=7))
    eod_t = _parse_hhmm(args.eod_time)
    needles = [args.track] if args.track else []

    t0 = time.time()
    all_trades = []
    traded_days = []

    for day in _iter_weekdays(start_dt, end_dt):
        if day > today:
            continue
        if not s3._session_traded(day, today):
            print(f"Skipping {day} (market holiday / no data)")
            continue
        if s3.STATS.auth_failed:
            print("[!] Upstox authentication failed (HTTP 401).")
            return 1

        traded_days.append(day)
        cutoff_dt = datetime.combine(day, eod_t)
        use_intraday = (day == today)
        print(f"\n[{day}] Running {args.mode} backtest up to {cutoff_dt:%H:%M} ...")

        universe_raw = s3.get_dynamic_universe(args.mode, cutoff_dt, use_intraday, needles=needles)
        if not universe_raw:
            print(f"   No universe instruments found for {day}.")
            continue

        if args.mode in ("INDEX_OPTIONS", "STOCK_FNO"):
            candidates = universe_raw
            if s3.ENABLE_OPTION_IV_ENGINE and args.mode == "INDEX_OPTIONS":
                for u_idx in s3.INDEX_CONFIG:
                    s3._get_cached_spot_1m(u_idx, day, s3.MIN1_HISTORY_DAYS, use_intraday)
        else:
            prog = s3.Progress(f"prefilter {day}", len(universe_raw))
            lookback_cal_days = s3.BACKTRACE_DAYS * s3.PREFILTER_LOOKBACK_MULT + s3.PREFILTER_LOOKBACK_PAD_DAYS
            with concurrent.futures.ThreadPoolExecutor(max_workers=s3.WORKERS) as ex:
                candidates = [
                    r for r in ex.map(
                        s3._history_worker_daily,
                        [(it, day - timedelta(days=lookback_cal_days), day, prog) for it in universe_raw]
                    ) if r is not None
                ]
            prog.done()

        if not candidates:
            print(f"   No candidates after prefilter for {day}.")
            continue

        checkpoints = [cutoff_dt]
        prog = s3.Progress(f"eval {day}", len(candidates))
        work_units = s3._build_work_units(candidates)
        with concurrent.futures.ThreadPoolExecutor(max_workers=s3.WORKERS) as ex:
            unit_results = list(ex.map(
                s3.process_work_unit,
                [(unit, checkpoints, [], use_intraday, s3.MIN1_HISTORY_DAYS, args.mode, prog) for unit in work_units]
            ))
        prog.done()

        day_trades = []
        for unit_res in unit_results:
            for outcomes, _ in unit_res:
                for _, status, payload in outcomes:
                    if status == "ok" and isinstance(payload, dict):
                        day_trades.extend(extract_trades_from_row(day, payload, include_open_eod=args.include_open_eod))

        st = compute_stats(day_trades, win_r=args.win_r)
        print(f"   -> {day}: {st['count']} trades | Win Rate: {st['win_rate']:.1f}% | Net R: {st['net_r']:+.2f}R")
        all_trades.extend(day_trades)

    print_backtest_report(all_trades, args.mode, start_dt, end_dt, traded_days, args.win_r)
    if args.csv_out and all_trades:
        save_trades_csv(all_trades, args.csv_out)
    print(f"\nBacktest finished in {time.time() - t0:.2f}s ({sum(l.total_calls for l in s3.LIMITERS.values())} API calls).")
    return 0


def parse_args():
    p = argparse.ArgumentParser(description="Historical Win-Rate Backtest Runner for System3.py")
    p.add_argument("--mode", choices=["STOCK_FNO", "CASH_EQUITY", "INDEX_OPTIONS"], default=s3.TRADING_MODE)
    p.add_argument("--start-date", type=str, default=None, help="Start date YYYY-MM-DD (default: 7 days before end-date)")
    p.add_argument("--end-date", type=str, default=None, help="End date YYYY-MM-DD (default: today IST)")
    p.add_argument("--eod-time", type=str, default="15:30", help="Daily cutoff time HH:MM (default: 15:30)")
    p.add_argument("--win-r", type=float, default=s3.PRIOR_ATTEMPT_MIN_POSITIVE_R, help="Minimum R-multiple to count as a win (default: 0.10)")
    p.add_argument("--closed-only", dest="include_open_eod", action="store_false", default=True, help="Exclude trades still open at EOD")
    p.add_argument("--csv-out", type=str, default="backtest_trades.csv", help="CSV output path for trade-by-trade log")
    p.add_argument("--track", type=str, default="", help="Optional symbol filter/pin")
    p.add_argument("--dual-band-std", type=float, default=None, help="Override DUAL_BAND_STD_MULT")
    p.add_argument("--dual-band-max-stale", type=float, default=None, help="Override DUAL_BAND_TWIN_MAX_STALE_MIN")
    p.add_argument("--strikes-up", type=int, default=None, help="Override strikes above ATM")
    p.add_argument("--strikes-down", type=int, default=None, help="Override strikes below ATM")
    p.add_argument("--atr-tf", type=str, default=None, help="Override ATR_BASIS_TF")
    return p.parse_args()


if __name__ == "__main__":
    sys.exit(run_backtest(parse_args()))
