"""Validación walk-forward con el motor de backtest real.

Divide el historial en ventanas rotatorias y ejecuta cada ventana con el
Backtester real (no el simulador simplificado). La EMA macro diaria usa TODA la
historia previa a la ventana (sin look-ahead: shift(1)), que es como opera en
producción.

Uso:
    python scripts/walkforward.py --ema-macro 140 --sl 2.0 --tp 4.0 --window 60 --step 30
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.optimize_macro import run_case
from tokenscan.config import Settings

PAIRS = ["BTC/USDT", "ETH/USDT", "SOL/USDT"]
CACHE = Path("data/cache")


def load(pair: str, tf: str) -> pd.DataFrame:
    fname = f"{pair.split('/')[0]}_{tf}.csv"
    return pd.read_csv(CACHE / fname, parse_dates=["timestamp"], index_col="timestamp")


class WindowMarket:
    """MarketData que expone solo la ventana 4h pero la daily macro COMPLETA previa."""

    def __init__(self, frames: dict, daily: dict):
        self.frames = frames
        self.daily = daily

    def fetch_ohlcv(self, pair: str, timeframe: str = "4h", limit: int = 500):
        return self.frames[pair]

    def macro_daily(self):
        return self.daily


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ema-macro", type=int, default=140)
    parser.add_argument("--sl", type=float, default=2.0)
    parser.add_argument("--tp", type=float, default=4.0)
    parser.add_argument("--fast", type=int, default=12)
    parser.add_argument("--slow", type=int, default=26)
    parser.add_argument("--window", type=int, default=60, help="días por ventana")
    parser.add_argument("--step", type=int, default=30, help="paso entre ventanas")
    parser.add_argument("--min-trades", type=int, default=3)
    parser.add_argument("--min-kaufman-er", type=float, default=0.0)
    args = parser.parse_args()

    data4h = {p: load(p, "4h") for p in PAIRS}
    data1d = {p: load(p, "1d") for p in PAIRS}
    end = max(df.index[-1] for df in data4h.values())
    start = min(df.index[0] for df in data4h.values())

    settings = Settings.load()
    results = []
    win_start = start
    while win_start + pd.Timedelta(days=args.window) <= end:
        win_end = win_start + pd.Timedelta(days=args.window)
        frames = {}
        for p, df in data4h.items():
            d = df[(df.index >= win_start) & (df.index <= win_end)].copy()
            if len(d) > 60:
                d.attrs["pair"] = p
                frames[p] = d
        # Daily macro: toda la historia hasta win_end (no solo la ventana)
        daily = {p: d["close"][d.index <= win_end] for p, d in data1d.items()}

        if not frames:
            break
        market = WindowMarket(frames, daily)
        params = {"fast": args.fast, "slow": args.slow, "ema_macro": args.ema_macro,
                  "sl_mult": args.sl, "tp_mult": args.tp,
                  "min_kaufman_er": args.min_kaufman_er}
        r = run_case(settings, market, params, args.window,
                     end_dt=win_end.to_pydatetime())
        m = r.metrics()
        results.append({"window": f"{win_start.date()}->{win_end.date()}",
                        "ret": m["total_return_pct"], "trades": m["n_trades"],
                        "pf": m["profit_factor"], "win": m["win_rate_pct"],
                        "dd": m["max_drawdown_pct"]})
        print(f"  [{win_start.date()}] ret={m['total_return_pct']:+7.2f}% "
              f"pf={m['profit_factor']:.2f} trades={m['n_trades']:3d} "
              f"wr={m['win_rate_pct']:5.1f}% dd={m['max_drawdown_pct']:.2f}%")
        win_start += pd.Timedelta(days=args.step)

    if not results:
        print("Sin ventanas válidas")
        return

    pos = [r for r in results if r["ret"] > 0]
    with_trades = [r for r in results if r["trades"] >= args.min_trades]
    pos_w = [r for r in with_trades if r["ret"] > 0]
    import numpy as np

    print("\n=== RESUMEN walk-forward (motor real) ===")
    print(f"  ventanas: {len(results)}  positivas: {len(pos)} ({len(pos)/len(results)*100:.0f}%)")
    if with_trades:
        print(f"  con >= {args.min_trades} trades: {len(with_trades)}  positivas: "
              f"{len(pos_w)} ({len(pos_w)/len(with_trades)*100:.0f}%)")
    print(f"  ret mediano: {np.median([r['ret'] for r in results]):+.2f}%  "
          f"PF mediano: {np.median([r['pf'] for r in with_trades]):.2f}  "
          f"trades totales: {sum(r['trades'] for r in results)}")


if __name__ == "__main__":
    main()
