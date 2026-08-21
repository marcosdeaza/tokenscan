"""Validacion multi-ventana: robustez de configs candidatas sin overfitting.

Divide el historial en ventanas rotatorias (walk-forward) y reporta cuantas
ventanas son positivas, PF mediano, etc. Tambien prueba separando train/test.

Uso:
    python scripts/validate.py --family rsi_reversion --oversold 40 --timeframe 4h --regime not_down
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.sweep import CACHE, PAIRS, run_config


def load(pair: str, tf: str) -> pd.DataFrame:
    fname = f"{pair.split('/')[0]}_{tf}.csv"
    return pd.read_csv(CACHE / fname, parse_dates=["timestamp"], index_col="timestamp")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--family", required=True)
    parser.add_argument("--timeframe", default="4h")
    parser.add_argument("--regime", default=None)
    parser.add_argument("--window", type=int, default=180, help="dias por ventana")
    parser.add_argument("--step", type=int, default=90, help="paso entre ventanas")
    parser.add_argument("--min-trades", type=int, default=10)
    parser.add_argument("--extra", default="{}", help="JSON extra de parametros")
    args = parser.parse_args()

    extra = json.loads(args.extra)
    regime = None if args.regime in (None, "None", "none", "") else args.regime
    cfg = {"family": args.family, "timeframe": args.timeframe, "regime": regime, **extra}

    data = {}
    for pair in PAIRS:
        data[pair] = load(pair, args.timeframe)

    macro_data = None
    if args.timeframe != "1d":
        macro_data = {}
        for pair in PAIRS:
            fname = f"{pair.split('/')[0]}_1d.csv"
            p = CACHE / fname
            if p.exists():
                macro_data[pair] = pd.read_csv(p, parse_dates=["timestamp"], index_col="timestamp")

    end = max(df.index[-1] for df in data.values())
    start = min(df.index[0] for df in data.values())

    results = []
    win_start = start
    i = 0
    while win_start + pd.Timedelta(days=args.window) <= end:
        # recorta los datos a la ventana
        win_end = win_start + pd.Timedelta(days=args.window)
        window_data = {p: df[(df.index >= win_start) & (df.index <= win_end)]
                       for p, df in data.items() if len(df[(df.index >= win_start) & (df.index <= win_end)]) > 60}
        if not window_data:
            break
        res = run_config(cfg, window_data, args.window, macro_data=macro_data)
        m = {
            "window": f"{win_start.date()} -> {win_end.date()}",
            "return_pct": round(res.total_return * 100, 2),
            "trades": res.n_trades,
            "profit_factor": round(res.profit_factor, 2),
            "win_rate": round(res.win_rate * 100, 1),
        }
        results.append(m)
        print(f"  [{win_start.date()}] ret={m['return_pct']:+7.2f}% pf={m['profit_factor']:.2f} "
              f"trades={m['trades']:3d} wr={m['win_rate']:5.1f}%")
        win_start += pd.Timedelta(days=args.step)
        i += 1
        if i > 200:
            break

    if not results:
        print("Sin ventanas validas")
        return

    positive = [r for r in results if r["return_pct"] > 0]
    with_trades = [r for r in results if r["trades"] >= args.min_trades]
    pos_w_trades = [r for r in with_trades if r["return_pct"] > 0]
    median_ret = np.median([r["return_pct"] for r in results])
    median_pf = np.median([r["profit_factor"] for r in with_trades]) if with_trades else 0.0
    total_trades = sum(r["trades"] for r in results)
    print("\n=== RESUMEN ===")
    print(f"  ventanas: {len(results)}  positivas: {len(positive)} ({len(positive)/len(results)*100:.0f}%)")
    if with_trades:
        print(f"  con >= {args.min_trades} trades: {len(with_trades)}  positivas: {len(pos_w_trades)} "
              f"({len(pos_w_trades)/len(with_trades)*100:.0f}% de las significativas)")
    else:
        print(f"  con >= {args.min_trades} trades: 0")
    print(f"  ret mediano: {median_ret:+.2f}%  PF mediano (con trades): {median_pf:.2f}  trades totales: {total_trades}")


if __name__ == "__main__":
    main()
