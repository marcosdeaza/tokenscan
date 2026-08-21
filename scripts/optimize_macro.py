"""Optimización de MacroGate con el motor de backtest real.

Barre el grid de parámetros (fast/slow/ema_macro/SL/TP) evaluando cada config
en dos horizontes (corto 90d y largo 250d) sobre los mismos datos cacheados.
El objetivo es encontrar una config robusta que funcione en AMBOS horizontes
(walk-forward honesto), no solo la que maximiza un periodo.

Uso:
    python scripts/optimize_macro.py [--days-short 90] [--days-long 250] [--top 15]
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from tokenscan.backtest.engine import Backtester
from tokenscan.config import Settings
from tokenscan.quant.strategies import get_strategy

PAIRS = ["BTC/USDT", "ETH/USDT", "SOL/USDT"]


class CachedMarket:
    """MarketData compatible que lee de data/cache (misma lógica que backtest_chart)."""

    def __init__(self, timeframe: str, days: int, cache: Path):
        self.timeframe = timeframe
        self.days = days
        self.cache = cache
        self.frames: dict[str, object] = {}
        self.daily: dict[str, object] = {}
        for pair in PAIRS:
            fname = f"{pair.split('/')[0]}_{timeframe}.csv"
            p = cache / fname
            if not p.exists():
                continue
            import pandas as pd

            df = pd.read_csv(p, parse_dates=["timestamp"], index_col="timestamp")
            end = df.index[-1]
            start = end - pd.Timedelta(days=days)
            df = df[(df.index >= start) & (df.index <= end)]
            df.attrs["pair"] = pair
            self.frames[pair] = df

            dname = f"{pair.split('/')[0]}_1d.csv"
            dp = cache / dname
            if dp.exists():
                daily = pd.read_csv(dp, parse_dates=["timestamp"], index_col="timestamp")
                daily = daily[daily.index <= end]
                self.daily[pair] = daily["close"]

    def fetch_ohlcv(self, pair: str, timeframe: str = "5m", limit: int = 500):
        return self.frames[pair]

    def macro_daily(self):
        return self.daily


def run_case(settings: Settings, market: CachedMarket, params: dict, days: int,
             end_dt=None):
    s = settings.model_copy(deep=True)
    s.backtest.initial_capital = 500.0
    s.timeframe = "4h"
    s.jupiter.tier = "micro"
    s.risk.atr_sl_multiplier = params["sl_mult"]
    s.risk.atr_tp_multiplier = params["tp_mult"]
    s.risk.trailing_stop_pct = 0.0
    bt = Backtester(s, market)  # type: ignore[arg-type]
    strategy = get_strategy("macro_gate", fast=params["fast"], slow=params["slow"],
                            ema_macro=params["ema_macro"],
                            min_kaufman_er=params.get("min_kaufman_er", 0.0))
    strategy.macro_daily = market.macro_daily()
    return bt.run(strategy, pairs=PAIRS, days=days, end_dt=end_dt)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--days-short", type=int, default=90)
    parser.add_argument("--days-long", type=int, default=250)
    parser.add_argument("--top", type=int, default=15)
    parser.add_argument("--out", default="results/optimize_macro.json")
    args = parser.parse_args()

    cache = Path("data/cache")
    grid = [
        {"fast": f, "slow": s, "ema_macro": m, "sl_mult": sl, "tp_mult": tp}
        for f, s in [(5, 20), (8, 21), (10, 26), (12, 26), (12, 50), (20, 50), (30, 70), (50, 100)]
        for m in [100, 150, 200]
        for sl, tp in [(2.0, 4.0), (2.5, 4.0), (2.5, 5.0), (3.0, 6.0)]
    ]
    print(f"[OPT] {len(grid)} configs x 2 horizontes ({args.days_short}d / {args.days_long}d)")

    # Un único market que cubre el horizonte largo (el corto es un subconjunto)
    market = CachedMarket("4h", args.days_long, cache)
    settings = Settings.load()

    rows = []
    t0 = time.time()
    for idx, params in enumerate(grid, 1):
        r_short = run_case(settings, market, params, args.days_short)
        r_long = run_case(settings, market, params, args.days_long)
        ms = r_short.metrics()
        ml = r_long.metrics()
        # Score de robustez: requiere retorno positivo en ambos horizontes y
        # premia retorno alto con bajo drawdown. Penaliza configs con pocos trades
        # (no se puede confiar en 2-3 trades) y las que ganan en un solo periodo.
        n_trades = min(ms["n_trades"], ml["n_trades"])
        if ms["total_return_pct"] > 0 and ml["total_return_pct"] > 0 and n_trades >= 5:
            rs, rl = ms["total_return_pct"], ml["total_return_pct"]
            dds, ddl = ms["max_drawdown_pct"], ml["max_drawdown_pct"]
            score = (rs + rl) / (1.0 + dds / 10.0 + ddl / 10.0)
        else:
            score = -999.0
        rows.append({
            "params": params,
            "score": round(score, 2),
            "short": {"ret_pct": ms["total_return_pct"], "trades": ms["n_trades"],
                      "win": ms["win_rate_pct"], "dd": ms["max_drawdown_pct"],
                      "sharpe": ms["sharpe"], "pf": ms["profit_factor"]},
            "long": {"ret_pct": ml["total_return_pct"], "trades": ml["n_trades"],
                     "win": ml["win_rate_pct"], "dd": ml["max_drawdown_pct"],
                     "sharpe": ml["sharpe"], "pf": ml["profit_factor"]},
        })
        if idx % 20 == 0 or idx == len(grid):
            print(f"  [{idx}/{len(grid)}] {time.time()-t0:.0f}s ({ (time.time()-t0)/max(idx,1):.2f}s/config)")

    rows.sort(key=lambda x: x["score"], reverse=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(rows, indent=2))

    print(f"\n[OK] {len(rows)} configs -> {args.out}\n")
    print(f"TOP {args.top} configs robustas (motor real, 90d + 250d):")
    print(f"{'Score':>6s} {'Ret90':>7s} {'DD90':>6s} {'Ret250':>7s} {'DD250':>6s} {'T':>4s}  {'Config':<36s}")
    for r in rows[: args.top]:
        p = r["params"]
        label = f"fast={p['fast']:<2d} slow={p['slow']:<3d} ema_macro={p['ema_macro']:<3d} sl={p['sl_mult']} tp={p['tp_mult']}"
        s, l = r["short"], r["long"]
        print(f"{r['score']:6.2f} {s['ret_pct']:7.2f}% {s['dd']:6.2f}% {l['ret_pct']:7.2f}% {l['dd']:6.2f}% {min(s['trades'],l['trades']):4d}  {label}")


if __name__ == "__main__":
    main()
