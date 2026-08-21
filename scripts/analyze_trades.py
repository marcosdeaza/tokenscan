"""Análisis de trades: identifica patrones de pérdida en los 20 trades reales.

Usa el mismo setup que optimize_macro.py (CachedMarket con attrs y macro_daily
real) para que los resultados sean idénticos a los validados.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tokenscan.backtest.engine import Backtester
from tokenscan.config import Settings
from tokenscan.quant.regime import detect_regime
from tokenscan.quant.strategies import get_strategy

PAIRS = ["BTC/USDT", "ETH/USDT", "SOL/USDT"]
CACHE = Path("data/cache")


class CachedMarket:
    """Idéntico al de optimize_macro.py: recorta los frames al horizonte."""

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


def main() -> None:
    days = 250
    cache = Path("data/cache")
    market = CachedMarket("4h", days, cache)
    settings = Settings.load()
    params = {"fast": 12, "slow": 26, "ema_macro": 140, "sl_mult": 2.0, "tp_mult": 4.0}

    s = settings.model_copy(deep=True)
    s.backtest.initial_capital = 500.0
    s.timeframe = "4h"
    s.jupiter.tier = "micro"
    s.risk.atr_sl_multiplier = params["sl_mult"]
    s.risk.atr_tp_multiplier = params["tp_mult"]
    s.risk.trailing_stop_pct = 0.0
    bt = Backtester(s, market)
    strategy = get_strategy("macro_gate", fast=params["fast"], slow=params["slow"],
                            ema_macro=params["ema_macro"])
    strategy.macro_daily = market.macro_daily()
    result = bt.run(strategy, pairs=PAIRS, days=days)

    # Indicadores de régimen para cada par
    ind = {}
    for p in PAIRS:
        df = market.fetch_ohlcv(p, "4h", 500)
        df = strategy.compute_indicators(df)
        ind[p] = df

    print(f"=== {len(result.trades)} trades, {result.total_return*100:+.2f}% ===\n")
    rows = []
    for t in result.trades:
        df = ind[t.pair]
        if t.open_index >= len(df):
            continue
        row = df.iloc[t.open_index]
        regime = detect_regime(df).as_dict()
        rows.append({
            "pair": t.pair,
            "pnl_pct": round(t.pnl_ratio * 100, 2),
            "win": t.pnl_ratio > 0,
            "exit": t.exit_reason,
            "bars": t.close_index - t.open_index,
            "adx": round(row.get("adx", 0), 1),
            "rsi": round(row.get("rsi", 50), 1),
            "atr_pct": round(row.get("atr", 0) / row.get("close", 1) * 100, 2),
            "dist_macro_pct": round((row.get("close", 0) / row.get("ema_macro", 1) - 1) * 100, 2),
            "regime": regime.get("regime", "?"),
            "kaufman_er": round(row.get("kaufman_er", 0), 3),
            "bb_pct_b": round(row.get("bb_pct_b", 0.5), 3),
            "vol_ratio": round(row.get("vol_ratio", 1), 2),
        })

    out = pd.DataFrame(rows)
    print(out.to_string(index=False))

    print(f"\n=== Resumen: {len(rows)} trades ===")
    print(f"  Win: {out['win'].mean()*100:.0f}%  "
          f"Avg: {out['pnl_pct'].mean():+.2f}%  "
          f"Med: {out['pnl_pct'].median():+.2f}%")

    print("\n=== Por régimen ===")
    if "regime" in out.columns:
        g = out.groupby("regime").agg(
            n=("pnl_pct", "size"), win=("win", "mean"),
            avg=("pnl_pct", "mean"), med=("pnl_pct", "median"),
        ).round(2)
        print(g.to_string())

    # Feature rank: qué discrimina mejor entre ganadores y perdedores
    for col in ["adx", "dist_macro_pct", "kaufman_er", "atr_pct", "bb_pct_b", "rsi", "vol_ratio"]:
        winners = out[out["win"]][col]
        losers = out[~out["win"]][col]
        if len(winners) > 1 and len(losers) > 1:
            diff = winners.mean() - losers.mean()
            print(f"  {col:16s}  win_avg={winners.mean():7.2f}  lose_avg={losers.mean():7.2f}  diff={diff:+.2f}")


if __name__ == "__main__":
    main()