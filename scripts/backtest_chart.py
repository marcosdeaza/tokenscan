"""Backtest de la estrategia con varios capitales y generacion de grafica PNG.

Uso:
    python scripts/backtest_chart.py [--days 90] [--out results/backtest.png]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

from tokenscan.backtest.engine import Backtester
from tokenscan.config import Settings
from tokenscan.data.market import MarketData
from tokenscan.execution.exchange import ExchangeClient
from tokenscan.quant.strategies import get_strategy

PALETTE = {"5": "#f97316", "50": "#06b6d4", "500": "#6366f1"}


def run_case(settings: Settings, capital: float, market: MarketData, days: int):
    s = settings.model_copy(deep=True)
    s.backtest.initial_capital = capital
    s.timeframe = "1h"
    s.risk.atr_sl_multiplier = 2.5
    s.risk.atr_tp_multiplier = 3.5
    bt = Backtester(s, market)
    strategy = get_strategy("trend_following", fast=10, slow=30)
    return bt.run(strategy, days=days), strategy


def make_chart(results: dict[str, object], days: int, out: Path) -> None:
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 9), gridspec_kw={"height_ratios": [2, 1]})
    fig.suptitle(
        f"TokenScan — Backtest {days} días (trend_following 10/30, 1h, BTC/ETH/SOL)",
        fontsize=13, fontweight="bold",
    )

    table_rows = []
    for label, result in results.items():
        m = result.metrics()
        color = PALETTE[label]
        df_curve = pd.DataFrame({"equity": result.equity_curve})
        x = pd.date_range(end=pd.Timestamp.utcnow().floor("min"), periods=len(df_curve), freq="1h")
        ax1.plot(x, df_curve["equity"], label=f"{label}€ → {m['final_equity']:.2f}€", color=color, linewidth=1.8)
        final = m["final_equity"]
        delta = final - float(label)
        pct = (delta / float(label)) * 100
        table_rows.append([
            f"{label}€", f"{m['n_trades']}", f"{m['win_rate_pct']}%",
            f"{m['max_drawdown_pct']}%", f"{delta:+.2f}€ ({pct:+.1f}%)", f"{m['profit_factor']}",
        ])

    ax1.axhline(float("5"), color="#9ca3af", ls="--", lw=0.8, alpha=0.5)
    ax1.set_ylabel("Equity (€)")
    ax1.legend(loc="upper left", framealpha=0.9)
    ax1.grid(alpha=0.3)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%d-%b"))

    ax2.axis("off")
    tbl = ax2.table(
        cellText=table_rows,
        colLabels=["Capital", "Trades", "Win rate", "Max DD", "Resultado", "Profit factor"],
        loc="center", cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1, 1.6)
    for (row, col), cell in tbl.get_celld().items():
        if row == 0:
            cell.set_facecolor("#1f2937")
            cell.set_text_props(color="white", fontweight="bold")
        elif col == 4 and "negative" in cell.get_text().get_text():
            cell.set_text_props(color="#dc2626", fontweight="bold")

    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[OK] Grafica guardada: {out}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=90)
    parser.add_argument("--out", default="results/backtest.png")
    args = parser.parse_args()

    settings = Settings.load()
    market = MarketData(ExchangeClient(settings))
    strategy = get_strategy(
        settings.strategy.name,
        rsi_period=settings.strategy.rsi_period,
        oversold=settings.strategy.rsi_oversold,
        overbought=settings.strategy.rsi_overbought,
    )

    results: dict[str, object] = {}
    for capital in (5, 50, 500):
        print(f"[BT] Backtest con {capital}€ ({args.days} días)...")
        result, _s = run_case(settings, capital, market, args.days)
        results[str(capital)] = result

    make_chart(results, args.days, Path(args.out))
    for label, result in results.items():
        m = result.metrics()
        print(f"  {label}€ -> {m['final_equity']:.2f}€ ({m['total_return_pct']:+.1f}%) "
              f"trades={m['n_trades']} win={m['win_rate_pct']}% dd={m['max_drawdown_pct']}% pf={m['profit_factor']}")


if __name__ == "__main__":
    main()
