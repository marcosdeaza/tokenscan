"""Motor de backtest: bucle vela a vela, fills con fees/slippage, métricas.

Reimplementación original del patrón clásico de los motores de backtest
(loop de candles -> señales -> ejecución -> PnL -> métricas de rendimiento).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

import pandas as pd

from ..config import Settings
from ..data.market import MarketData
from ..quant.risk import position_size_atr, stop_price_atr, take_profit_price_atr
from ..quant.strategies import BaseStrategy
from ..utils.logger import setup_logger
from ..utils.pnl import (
    cagr,
    calmar_ratio,
    expectancy,
    max_drawdown,
    profit_factor,
    sharpe_ratio,
    sortino_ratio,
    sqn,
    win_loss_streaks,
    win_rate,
)

log = setup_logger("tokenscan.backtest")


@dataclass
class BacktestTrade:
    pair: str
    side: str
    open_price: float
    close_price: float
    amount: float
    open_index: int
    close_index: int
    pnl_abs: float
    pnl_ratio: float
    exit_reason: str


@dataclass
class BacktestResult:
    initial_capital: float
    final_equity: float
    total_return: float
    trades: list[BacktestTrade] = field(default_factory=list)
    equity_curve: list[float] = field(default_factory=list)
    days: float = 0.0

    @property
    def returns(self) -> list[float]:
        out = []
        for i in range(1, len(self.equity_curve)):
            prev = self.equity_curve[i - 1]
            if prev > 0:
                out.append(self.equity_curve[i] / prev - 1)
        return out

    def metrics(self) -> dict:
        r = self.returns
        _, dd_rel = max_drawdown(self.equity_curve)
        pnls = [t.pnl_abs for t in self.trades]
        cagr_value = cagr(self.initial_capital, self.final_equity, self.days)
        streaks = win_loss_streaks(self.trades)
        return {
            "total_return_pct": self.total_return * 100,
            "final_equity": round(self.final_equity, 2),
            "n_trades": len(self.trades),
            "win_rate_pct": round(win_rate(self.trades) * 100, 1),
            "profit_factor": round(profit_factor(self.trades), 2) if self.trades else 0.0,
            "max_drawdown_pct": round(dd_rel * 100, 2),
            "sharpe": round(sharpe_ratio(r), 2),
            "sortino": round(sortino_ratio(r), 2),
            "cagr_pct": round(cagr_value * 100, 2),
            "calmar": round(calmar_ratio(cagr_value, dd_rel), 2) if dd_rel > 0 else 0.0,
            "sqn": round(sqn(self.trades), 2),
            "expectancy": round(expectancy(self.trades), 4),
            "avg_trade_pnl_pct": round(
                (sum(pnls) / len(pnls) / self.initial_capital * 100) if pnls else 0.0, 2
            ),
            "max_win_streak": streaks["max_win_streak"],
            "max_loss_streak": streaks["max_loss_streak"],
        }


class Backtester:
    def __init__(self, settings: Settings, market: MarketData):
        self.s = settings
        self.market = market
        self.fee = settings.backtest.fee_pct / 100.0
        self.slippage = settings.backtest.slippage_pct / 100.0

    def run(self, strategy: BaseStrategy, pairs: list[str] | None = None,
            days: int | None = None) -> BacktestResult:
        pairs = pairs or self.s.trading_pairs
        days = days or self.s.backtest.days
        capital = self.s.backtest.initial_capital
        equity = capital
        trades: list[BacktestTrade] = []
        curve = [capital]

        # Cargamos todas las velas de cada par en el rango
        end = datetime.now(timezone.utc).replace(tzinfo=None)
        start = end - timedelta(days=days)
        frames: dict[str, pd.DataFrame] = {}
        for pair in pairs:
            df = self.market.fetch_ohlcv(pair, self.s.timeframe, 5000)
            df = df[(df.index >= start) & (df.index <= end)]
            df = strategy.compute_indicators(df)
            if len(df) < 50:
                log.warning("Pocas velas para %s (%d)", pair, len(df))
                continue
            frames[pair] = df

        max_len = max((len(df) for df in frames.values()), default=0)
        open_positions: dict[str, dict] = {}

        for i in range(1, max_len):
            for pair, df in frames.items():
                if i >= len(df):
                    continue
                row = df.iloc[i]
                price = row["close"]
                low, high = row["low"], row["high"]

                pos = open_positions.get(pair)
                # 1) Gestión de salida: stop-loss / take-profit / trailing en el rango de la vela
                if pos:
                    closed_pnl = self._manage_position(pos, low, high, trades, i)
                    equity += closed_pnl

                    if closed_pnl:
                        open_positions.pop(pair, None)

                    pos = open_positions.get(pair)  # puede haberse cerrado
                    if not closed_pnl and pos:
                        # 2) Señal de salida de estrategia
                        if strategy.exit_signal(df, row, pos["side"]):
                            pnl = self._close_at_price(pair, pos, price, i, "signal", trades)
                            equity += pnl
                            open_positions.pop(pair, None)
                        continue

                # 3) Señal de entrada
                signal = strategy.entry_signal(df, row)
                if signal and len(open_positions) < self.s.risk.max_open_trades:
                    atr_value = float(row.get("atr", 0.0)) or 0.0
                    if atr_value > 0:
                        stake = position_size_atr(
                            self.s.risk, equity, atr_value, price,
                            risk_pct=self.s.risk.risk_per_trade_pct,
                            atr_multiplier=self.s.risk.atr_sl_multiplier,
                            open_trades=len(open_positions),
                        )
                    else:
                        stake = equity * self.s.risk.max_position_pct
                    if stake > 1:
                        exec_price = price * (1 + self.slippage)
                        amount = stake / exec_price
                        open_positions[pair] = {
                            "pair": pair, "side": signal, "amount": amount, "open_price": exec_price,
                            "stake": stake, "open_index": i,
                            "stop_loss": stop_price_atr(signal, exec_price, atr_value, self.s.risk.atr_sl_multiplier)
                            if atr_value > 0 else None,
                            "take_profit": take_profit_price_atr(signal, exec_price, atr_value, self.s.risk.atr_tp_multiplier)
                            if atr_value > 0 else None,
                        }
                        log.debug("[BT] ENTRY %s %s @ %.2f", pair, signal, exec_price)

            # Equity del día (mark-to-market: caja + posiciones abiertas valoradas)
            mark = equity
            for pair, pos in open_positions.items():
                df = frames.get(pair)
                if df is not None and i < len(df):
                    px = df.iloc[i]["close"]
                    unrealized = (pos["amount"] * px - pos["stake"]) if pos["side"] == "long" \
                        else (pos["stake"] - pos["amount"] * px)
                    mark = mark + unrealized
            curve.append(mark)

        # Cerramos posiciones restantes al último precio
        for pair, pos in open_positions.items():
            df = frames[pair]
            price = df.iloc[-1]["close"]
            pnl = self._close_at_price(pair, pos, price, len(df) - 1, "end_of_test", trades)
            equity += pnl
        equity = curve[-1]

        total_return = (equity - capital) / capital if capital else 0.0
        return BacktestResult(
            initial_capital=capital,
            final_equity=equity,
            total_return=total_return,
            trades=trades,
            equity_curve=curve,
            days=max(1.0, days),
        )

    def _manage_position(self, pos: dict, low: float, high: float, trades: list, i: int) -> float:
        """Gestiona SL/TP/trailing de una posición en el rango de la vela.

        Devuelve el pnl_abs si se cerró, 0.0 si no hubo cierre.
        """
        risk = self.s.risk
        side = pos["side"]
        sl = pos.get("stop_loss")
        tp = pos.get("take_profit")

        # Stop-loss por defecto si no se especificó en la entrada
        if sl is None:
            sl = pos["open_price"] * (1 - risk.stop_loss_pct) if side == "long" else pos["open_price"] * (1 + risk.stop_loss_pct)
        if tp is None:
            tp = pos["open_price"] * (1 + risk.take_profit_pct) if side == "long" else pos["open_price"] * (1 - risk.take_profit_pct)

        # Trailing stop: mueve el SL con el mejor precio tras activarse
        if risk.trailing_stop_pct > 0:
            best = pos.get("best_price", pos["open_price"])
            if side == "long":
                best = max(best, high)
                if best >= pos["open_price"] * (1 + risk.trailing_activate_pct):
                    candidate = best * (1 - risk.trailing_stop_pct)
                    if candidate > (pos.get("stop_loss") or 0):
                        sl = candidate
            else:
                best = min(best, low)
                if best <= pos["open_price"] * (1 - risk.trailing_activate_pct):
                    candidate = best * (1 + risk.trailing_stop_pct)
                    if candidate < (pos.get("stop_loss") or float("inf")):
                        sl = candidate
            pos["best_price"] = best
            pos["stop_loss"] = sl

        if side == "long":
            if low <= sl:
                return self._close_at_price(pos["pair"], pos, sl, i, "stop_loss", trades)
            if high >= tp:
                return self._close_at_price(pos["pair"], pos, tp, i, "take_profit", trades)
        else:
            if high >= sl:
                return self._close_at_price(pos["pair"], pos, sl, i, "stop_loss", trades)
            if low <= tp:
                return self._close_at_price(pos["pair"], pos, tp, i, "take_profit", trades)
        return 0.0

    def _close_at_price(self, pair: str, pos: dict, price: float, i: int,
                        reason: str, trades: list) -> float:
        exec_price = price * (1 - self.slippage) if pos["side"] == "long" else price * (1 + self.slippage)
        open_val = pos["amount"] * pos["open_price"]
        close_val = pos["amount"] * exec_price
        if pos["side"] == "long":
            pnl_abs = close_val * (1 - self.fee) - open_val * (1 + self.fee)
        else:
            pnl_abs = open_val * (1 - self.fee) - close_val * (1 + self.fee)
        pnl_ratio = pnl_abs / open_val if open_val else 0.0
        trades.append(BacktestTrade(
            pair=pair, side=pos["side"],
            open_price=pos["open_price"], close_price=exec_price,
            amount=pos["amount"], open_index=pos["open_index"], close_index=i,
            pnl_abs=pnl_abs, pnl_ratio=pnl_ratio, exit_reason=reason,
        ))
        return pnl_abs


def run_backtest(settings: Settings, market: MarketData, strategy: BaseStrategy) -> BacktestResult:
    return Backtester(settings, market).run(strategy)