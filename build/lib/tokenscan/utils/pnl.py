"""Cálculo de PnL y métricas de rendimiento (fórmulas estándar de mercado)."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class TradeResult:
    pair: str
    side: str  # "long" | "short"
    open_price: float
    close_price: float
    amount: float
    fee_open: float
    fee_close: float
    open_date: str
    close_date: str
    exit_reason: str

    @property
    def open_value(self) -> float:
        if self.side == "long":
            return self.amount * self.open_price * (1 + self.fee_open)
        return self.amount * self.open_price * (1 - self.fee_open)

    @property
    def close_value(self) -> float:
        if self.side == "long":
            return self.amount * self.close_price * (1 - self.fee_close)
        return self.amount * self.close_price * (1 + self.fee_close)

    @property
    def pnl_abs(self) -> float:
        if self.side == "long":
            return self.close_value - self.open_value
        return self.open_value - self.close_value

    @property
    def pnl_ratio(self) -> float:
        if self.open_value == 0:
            return 0.0
        if self.side == "long":
            return (self.close_value / self.open_value) - 1
        return 1 - (self.close_value / self.open_value)


def profit_factor(trades: list[TradeResult]) -> float:
    gross_profit = sum(t.pnl_abs for t in trades if t.pnl_abs > 0)
    gross_loss = abs(sum(t.pnl_abs for t in trades if t.pnl_abs < 0))
    if gross_loss == 0:
        return float("inf") if gross_profit > 0 else 0.0
    return gross_profit / gross_loss


def win_rate(trades: list[TradeResult]) -> float:
    if not trades:
        return 0.0
    wins = sum(1 for t in trades if t.pnl_abs > 0)
    return wins / len(trades)


def max_drawdown(equity_curve: list[float]) -> tuple[float, float]:
    """Máximo drawdown (absoluto y relativo) sobre una serie de equity."""
    if not equity_curve:
        return 0.0, 0.0
    peak = equity_curve[0]
    max_abs = 0.0
    max_rel = 0.0
    for value in equity_curve:
        peak = max(peak, value)
        dd_abs = peak - value
        dd_rel = dd_abs / peak if peak > 0 else 0.0
        max_abs = max(max_abs, dd_abs)
        max_rel = max(max_rel, dd_rel)
    return max_abs, max_rel


def sharpe_ratio(returns: list[float], periods_per_year: float = 365, risk_free: float = 0.0) -> float:
    if len(returns) < 2:
        return 0.0
    mean = sum(returns) / len(returns)
    variance = sum((r - mean) ** 2 for r in returns) / (len(returns) - 1)
    std = math.sqrt(variance)
    if std == 0:
        return 0.0
    excess = mean - risk_free / periods_per_year
    return (excess / std) * math.sqrt(periods_per_year)


def sortino_ratio(returns: list[float], periods_per_year: float = 365, risk_free: float = 0.0) -> float:
    if len(returns) < 2:
        return 0.0
    mean = sum(returns) / len(returns)
    downside = [r for r in returns if r < 0]
    if not downside:
        return float("inf")
    downside_std = math.sqrt(sum((r - mean) ** 2 for r in downside) / len(downside))
    if downside_std == 0:
        return 0.0
    excess = mean - risk_free / periods_per_year
    return (excess / downside_std) * math.sqrt(periods_per_year)


def kelly_fraction(win_rate: float, avg_win: float, avg_loss: float) -> float:
    """Fracción óptima de Kelly para sizing. Se usa a lo sumo la mitad (media-Kelly)."""
    if avg_loss <= 0 or avg_win <= 0:
        return 0.0
    b = avg_win / avg_loss  # ratio win/loss
    p = win_rate
    q = 1 - p
    if b == 0:
        return 0.0
    f = (p * b - q) / b
    return max(0.0, min(f * 0.5, 0.25))  # media-Kelly, cap 25%
