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


def cagr(initial: float, final: float, days: float) -> float:
    """Compound Annual Growth Rate: (final/initial)^(365/days) - 1."""
    if initial <= 0 or days <= 0:
        return 0.0
    return (final / initial) ** (365.0 / days) - 1.0


def calmar_ratio(cagr_value: float, max_dd_rel: float) -> float:
    """Calmar ratio: CAGR / max_drawdown (absoluto)."""
    if max_dd_rel <= 0:
        return 0.0
    return cagr_value / max_dd_rel


def sqn(trades: list[TradeResult]) -> float:
    """System Quality Number (Van Tharp): sqrt(n) * mean(pnl) / std(pnl).

    Requiere >= 30 trades para ser estable. Escala:
    < 1 pobre, 1.6-1.9 regular, 2.0-2.4 promedio, 2.5-2.9 bueno, 3.0-5.0 excelente, >5 excepcional.
    """
    if len(trades) < 2:
        return 0.0
    pnls = [t.pnl_abs for t in trades]
    mean = sum(pnls) / len(pnls)
    var = sum((p - mean) ** 2 for p in pnls) / (len(pnls) - 1)
    std = math.sqrt(var) if var > 0 else 0.0
    if std <= 0:
        return 0.0
    return math.sqrt(len(pnls)) * mean / std


def expectancy(trades: list[TradeResult]) -> float:
    """Expectancy: (win_rate * avg_win) - (loss_rate * avg_loss).

    Equivalente a mean(pnl) de los trades. Positiva = edge positivo.
    """
    if not trades:
        return 0.0
    wins = [t.pnl_abs for t in trades if t.pnl_abs > 0]
    losses = [t.pnl_abs for t in trades if t.pnl_abs < 0]
    if not wins and not losses:
        return 0.0
    win_rate_val = len(wins) / len(trades) if trades else 0.0
    loss_rate_val = len(losses) / len(trades) if trades else 0.0
    avg_win = sum(wins) / len(wins) if wins else 0.0
    avg_loss = abs(sum(losses)) / len(losses) if losses else 0.0
    return (win_rate_val * avg_win) - (loss_rate_val * avg_loss)


def win_loss_streaks(trades: list[TradeResult]) -> dict:
    """Máximas rachas de ganancias y pérdidas consecutivas."""
    max_win_streak = max_loss_streak = 0
    current_win = current_loss = 0
    for t in trades:
        if t.pnl_abs > 0:
            current_win += 1
            current_loss = 0
            max_win_streak = max(max_win_streak, current_win)
        else:
            current_loss += 1
            current_win = 0
            max_loss_streak = max(max_loss_streak, current_loss)
    return {"max_win_streak": max_win_streak, "max_loss_streak": max_loss_streak}
