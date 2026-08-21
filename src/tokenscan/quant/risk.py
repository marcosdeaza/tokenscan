"""Gestión de riesgo: sizing (Kelly + límites), stop-loss, trailing y protecciones.

Fórmulas probadas del ecosistema (freqtrade et al.), reimplementadas de forma original.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from ..config import RiskConfig


@dataclass
class RiskSnapshot:
    equity: float
    open_trades: int
    daily_pnl: float
    last_stop_loss_time: datetime | None = None
    pair_locked_until: dict[str, datetime] | None = None

    def pair_locked(self, pair: str, now: datetime) -> bool:
        if not self.pair_locked_until:
            return False
        until = self.pair_locked_until.get(pair)
        return bool(until and now < until)


def position_size(
    risk: RiskConfig,
    equity: float,
    win_rate: float,
    avg_win: float,
    avg_loss: float,
    open_trades: int,
) -> float:
    """Tamaño de posición: media-Kelly limitado por límite por posición y slots."""
    from ..utils.pnl import kelly_fraction

    kelly = kelly_fraction(win_rate, avg_win, avg_loss)
    slot_cap = equity / max(1, risk.max_open_trades - open_trades)
    pct = max(min(kelly, risk.max_position_pct), risk.max_position_pct * 0.2)
    return max(0.0, min(equity * pct, slot_cap))


def stop_price(side: str, open_price: float, stop_pct: float) -> float:
    if side == "long":
        return open_price * (1 - abs(stop_pct))
    return open_price * (1 + abs(stop_pct))


def take_profit_price(side: str, open_price: float, tp_pct: float) -> float:
    if side == "long":
        return open_price * (1 + abs(tp_pct))
    return open_price * (1 - abs(tp_pct))


def trailing_stop_price(side: str, best_price: float, activate_price: float, trail_pct: float) -> float | None:
    """Devuelve el nuevo stop de trailing si procede (solo mejora el actual)."""
    if side == "long":
        if best_price < activate_price:
            return None
        return best_price * (1 - abs(trail_pct))
    if best_price > activate_price:
        return None
    return best_price * (1 + abs(trail_pct))


def should_halt_daily_loss(risk: RiskConfig, daily_pnl: float, equity: float) -> bool:
    if equity <= 0:
        return True
    return (daily_pnl / equity) <= -abs(risk.max_daily_loss_pct)
