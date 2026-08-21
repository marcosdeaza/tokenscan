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


def position_size_atr(
    risk: RiskConfig,
    equity: float,
    atr_value: float,
    price: float,
    risk_pct: float = 0.01,
    atr_multiplier: float = 2.0,
    open_trades: int = 0,
) -> float:
    """Sizing por volatilidad (reglas Turtle).

    size = (equity * risk_pct) / (atr_multiplier * ATR)
    Arriesgas `risk_pct` del equity por operación (por defecto 1%): si el precio
    se mueve atr_multiplier * ATR en tu contra, pierdes exactamente ese riesgo.
    """
    if atr_value <= 0 or price <= 0:
        return 0.0
    risk_amount = equity * risk_pct
    notional = risk_amount / (atr_multiplier * atr_value)
    slot_cap = equity / max(1, risk.max_open_trades - open_trades)
    return max(0.0, min(notional, slot_cap, equity))


def portfolio_vol_target(
    equity: float,
    daily_returns: list[float],
    target_vol: float = 0.15,
    max_leverage: float = 1.0,
) -> float:
    """Vol targeting de cartera: exposición = equity * min(vol_obj / vol_real, max_lev).

    vol_real se estima como std(daily_returns) * sqrt(252). Si la volatilidad
    sube, la exposición baja automáticamente (y viceversa).
    """
    if len(daily_returns) < 5:
        return equity
    import math

    mean = sum(daily_returns) / len(daily_returns)
    var = sum((r - mean) ** 2 for r in daily_returns) / (len(daily_returns) - 1)
    realized_vol = math.sqrt(var) * math.sqrt(252)
    if realized_vol <= 0:
        return equity
    leverage = min(target_vol / realized_vol, max_leverage)
    return equity * max(0.0, leverage)


def stop_price_atr(side: str, open_price: float, atr_value: float, multiplier: float = 2.0) -> float:
    """Stop-loss dinámico a `multiplier` ATRs del precio de entrada."""
    offset = atr_value * multiplier
    if side == "long":
        return open_price - offset
    return open_price + offset


def take_profit_price_atr(side: str, open_price: float, atr_value: float,
                          multiplier: float = 3.0) -> float:
    """Take-profit a `multiplier` ATRs (riesgo:recompensa controlado)."""
    offset = atr_value * multiplier
    if side == "long":
        return open_price + offset
    return open_price - offset


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
