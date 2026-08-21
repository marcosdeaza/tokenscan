"""Detección de régimen de mercado: tendencia alcista, bajista o rango.

Combina tres señales independientes (ADX, Efficiency Ratio de Kaufman y pendiente
de EMA) en un único régimen votado por mayoría. Referencia: Wilder (1978) para
ADX, Kaufman (1995) para el Efficiency Ratio, y el patrón clásico EMA slope de
freqtrade/QuantConnect para la dirección.
"""

from __future__ import annotations

import dataclasses
from enum import Enum

import pandas as pd

from .indicators import add_indicators

ADX_TREND_MIN = 20.0
ER_TREND_MIN = 0.35


class Regime(str, Enum):
    TREND_UP = "trend_up"
    TREND_DOWN = "trend_down"
    RANGING = "ranging"


@dataclasses.dataclass
class RegimeSignal:
    regime: Regime
    strength: float = 0.0  # 0..1 intensidad del régimen dominante
    adx: float = 0.0
    kaufman_er: float = 0.0
    ema_slope: float = 0.0

    def as_dict(self) -> dict:
        return {
            "regime": self.regime.value,
            "strength": round(self.strength, 3),
            "adx": round(self.adx, 1),
            "kaufman_er": round(self.kaufman_er, 3),
            "ema_slope": round(self.ema_slope, 5),
        }


def detect_regime(df: pd.DataFrame, ema_period: int = 50, slope_lookback: int = 5) -> RegimeSignal:
    """Clasifica el régimen de la última vela del DataFrame.

    Votos:
    1. ADX: >= umbral con +DI > -DI -> alcista; -DI > +DI -> bajista; < umbral -> rango.
    2. Efficiency Ratio: >= umbral -> tendencia (dirección por pendiente de precio).
    3. EMA slope: pendiente de la EMA de periodo `ema_period` a `slope_lookback` velas.

    El régimen ganador es el que reúna más votos; la fuerza es la fracción de
    señales que coinciden.
    """
    df = add_indicators(df) if "adx" not in df.columns else df
    last = df.iloc[-1]

    adx_v, plus_di, minus_di = last.get("adx", 0.0), last.get("plus_di", 0.0), last.get("minus_di", 0.0)
    er = last.get("kaufman_er", 0.0)
    close = df["close"]

    if len(df) >= ema_period + slope_lookback:
        ema_series = close.ewm(span=ema_period, adjust=False).mean()
        ema_now = ema_series.iloc[-1]
        ema_prev = ema_series.iloc[-1 - slope_lookback]
        ema_slope = (ema_now - ema_prev) / ema_prev if ema_prev else 0.0
    else:
        ema_slope = 0.0

    votes = {"trend_up": 0, "trend_down": 0, "ranging": 0}

    if adx_v >= ADX_TREND_MIN:
        votes["trend_up" if plus_di >= minus_di else "trend_down"] += 1
    else:
        votes["ranging"] += 1

    if er >= ER_TREND_MIN:
        votes["trend_up" if close.iloc[-1] >= close.iloc[-1 - min(10, len(df) - 1)] else "trend_down"] += 1
    else:
        votes["ranging"] += 1

    slope_eps = 0.002  # ~0.2% de pendiente por vela como umbral de "plano"
    if ema_slope > slope_eps:
        votes["trend_up"] += 1
    elif ema_slope < -slope_eps:
        votes["trend_down"] += 1
    else:
        votes["ranging"] += 1

    winner = max(votes, key=votes.get)
    total = sum(votes.values())
    strength = votes[winner] / total if total else 0.0
    return RegimeSignal(
        regime=Regime(winner),
        strength=strength,
        adx=float(adx_v),
        kaufman_er=float(er),
        ema_slope=float(ema_slope),
    )
