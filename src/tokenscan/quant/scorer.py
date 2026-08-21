"""Scoring compuesto: combina señales técnicas normalizadas en [-1, 1].

Cada sub-señal emite una convicción en [-1, 1]; el blend es la media ponderada
solo sobre las señales que "votan" (sin abstención), como en
virattt/ai-hedge-fund. Umbrales por defecto: long > +0.3, short < -0.25.
"""

from __future__ import annotations

import math

import pandas as pd

from .indicators import add_indicators


def _tanh_scaled(value: float, scale: float = 5.0) -> float:
    """Lleva un valor no acotado a [-1, 1] con tanh(x * scale)."""
    return math.tanh(float(value) * scale)


def _clamp(v: float) -> float:
    return max(-1.0, min(1.0, float(v)))


def _rsi_signal(rsi_value: float) -> float:
    return _clamp((rsi_value - 50.0) / 50.0)


def _stoch_signal(k_value: float) -> float:
    return _clamp((k_value - 50.0) / 50.0)


def _macd_signal(hist_value: float, hist_std: float) -> float:
    if hist_std <= 0:
        return 0.0
    return _tanh_scaled(hist_value / max(hist_std * 2.0, 1e-9))


def _bb_signal(pct_b: float) -> float:
    return _clamp((pct_b - 0.5) * 2.0)


def _momentum_signal(roc_value: float, roc_std: float) -> float:
    if roc_std <= 0:
        return 0.0
    return _clamp(roc_value / max(roc_std * 2.0, 1e-9))


def _ema_trend_signal(ema_fast: float, ema_slow: float) -> float:
    if ema_slow <= 0:
        return 0.0
    return _clamp((ema_fast - ema_slow) / ema_slow * 20.0)


def _trend_filter(adx_value: float, sign: float) -> float:
    """Atenúa las señales de reversión cuando hay tendencia fuerte."""
    if adx_value < 20:
        return 1.0
    if adx_value > 40:
        return 0.0
    return max(0.0, 1.0 - (adx_value - 20.0) / 20.0)


def composite_score(df: pd.DataFrame, weights: dict[str, float] | None = None) -> dict:
    """Calcula el score compuesto de la última vela.

    Devuelve: {score, signals: {nombre: convicción}, decision: "long"/"short"/"hold",
    confidence: 0..1}. Aplica un filtro de régimen que apaga las señales de
    reversión (RSI, Stoch, BB) cuando hay tendencia fuerte (ADX alto).
    """
    if "adx" not in df.columns:
        df = add_indicators(df)
    last = df.iloc[-1]
    hist_std = float(df["macd_hist"].std(ddof=0)) if len(df) > 20 else 0.0
    roc_std = float(df["roc"].std(ddof=0)) if len(df) > 20 else 0.0

    adx_value = float(last.get("adx", 0.0))

    mean_revert_filter = _trend_filter(adx_value, 0.0)

    signals: dict[str, float] = {
        "rsi": _rsi_signal(last.get("rsi", 50.0)),
        "stoch": _stoch_signal(last.get("stoch_k", 50.0)),
        "macd": _macd_signal(last.get("macd_hist", 0.0), hist_std),
        "bb": _bb_signal(last.get("bb_pct_b", 0.5)),
        "momentum": _momentum_signal(last.get("roc", 0.0), roc_std),
        "ema_trend": _ema_trend_signal(last.get("ema_fast", 0.0), last.get("ema_slow", 0.0)),
        "price_pos": _clamp((last.get("price_pos", 0.5) - 0.5) * 2.0),
    }

    weights = weights or {
        "rsi": 1.0,
        "stoch": 0.5,
        "macd": 1.0,
        "bb": 0.5,
        "momentum": 1.0,
        "ema_trend": 1.0,
        "price_pos": 0.3,
    }

    weighted: dict[str, float] = {}
    total_w = 0.0
    for name, value in signals.items():
        if name in ("rsi", "stoch", "bb"):
            value *= mean_revert_filter
        if name in ("macd", "ema_trend", "momentum"):
            value *= (1.0 - mean_revert_filter * 0.5)  # las de tendencia pierden algo en rango
        if value is None or value == 0.0:
            continue
        w = weights.get(name, 0.0)
        if w <= 0:
            continue
        weighted[name] = value
        total_w += w

    score = sum(v * weights[n] for n, v in weighted.items()) / total_w if total_w else 0.0
    score = _clamp(score)

    decision = "hold"
    if score > 0.3:
        decision = "long"
    elif score < -0.25:
        decision = "short"
    confidence = min(1.0, abs(score) * 2.0)

    return {
        "score": score,
        "signals": {k: round(v, 3) for k, v in signals.items()},
        "decision": decision,
        "confidence": round(confidence, 3),
    }


def composite_decision(df: pd.DataFrame, weights: dict[str, float] | None = None) -> str:
    """Atajo: devuelve solo la decisión ('long'/'short'/'hold')."""
    return composite_score(df, weights)["decision"]