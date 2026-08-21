"""Indicadores técnicos implementados con fórmulas estándar de mercado.

Sin dependencia de TA-Lib. Solo numpy/pandas.
Fuentes: fórmulas de Wilder (RSI/ATR), EMA, MACD, Bollinger clásicos.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def sma(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(period, min_periods=period).mean()


def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """RSI de Wilder: media suavizada de ganancias/pérdidas con alpha=1/period."""
    delta = series.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    out = 100 - (100 / (1 + rs))
    return out.fillna(50.0)


def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr


def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    return true_range(high, low, close).ewm(alpha=1 / period, adjust=False, min_periods=period).mean()


def macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> tuple[pd.Series, pd.Series, pd.Series]:
    line = ema(series, fast) - ema(series, slow)
    signal_line = ema(line, signal)
    hist = line - signal_line
    return line, signal_line, hist


def bollinger(series: pd.Series, period: int = 20, k: float = 2.0) -> tuple[pd.Series, pd.Series, pd.Series]:
    mid = sma(series, period)
    std = series.rolling(period, min_periods=period).std(ddof=0)
    return mid - k * std, mid, mid + k * std


def volatility(series: pd.Series, period: int = 21) -> pd.Series:
    """Volatilidad anualizada (sqrt de días de trading en un año ≈ 365)."""
    returns = series.pct_change()
    return returns.rolling(period, min_periods=period).std(ddof=0) * np.sqrt(365)


def add_indicators(df: pd.DataFrame, rsi_period: int = 14) -> pd.DataFrame:
    out = df.copy()
    out["rsi"] = rsi(out["close"], rsi_period)
    out["atr"] = atr(out["high"], out["low"], out["close"])
    out["ema_fast"], out["ema_slow"] = ema(out["close"], 12), ema(out["close"], 26)
    out["macd"], out["macd_signal"], out["macd_hist"] = macd(out["close"])
    out["bb_low"], out["bb_mid"], out["bb_high"] = bollinger(out["close"])
    out["vol_ann"] = volatility(out["close"])
    return out
