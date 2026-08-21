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


def adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> tuple[pd.Series, pd.Series, pd.Series]:
    """ADX (Wilder 1978): fuerza de tendencia. Devuelve (adx, +di, -di).

    +DM = H(t) - H(t-1) si > 0 y > D(t-1) - D(t); igual para -DM.
    +DI = 100 * Wilder_smooth(+DM, n) / ATR(n); DX = 100*|+DI--DI|/(+DI+-DI).
    """
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = pd.Series(
        np.where((up_move > down_move) & (up_move > 0), up_move, 0.0),
        index=high.index,
    )
    minus_dm = pd.Series(
        np.where((down_move > up_move) & (down_move > 0), down_move, 0.0),
        index=high.index,
    )
    atr_series = atr(high, low, close, period)

    def wilder(s: pd.Series) -> pd.Series:
        return s.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()

    plus_di = 100 * wilder(plus_dm) / atr_series.replace(0.0, np.nan)
    minus_di = 100 * wilder(minus_dm) / atr_series.replace(0.0, np.nan)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0.0, np.nan)
    adx_series = wilder(dx)
    return adx_series.fillna(0.0), plus_di.fillna(0.0), minus_di.fillna(0.0)


def efficiency_ratio(series: pd.Series, period: int = 10) -> pd.Series:
    """Efficiency Ratio de Kaufman: |P_t - P_{t-n}| / sum(|P_i - P_{i-1}|).

    Rango [0, 1]. ER > 0.5 -> tendencia; ER < 0.3 -> rango.
    """
    net_change = series.diff(period).abs()
    path = series.diff().abs().rolling(period, min_periods=period).sum()
    return (net_change / path.replace(0.0, np.nan)).fillna(0.0)


def bollinger_pct_b(series: pd.Series, period: int = 20, k: float = 2.0) -> pd.Series:
    """%B: posición del precio dentro de las bandas de Bollinger en [0, 1]."""
    low, _mid, high = bollinger(series, period, k)
    denom = (high - low).replace(0.0, np.nan)
    return ((series - low) / denom).clip(0.0, 1.0).fillna(0.5)


def bollinger_width(series: pd.Series, period: int = 20, k: float = 2.0) -> pd.Series:
    """Anchura de bandas normalizada: (upper - lower) / mid."""
    low, mid, high = bollinger(series, period, k)
    return ((high - low) / mid.replace(0.0, np.nan)).fillna(0.0)


def stochastic(high: pd.Series, low: pd.Series, close: pd.Series,
               k_period: int = 14, d_period: int = 3) -> tuple[pd.Series, pd.Series]:
    """Stochastic %K y %D: nivel del cierre dentro del rango de n velas."""
    lowest = low.rolling(k_period, min_periods=k_period).min()
    highest = high.rolling(k_period, min_periods=k_period).max()
    denom = (highest - lowest).replace(0.0, np.nan)
    pct_k = 100 * (close - lowest) / denom
    pct_d = pct_k.rolling(d_period, min_periods=d_period).mean()
    return pct_k.fillna(50.0), pct_d.fillna(50.0)


def volume_ratio(volume: pd.Series, period: int = 20) -> pd.Series:
    """Volumen relativo: volumen actual / media de volumen de n velas."""
    avg = volume.rolling(period, min_periods=period).mean().replace(0.0, np.nan)
    return (volume / avg).fillna(1.0)


def vwap(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series) -> pd.Series:
    """VWAP rolling: precio típico ponderado por volumen (ventana completa)."""
    typical = (high + low + close) / 3
    cum_vp = (typical * volume).cumsum()
    cum_v = volume.cumsum().replace(0.0, np.nan)
    return (cum_vp / cum_v).fillna(close)


def roc(series: pd.Series, period: int = 10) -> pd.Series:
    """Rate of change (momentum): (P_t / P_{t-n}) - 1."""
    return series.pct_change(period).fillna(0.0)


def price_position(series: pd.Series, period: int = 20) -> pd.Series:
    """Percentil del precio actual dentro de su rango de n velas: (P - min)/(max - min)."""
    lowest = series.rolling(period, min_periods=period).min()
    highest = series.rolling(period, min_periods=period).max()
    return ((series - lowest) / (highest - lowest).replace(0.0, np.nan)).clip(0.0, 1.0).fillna(0.5)


def add_indicators(df: pd.DataFrame, rsi_period: int = 14) -> pd.DataFrame:
    out = df.copy()
    out["rsi"] = rsi(out["close"], rsi_period)
    out["atr"] = atr(out["high"], out["low"], out["close"])
    out["ema_fast"], out["ema_slow"] = ema(out["close"], 12), ema(out["close"], 26)
    out["macd"], out["macd_signal"], out["macd_hist"] = macd(out["close"])
    out["bb_low"], out["bb_mid"], out["bb_high"] = bollinger(out["close"])
    out["bb_pct_b"] = bollinger_pct_b(out["close"])
    out["bb_width"] = bollinger_width(out["close"])
    out["vol_ann"] = volatility(out["close"])
    out["adx"], out["plus_di"], out["minus_di"] = adx(out["high"], out["low"], out["close"])
    out["kaufman_er"] = efficiency_ratio(out["close"])
    out["stoch_k"], out["stoch_d"] = stochastic(out["high"], out["low"], out["close"])
    if "volume" in out.columns:
        out["vol_ratio"] = volume_ratio(out["volume"])
        out["vwap"] = vwap(out["high"], out["low"], out["close"], out["volume"])
    out["roc"] = roc(out["close"])
    out["price_pos"] = price_position(out["close"])
    return out
