"""Estrategias de trading deterministas.

Cada estrategia extiende BaseStrategy e implementa:
- compute_indicators(df) -> DataFrame
- entry_signal(df, row) -> "long" | "short" | None
- exit_signal(df, row, side) -> bool
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd

from .indicators import add_indicators


class BaseStrategy(ABC):
    name: str = "base"

    @abstractmethod
    def compute_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        ...

    @abstractmethod
    def entry_signal(self, df: pd.DataFrame, row: pd.Series) -> str | None:
        ...

    @abstractmethod
    def exit_signal(self, df: pd.DataFrame, row: pd.Series, side: str) -> bool:
        ...


class RSIReversion(BaseStrategy):
    name = "rsi_reversion"

    def __init__(self, rsi_period: int = 14, oversold: float = 30, overbought: float = 70):
        self.rsi_period = rsi_period
        self.oversold = oversold
        self.overbought = overbought

    def compute_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        out = add_indicators(df, rsi_period=self.rsi_period)
        return out

    def entry_signal(self, df: pd.DataFrame, row: pd.Series) -> str | None:
        rsi_val = row.get("rsi", 50)
        if rsi_val < self.oversold:
            return "long"
        if rsi_val > self.overbought:
            return "short"
        return None

    def exit_signal(self, df: pd.DataFrame, row: pd.Series, side: str) -> bool:
        rsi_val = row.get("rsi", 50)
        return (side == "long" and rsi_val > 50) or (side == "short" and rsi_val < 50)


class TrendFollowing(BaseStrategy):
    name = "trend_following"

    def __init__(self, fast: int = 12, slow: int = 26):
        self.fast = fast
        self.slow = slow

    def compute_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        out = add_indicators(df)
        return out

    def entry_signal(self, df: pd.DataFrame, row: pd.Series) -> str | None:
        fast = row.get("ema_fast", 0)
        slow = row.get("ema_slow", 0)
        if fast > slow > 0:
            return "long"
        if fast < slow:
            return "short"
        return None

    def exit_signal(self, df: pd.DataFrame, row: pd.Series, side: str) -> bool:
        fast = row.get("ema_fast", 0)
        slow = row.get("ema_slow", 0)
        return (side == "long" and fast < slow) or (side == "short" and fast > slow)


STRATEGIES: dict[str, type[BaseStrategy]] = {
    "rsi_reversion": RSIReversion,
    "trend_following": TrendFollowing,
}


def get_strategy(name: str, **kwargs) -> BaseStrategy:
    cls = STRATEGIES.get(name)
    if not cls:
        raise ValueError(f"Estrategia desconocida: {name}. Disponibles: {list(STRATEGIES)}")
    return cls(**kwargs)