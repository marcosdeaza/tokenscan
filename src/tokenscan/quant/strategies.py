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


class MacroGate(TrendFollowing):
    """Gate macro EMA200 defensivo: solo comprar con tendencia viva.

    Filtra el régimen macro con una EMA de periodo largo sobre el close diario:
    solo abre largos si el precio cotiza por encima de esa EMA (mercado alcista)
    y cierra si la cruza hacia abajo. Long-only: en bear market se queda en cash,
    que es lo único que protege capital de forma honesta.

    Si se inyecta el histórico diario completo (macro_daily), el filtro usa una
    EMA diaria real (periodo `ema_macro`) sobre toda la historia; si no, resamplea
    el propio DataFrame a 1D como aproximación.
    """

    name = "macro_gate"

    def __init__(self, fast: int = 12, slow: int = 26, ema_macro: int = 200,
                 macro_daily: dict[str, pd.Series] | None = None,
                 min_kaufman_er: float = 0.0):
        super().__init__(fast=fast, slow=slow)
        self.ema_macro = ema_macro
        self.macro_daily = macro_daily or {}
        # Filtro de calidad de tendencia: exige un mínimo de direccionalidad en
        # la entrada. Las entradas con ER bajo (tendencia débil) son las que
        # revierten en 1-3 velas (los "trades malos").
        self.min_kaufman_er = min_kaufman_er

    def compute_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        out = add_indicators(df)
        pair = df.attrs.get("pair")
        daily = self.macro_daily.get(pair) if pair else None
        if daily is not None and len(daily.dropna()) > 20:
            span = min(self.ema_macro, max(20, len(daily.dropna()) // 3))
            ema_macro_daily = daily.ewm(span=span, adjust=False).mean()
            # Sin look-ahead: la EMA diaria del día D solo es conocida al final del
            # día D. Desplazamos 1 día para que la vela intraday de hoy use la EMA
            # calculada hasta el cierre de ayer (patrón freqtrade/jesse).
            ema_known = ema_macro_daily.shift(1)
            out["ema_macro"] = ema_known.reindex(out.index, method="ffill")
        else:
            close_daily = out["close"].resample("1D").last()
            span = min(self.ema_macro, max(20, len(close_daily.dropna()) // 3))
            ema_macro_daily = close_daily.ewm(span=span, adjust=False).mean()
            ema_known = ema_macro_daily.shift(1)
            out["ema_macro"] = ema_known.reindex(out.index, method="ffill")
        return out

    def entry_signal(self, df: pd.DataFrame, row: pd.Series) -> str | None:
        close = row.get("close", 0)
        macro = row.get("ema_macro", 0)
        fast = row.get("ema_fast", 0)
        slow = row.get("ema_slow", 0)
        if not (macro and close > macro and fast > slow):
            return None
        if self.min_kaufman_er > 0 and row.get("kaufman_er", 0) < self.min_kaufman_er:
            return None
        return "long"

    def exit_signal(self, df: pd.DataFrame, row: pd.Series, side: str) -> bool:
        close = row.get("close", 0)
        macro = row.get("ema_macro", 0)
        fast = row.get("ema_fast", 0)
        slow = row.get("ema_slow", 0)
        if side != "long":
            return False
        return bool(macro and (close < macro or fast < slow))


STRATEGIES: dict[str, type[BaseStrategy]] = {
    "rsi_reversion": RSIReversion,
    "trend_following": TrendFollowing,
    "macro_gate": MacroGate,
}


def get_strategy(name: str, **kwargs) -> BaseStrategy:
    cls = STRATEGIES.get(name)
    if not cls:
        raise ValueError(f"Estrategia desconocida: {name}. Disponibles: {list(STRATEGIES)}")
    return cls(**kwargs)