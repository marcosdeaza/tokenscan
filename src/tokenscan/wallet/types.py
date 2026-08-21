"""Tipos compartidos del módulo de carteras on-chain."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class Chain(str, Enum):
    """Cadenas soportadas por el módulo de carteras."""

    EVM = "evm"
    BASE = "base"
    SOLANA = "solana"


@dataclass
class TokenBalance:
    symbol: str
    balance: float
    decimals: int = 18
    address: str = ""

    @property
    def formatted(self) -> str:
        return f"{self.balance:,.8f}".rstrip("0").rstrip(".")


@dataclass
class WalletInfo:
    chain: Chain
    address: str
    private_key: str = ""
    native_balance: float = 0.0
    tokens: list[TokenBalance] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            f"  • Red: {self.chain.value}",
            f"  • Dirección: <code>{self.address}</code>",
        ]
        if self.native_balance:
            symbol = "SOL" if self.chain == Chain.SOLANA else "ETH"
            lines.append(f"  • Saldo nativo: {self.native_balance:.6f} {symbol}")
        for t in self.tokens:
            lines.append(f"  • {t.symbol}: {t.formatted}")
        return "\n".join(lines)
