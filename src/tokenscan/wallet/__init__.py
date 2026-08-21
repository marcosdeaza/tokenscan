"""Carteras on-chain: EVM/Base y Solana.

Usa una fábrica por cadena. Requiere deps opcionales:
- EVM/Base:  pip install "tokenscan[dex]"   (web3)
- Solana:    pip install "tokenscan[solana]" (solders)
"""

from __future__ import annotations

import importlib.util

from ..utils.logger import setup_logger
from .types import Chain, TokenBalance, WalletInfo

log = setup_logger("tokenscan.wallet")

__all__ = ["Chain", "TokenBalance", "WalletInfo", "create_wallet", "wallet_factory"]


def _missing_deps(extra: str) -> RuntimeError:
    return RuntimeError(
        f"Faltan dependencias para esta cartera. Ejecuta: pip install 'tokenscan[{extra}]'"
    )


def _require(pkg: str, extra: str) -> None:
    if importlib.util.find_spec(pkg) is None:
        raise _missing_deps(extra)


def wallet_factory(
    chain: str,
    rpc_url: str,
    private_key: str | None = None,
):
    """Devuelve una instancia de cartera para la cadena indicada."""
    chain = (chain or "").lower()
    if chain in ("base", "evm", "ethereum"):
        _require("web3", "dex")
        from .evm import EVMWallet
        return EVMWallet(rpc_url, private_key)
    if chain in ("solana", "sol"):
        _require("solders", "solana")
        from .solana import SolanaWallet
        return SolanaWallet(rpc_url, private_key)
    raise ValueError(f"Chain no soportada: {chain}")


def create_wallet(chain: str, rpc_url: str):
    """Crea una cartera nueva (genera un par de claves) para la cadena indicada."""
    chain = (chain or "").lower()
    if chain in ("base", "evm", "ethereum"):
        _require("web3", "dex")
        from .evm import EVMWallet
        return EVMWallet.create(rpc_url)
    if chain in ("solana", "sol"):
        _require("solders", "solana")
        from .solana import SolanaWallet
        return SolanaWallet.create(rpc_url)
    raise ValueError(f"Chain no soportada: {chain}")