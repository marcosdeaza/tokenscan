"""Cartera Solana usando solders (keypair) + requests (RPC)."""

from __future__ import annotations

import base58
import requests

from ..utils.logger import setup_logger
from .types import Chain, TokenBalance, WalletInfo

log = setup_logger("tokenscan.wallet.solana")

USDC_SOLANA = "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v"


def _rpc_call(url: str, method: str, params: list | None = None) -> dict | None:
    try:
        r = requests.post(
            url,
            json={"jsonrpc": "2.0", "id": 1, "method": method, "params": params or []},
            timeout=15,
        )
        if r.ok:
            data = r.json()
            return data.get("result")
        log.warning("Solana RPC error %s: %s", method, r.status_code)
    except Exception as e:  # noqa: BLE001
        log.warning("Solana RPC error %s: %s", method, e)
    return None


class SolanaWallet:
    def __init__(self, rpc_url: str, private_key: str | None = None):
        self.rpc_url = rpc_url
        self.private_key = private_key
        self._keypair = None

    @property
    def keypair(self):
        if self._keypair is None and self.private_key:
            from solders.keypair import Keypair
            self._keypair = Keypair.from_base58_string(self.private_key)
        return self._keypair

    @property
    def address(self) -> str | None:
        return str(self.keypair.pubkey()) if self.keypair else None

    @classmethod
    def create(cls, rpc_url: str) -> SolanaWallet:
        from solders.keypair import Keypair
        kp = Keypair()
        pk_b58 = base58.b58encode(bytes(kp)).decode()
        return cls(rpc_url, pk_b58)

    def get_native_balance(self) -> float:
        if not self.address:
            return 0.0
        result = _rpc_call(self.rpc_url, "getBalance", [self.address])
        if result and "value" in result:
            return float(result["value"]) / 1e9
        return 0.0

    def get_token_balance(self, mint: str) -> float:
        if not self.address:
            return 0.0
        result = _rpc_call(
            self.rpc_url,
            "getTokenAccountsByOwner",
            [
                self.address,
                {"mint": mint},
                {"encoding": "jsonParsed"},
            ],
        )
        if result and "value" in result:
            for account in result["value"]:
                try:
                    info = account["account"]["data"]["parsed"]["info"]
                    ui_amount = info.get("tokenAmount", {}).get("uiAmount", 0)
                    return float(ui_amount or 0)
                except (KeyError, TypeError):
                    continue
        return 0.0

    def get_usdc_balance(self) -> float:
        return self.get_token_balance(USDC_SOLANA)

    def get_info(self) -> WalletInfo:
        return WalletInfo(
            chain=Chain.SOLANA,
            address=self.address or "",
            private_key=self.private_key or "",
            native_balance=self.get_native_balance(),
            tokens=[
                TokenBalance(symbol="USDC", balance=self.get_usdc_balance(), decimals=6, address=USDC_SOLANA),
            ],
        )