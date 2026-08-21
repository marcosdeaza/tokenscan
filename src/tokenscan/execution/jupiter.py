"""Cliente de swaps on-chain en Solana vía Jupiter (agregador DEX).

Permite ejecutar swaps reales SOL <-> USDC usando la wallet del .env:
- Consulta rutas y precio (quote API)
- Construye la transacción de swap, la firma con la clave privada
- La envía al RPC y confirma el estado on-chain

Referencias:
- Jupiter API v6: https://station.jup.ag/docs/apis/swap-api
"""

from __future__ import annotations

import base64
import time
from typing import Any

import requests

from ..config import Settings
from ..utils.logger import setup_logger

log = setup_logger("tokenscan.jupiter")

SOL_MINT = "So11111111111111111111111111111111111111112"
USDC_MINT = "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v"
SOL_DECIMALS = 9
USDC_DECIMALS = 6

QUOTE_URL = "https://quote-api.jup.ag/v6/quote"
SWAP_URL = "https://quote-api.jup.ag/v6/swap"

# Pares soportados en modo live on-chain: base siempre SOL en Solana.
LIVE_PAIRS = {"SOL/USDC": (SOL_MINT, USDC_MINT), "SOL/USDT": (SOL_MINT, USDC_MINT)}


def _ui_to_raw(amount: float, decimals: int) -> int:
    return int(amount * 10**decimals)


def _raw_to_ui(raw: int, decimals: int) -> float:
    return float(raw) / 10**decimals


class JupiterClient:
    def __init__(self, settings: Settings):
        self.s = settings
        if not settings.wallet_private_key:
            raise ValueError("WALLET_PRIVATE_KEY requerida para operar on-chain")
        self._keypair = None
        self._load_keypair()

    def _load_keypair(self) -> None:
        try:
            from solders.keypair import Keypair
        except ImportError as e:  # pragma: no cover
            raise RuntimeError("Dependencia 'solders' no instalada (pip install 'tokenscan[solana]')") from e
        try:
            self._keypair = Keypair.from_base58_string(self.s.wallet_private_key)
        except Exception as e:
            raise ValueError(f"WALLET_PRIVATE_KEY inválida: {e}") from e

    @property
    def address(self) -> str:
        return str(self._keypair.pubkey())

    @property
    def enabled(self) -> bool:
        return self.s.jupiter.enabled

    def is_supported(self, pair: str) -> bool:
        return pair in LIVE_PAIRS

    # ── RPC ──────────────────────────────────────────────────

    def _rpc(self, method: str, params: list | None = None) -> dict | None:
        try:
            r = requests.post(
                self.s.solana_rpc_url,
                json={"jsonrpc": "2.0", "id": 1, "method": method, "params": params or []},
                timeout=20,
            )
            if r.ok:
                return r.json()
            log.warning("Solana RPC %s -> %s: %s", method, r.status_code, r.text[:200])
        except Exception as e:  # noqa: BLE001
            log.warning("Solana RPC %s error: %s", method, e)
        return None

    # ── Saldos on-chain ──────────────────────────────────────

    def get_sol_balance(self) -> float:
        res = self._rpc("getBalance", [self.address])
        if res and "result" in res:
            return float(res["result"]["value"]) / 1e9
        return 0.0

    def get_usdc_balance(self) -> float:
        res = self._rpc(
            "getTokenAccountsByOwner",
            [self.address, {"mint": USDC_MINT}, {"encoding": "jsonParsed"}],
        )
        if res and "result" in res:
            for account in res["result"]["value"]:
                try:
                    info = account["account"]["data"]["parsed"]["info"]
                    return float(info.get("tokenAmount", {}).get("uiAmount", 0) or 0)
                except (KeyError, TypeError):
                    continue
        return 0.0

    # ── Jupiter quote + swap ─────────────────────────────────

    def quote(self, in_mint: str, out_mint: str, amount_ui: float) -> dict:
        """Consulta la mejor ruta para un swap in -> out."""
        if not self.enabled:
            raise RuntimeError("Intercambios on-chain desactivados (jupiter.enabled=false)")
        in_dec = SOL_DECIMALS if in_mint == SOL_MINT else USDC_DECIMALS
        params = {
            "inputMint": in_mint,
            "outputMint": out_mint,
            "amount": _ui_to_raw(amount_ui, in_dec),
            "slippageBps": self.s.jupiter.slippage_bps,
        }
        r = requests.get(QUOTE_URL, params=params, timeout=20)
        if not r.ok:
            raise RuntimeError(f"Jupiter quote error {r.status_code}: {r.text[:300]}")
        data = r.json()
        if "outAmount" not in data:
            raise RuntimeError(f"Jupiter sin ruta para {in_mint}->{out_mint}: {str(data)[:200]}")
        return data

    def swap(self, in_mint: str, out_mint: str, amount_ui: float) -> dict[str, Any]:
        """Ejecuta un swap real y devuelve {signature, in_amount, out_amount, out_price}."""
        if not self.enabled:
            raise RuntimeError("Intercambios on-chain desactivados (jupiter.enabled=false)")
        quote = self.quote(in_mint, out_mint, amount_ui)
        out_dec = SOL_DECIMALS if out_mint == SOL_MINT else USDC_DECIMALS

        payload = {
            "quoteResponse": quote,
            "userPublicKey": self.address,
            "wrapAndUnwrapSol": True,
            "dynamicComputeUnitLimit": True,
            "prioritizationFeeLamports": self.s.jupiter.priority_fee_lamports,
        }
        r = requests.post(SWAP_URL, json=payload, timeout=30)
        if not r.ok:
            raise RuntimeError(f"Jupiter swap error {r.status_code}: {r.text[:300]}")
        tx_b64 = r.json().get("swapTransaction")
        if not tx_b64:
            raise RuntimeError("Jupiter no devolvió swapTransaction")

        signature = self._sign_and_send(tx_b64)
        out_amount = _raw_to_ui(int(quote["outAmount"]), out_dec)
        in_dec = SOL_DECIMALS if in_mint == SOL_MINT else USDC_DECIMALS
        in_amount = _raw_to_ui(int(quote["inAmount"]), in_dec)
        out_price = in_amount / out_amount if out_amount else 0.0
        return {
            "signature": signature,
            "in_amount": in_amount,
            "out_amount": out_amount,
            "out_price": out_price,
        }

    def _sign_and_send(self, tx_b64: str) -> str:
        from solders.transaction import VersionedTransaction

        tx_bytes = base64.b64decode(tx_b64)
        tx = VersionedTransaction.from_bytes(tx_bytes)
        sig = self._keypair.sign_message(tx.message.to_bytes())
        signed = VersionedTransaction.populate(tx.message, [sig])
        raw = base64.b64encode(bytes(signed)).decode()

        res = self._rpc("sendTransaction", [raw, {"encoding": "base64", "preflightCommitment": "confirmed"}])
        if res and "result" in res:
            return str(res["result"])
        raise RuntimeError(f"sendTransaction falló: {str(res)[:300]}" if res else "sin respuesta RPC")

    def confirm(self, signature: str, timeout_s: int | None = None) -> bool:
        """Confirma una transacción esperando a 'confirmed'/'finalized'."""
        timeout_s = timeout_s or self.s.jupiter.rpc_confirm_timeout_s
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            res = self._rpc("getSignatureStatuses", [[signature]])
            if res and "result" in res and res["result"]["value"]:
                status = res["result"]["value"][0]
                if status:
                    if status.get("confirmationStatus") in ("confirmed", "finalized") and not status.get("err"):
                        return True
                    if status.get("err"):
                        log.warning("Tx %s falló on-chain: %s", signature, status["err"])
                        return False
            time.sleep(2)
        log.warning("Timeout confirmando %s", signature)
        return False

    def price_sol(self) -> float:
        """Precio SOL en USDC vía Jupiter (swap de 1 SOL)."""
        quote = self.quote(SOL_MINT, USDC_MINT, 1.0)
        return _raw_to_ui(int(quote["outAmount"]), USDC_DECIMALS)

    def get_price(self, pair: str) -> float:
        if pair in LIVE_PAIRS:
            return self.price_sol()
        raise ValueError(f"Par no soportado on-chain: {pair}")
