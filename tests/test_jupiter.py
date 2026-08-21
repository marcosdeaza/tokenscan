"""Tests del cliente Jupiter y el broker live (sin red real, todo mockeado).

La clave usada es una clave DE PRUEBA generada al azar (sin fondos).
JAMÁS uses una clave real en tests ni la subas a un repositorio.
"""

from __future__ import annotations

import base64
from unittest.mock import patch

import pytest

from tokenscan.config import Settings
from tokenscan.execution.jupiter import (
    SOL_MINT,
    USDC_MINT,
    JupiterClient,
)
from tokenscan.execution.live import LiveBroker
from tokenscan.storage.db import Database

# Clave de prueba aleatoria (sin fondos reales). Generada únicamente para tests.
TEST_PK = "3sAQabQNYj5QqTMnV5tkPBUaSfZbkyKNfKWJnoFyZMrQyqpLnsgmwzpd9RxLY2SutWTurNXChmHwqCknrGHJnNoZ"


def make_settings(**overrides) -> Settings:
    data = {
        "mode": "live",
        "chain": "solana",
        "solana_rpc_url": "https://api.mainnet-beta.solana.com",
        "wallet_private_key": TEST_PK,
        "telegram": {"enabled": False},
        "agent": {"enabled": False},
    }
    data.update(overrides)
    return Settings(**data)


# ── JupiterClient ──────────────────────────────────────────


def test_client_rejects_missing_key():
    with pytest.raises(ValueError):
        JupiterClient(make_settings(wallet_private_key=""))


def test_client_rejects_invalid_key():
    with pytest.raises(ValueError):
        JupiterClient(make_settings(wallet_private_key="clave-invalida!!!"))


def test_client_derives_address():
    c = JupiterClient(make_settings())
    assert len(c.address) == 44


def test_is_supported_pair():
    c = JupiterClient(make_settings())
    assert c.is_supported("SOL/USDC")
    assert c.is_supported("SOL/USDT")
    assert not c.is_supported("BTC/USDT")


def test_quote_builds_correct_params():
    c = JupiterClient(make_settings())
    with patch("tokenscan.execution.jupiter.requests.get") as m:
        m.return_value.ok = True
        m.return_value.json.return_value = {
            "outAmount": "1000000",
            "inAmount": "1000",
            "priceImpactPct": "0.1",
        }
        q = c.quote(SOL_MINT, USDC_MINT, 1.0)
        assert q["outAmount"] == "1000000"
        _, kwargs = m.call_args
        assert kwargs["params"]["slippageBps"] == 100
        assert kwargs["params"]["inputMint"] == SOL_MINT
        assert kwargs["params"]["outputMint"] == USDC_MINT
        assert kwargs["params"]["amount"] == 1_000_000_000


def test_quote_raises_when_disabled():
    c = JupiterClient(make_settings())
    c.s.jupiter.enabled = False
    with pytest.raises(RuntimeError):
        c.quote(SOL_MINT, USDC_MINT, 1.0)


def test_quote_raises_without_route():
    c = JupiterClient(make_settings())
    with patch("tokenscan.execution.jupiter.requests.get") as m:
        m.return_value.ok = True
        m.return_value.json.return_value = {"error": "No route found"}
        with pytest.raises(RuntimeError):
            c.quote(SOL_MINT, USDC_MINT, 1.0)


def test_swap_signs_and_returns_result():
    c = JupiterClient(make_settings())
    quote = {"outAmount": "2000000000", "inAmount": "1000000000", "priceImpactPct": "0.2"}
    with (
        patch("tokenscan.execution.jupiter.requests.get") as mget,
        patch("tokenscan.execution.jupiter.requests.post") as mpost,
        patch.object(c, "_sign_and_send", return_value="firmsig"),
    ):
        mget.return_value.ok = True
        mget.return_value.json.return_value = quote
        mpost.return_value.ok = True
        mpost.return_value.json.return_value = {"swapTransaction": "dGVzdA=="}
        r = c.swap(SOL_MINT, USDC_MINT, 1.0)
    assert r["signature"] == "firmsig"
    assert r["in_amount"] == 1.0
    assert r["out_amount"] == 2000.0


def test_swap_raises_when_disabled():
    c = JupiterClient(make_settings())
    c.s.jupiter.enabled = False
    with pytest.raises(RuntimeError):
        c.swap(SOL_MINT, USDC_MINT, 1.0)


def test_swap_raises_on_jupiter_error():
    c = JupiterClient(make_settings())
    with patch("tokenscan.execution.jupiter.requests.get") as m:
        m.return_value.ok = False
        m.return_value.text = "error"
        m.return_value.status_code = 400
        with pytest.raises(RuntimeError):
            c.swap(SOL_MINT, USDC_MINT, 1.0)


def test_get_sol_balance_zero_when_no_result():
    c = JupiterClient(make_settings())
    with patch.object(c, "_rpc", return_value=None):
        assert c.get_sol_balance() == 0.0


def test_get_sol_balance_parses_lamports():
    c = JupiterClient(make_settings())
    with patch.object(c, "_rpc", return_value={"result": {"value": 250_000_000}}):
        assert c.get_sol_balance() == 0.25


def test_get_usdc_balance_zero_when_no_result():
    c = JupiterClient(make_settings())
    with patch.object(c, "_rpc", return_value=None):
        assert c.get_usdc_balance() == 0.0


def test_get_usdc_balance_parses_token_account():
    c = JupiterClient(make_settings())
    payload = {
        "result": {
            "value": [
                {
                    "account": {
                        "data": {
                            "parsed": {
                                "info": {
                                    "tokenAmount": {"uiAmount": 12.34, "decimals": 6}
                                }
                            }
                        }
                    }
                }
            ]
        }
    }
    with patch.object(c, "_rpc", return_value=payload):
        assert c.get_usdc_balance() == 12.34


def test_price_sol():
    c = JupiterClient(make_settings())
    with patch.object(c, "quote", return_value={"outAmount": "150000000"}):
        assert c.price_sol() == 150.0


def test_get_price_supported_pair():
    c = JupiterClient(make_settings())
    with patch.object(c, "quote", return_value={"outAmount": "150000000"}):
        assert c.get_price("SOL/USDC") == 150.0


def test_get_price_unsupported_pair():
    c = JupiterClient(make_settings())
    with pytest.raises(ValueError):
        c.get_price("BTC/USDT")


def test_confirm_true_when_finalized():
    c = JupiterClient(make_settings())
    with patch.object(
        c,
        "_rpc",
        return_value={
            "result": {
                "value": [
                    {
                        "confirmationStatus": "finalized",
                        "err": None,
                    }
                ]
            }
        },
    ):
        assert c.confirm("sigs") is True


def test_confirm_false_when_error():
    c = JupiterClient(make_settings())
    with patch.object(
        c,
        "_rpc",
        return_value={
            "result": {
                "value": [
                    {
                        "confirmationStatus": "confirmed",
                        "err": {"InstructionError": [0, {"Custom": 1}]},
                    }
                ]
            }
        },
    ):
        assert c.confirm("sigs") is False


def test_confirm_timeout_returns_false():
    c = JupiterClient(make_settings())
    with patch.object(
        c,
        "_rpc",
        return_value={"result": {"value": [None]}},
    ):
        assert c.confirm("sigs", timeout_s=0.1) is False


def test_sign_and_send_signs_real_tx():
    """Firma una transacción versionada real (solders 0.29) y la envía (mock RPC)."""
    from solders.hash import Hash
    from solders.instruction import Instruction
    from solders.keypair import Keypair
    from solders.message import MessageV0
    from solders.pubkey import Pubkey
    from solders.transaction import VersionedTransaction

    kp = Keypair()
    prog = Pubkey.from_string("11111111111111111111111111111111")
    inst = Instruction(prog, b"", [])
    blockhash = Hash.from_bytes(
        bytes.fromhex("1111111111111111111111111111111111111111111111111111111111111111")
    )
    msg = MessageV0.try_compile(kp.pubkey(), [inst], [], blockhash)
    tx = VersionedTransaction.populate(msg, [kp.sign_message(b"\x80" + bytes(msg))])
    tx.verify_and_hash_message()
    tx_b64 = base64.b64encode(bytes(tx)).decode()

    client = JupiterClient(make_settings())
    with patch.object(client, "_rpc", return_value={"result": "firmsig"}):
        sig = client._sign_and_send(tx_b64)
        assert sig == "firmsig"


def test_sign_and_send_raises_on_rpc_error():
    from solders.hash import Hash
    from solders.instruction import Instruction
    from solders.keypair import Keypair
    from solders.message import MessageV0
    from solders.pubkey import Pubkey
    from solders.transaction import VersionedTransaction

    kp = Keypair()
    prog = Pubkey.from_string("11111111111111111111111111111111")
    inst = Instruction(prog, b"", [])
    blockhash = Hash.from_bytes(
        bytes.fromhex("1111111111111111111111111111111111111111111111111111111111111111")
    )
    msg = MessageV0.try_compile(kp.pubkey(), [inst], [], blockhash)
    tx = VersionedTransaction.populate(msg, [kp.sign_message(b"\x80" + bytes(msg))])
    tx_b64 = base64.b64encode(bytes(tx)).decode()

    client = JupiterClient(make_settings())
    with patch.object(client, "_rpc", return_value=None), pytest.raises(RuntimeError):
        client._sign_and_send(tx_b64)


# ── LiveBroker ─────────────────────────────────────────────


def make_broker(tmp_path, **overrides):
    db = Database(str(tmp_path / "test.db"))
    b = LiveBroker(make_settings(**overrides), db)
    b._wallet_id = db.create_wallet("onchain", "USDC", 0.0)
    return b, db


def test_livebroker_requires_live_mode(tmp_path):
    db = Database(str(tmp_path / "test.db"))
    with pytest.raises(ValueError):
        LiveBroker(make_settings(mode="paper"), db)


def test_livebroker_requires_solana(tmp_path):
    db = Database(str(tmp_path / "test.db"))
    with pytest.raises(ValueError):
        LiveBroker(make_settings(chain="base"), db)


def test_ensure_wallet_creates_onchain(tmp_path):
    b, db = make_broker(tmp_path)
    with (
        patch.object(b.jupiter, "get_sol_balance", return_value=0.0),
        patch.object(b.jupiter, "get_usdc_balance", return_value=0.0),
    ):
        wid = b.ensure_wallet()
    assert db.get_wallet_by_label("onchain") is not None
    assert wid == db.get_wallet_by_label("onchain")["id"]


def test_auto_fund_converts_excess_sol(tmp_path):
    b, _ = make_broker(tmp_path)
    with (
        patch.object(b.jupiter, "get_sol_balance", return_value=1.0),
        patch.object(
            b.jupiter,
            "swap",
            return_value={
                "signature": "sigs",
                "in_amount": 0.9,
                "out_amount": 130.0,
                "out_price": 144.0,
            },
        ) as swap,
        patch.object(b.notifier, "send", return_value=True),
    ):
        usdc = b._auto_fund()
    assert usdc == 130.0
    swap.assert_called_once()


def test_auto_fund_skips_within_reserve(tmp_path):
    b, _ = make_broker(tmp_path)
    with (
        patch.object(b.jupiter, "get_sol_balance", return_value=0.005),
        patch.object(b.jupiter, "swap") as swap,
    ):
        usdc = b._auto_fund()
    assert usdc == 0.0
    swap.assert_not_called()


def test_auto_fund_swallows_swap_error(tmp_path):
    b, _ = make_broker(tmp_path)
    with (
        patch.object(b.jupiter, "get_sol_balance", return_value=1.0),
        patch.object(b.jupiter, "swap", side_effect=RuntimeError("boom")),
    ):
        assert b._auto_fund() == 0.0


def test_open_trade_falls_back_to_auto_fund(tmp_path):
    b, _ = make_broker(tmp_path)
    with (
        patch.object(b.jupiter, "get_sol_balance", return_value=1.0),
        patch.object(b.jupiter, "get_usdc_balance", side_effect=[0.0, 130.0]),
        patch.object(
            b.jupiter,
            "swap",
            side_effect=[
                {
                    "signature": "af",
                    "in_amount": 0.9,
                    "out_amount": 130.0,
                    "out_price": 144.0,
                },
                {
                    "signature": "buy",
                    "in_amount": 10.0,
                    "out_amount": 0.0694,
                    "out_price": 144.0,
                },
            ],
        ),
        patch.object(b.notifier, "send", return_value=True),
        patch.object(b.notifier, "trade_opened", return_value=True),
    ):
        pos = b.open_trade("SOL/USDC", "long", 10.0, 144.0, 136.0, 158.0)
    assert pos.amount == pytest.approx(0.0694)
    assert pos.signature == "buy"


def test_open_trade_raises_when_usdc_insufficient(tmp_path):
    b, _ = make_broker(tmp_path)
    with (
        patch.object(b.jupiter, "get_sol_balance", return_value=1.0),
        patch.object(b.jupiter, "get_usdc_balance", return_value=0.0),
        patch.object(b.jupiter, "swap", side_effect=RuntimeError("boom")),
        pytest.raises(RuntimeError),
    ):
        b.open_trade("SOL/USDC", "long", 10.0, 144.0, 136.0, 158.0)


def test_open_trade_rejects_short(tmp_path):
    b, _ = make_broker(tmp_path)
    with (
        patch.object(b.jupiter, "get_sol_balance", return_value=1.0),
        patch.object(b.jupiter, "get_usdc_balance", return_value=100.0),pytest.raises(RuntimeError)
    ):
        b.open_trade("SOL/USDC", "short", 10.0, 144.0, 136.0, 158.0)


def test_open_trade_rejects_unsupported_pair(tmp_path):
    b, _ = make_broker(tmp_path)
    with pytest.raises(RuntimeError):
        b.open_trade("BTC/USDT", "long", 10.0, 60000.0, 57000.0, 66000.0)


def test_open_trade_rejects_below_min_stake(tmp_path):
    b, _ = make_broker(tmp_path)
    with (
        patch.object(b.jupiter, "get_sol_balance", return_value=1.0),
        patch.object(b.jupiter, "get_usdc_balance", return_value=100.0),pytest.raises(RuntimeError)
    ):
        b.open_trade("SOL/USDC", "long", 0.01, 144.0, 136.0, 158.0)


def test_open_trade_caps_at_max_stake(tmp_path):
    b, _ = make_broker(tmp_path)
    b.s.jupiter.max_trade_usd = 50.0
    with (
        patch.object(b.jupiter, "get_sol_balance", return_value=1.0),
        patch.object(b.jupiter, "get_usdc_balance", return_value=1000.0),
        patch.object(
            b.jupiter,
            "swap",
            return_value={
                "signature": "buy",
                "in_amount": 50.0,
                "out_amount": 0.347,
                "out_price": 144.0,
            },
        ),
        patch.object(b.notifier, "trade_opened", return_value=True),
    ):
        pos = b.open_trade("SOL/USDC", "long", 100.0, 144.0, 136.0, 158.0)
    assert pos.stake == 50.0


def test_close_trade_returns_signature(tmp_path):
    b, _ = make_broker(tmp_path)
    with (
        patch.object(b.jupiter, "get_sol_balance", return_value=1.0),
        patch.object(b.jupiter, "get_usdc_balance", return_value=100.0),
        patch.object(
            b.jupiter,
            "swap",
            side_effect=[
                {
                    "signature": "buy",
                    "in_amount": 10.0,
                    "out_amount": 0.0694,
                    "out_price": 144.0,
                },
                {
                    "signature": "sell",
                    "in_amount": 0.0694,
                    "out_amount": 11.0,
                    "out_price": 158.0,
                },
            ],
        ),
        patch.object(b.notifier, "send", return_value=True),
        patch.object(b.notifier, "trade_opened", return_value=True),
        patch.object(b.notifier, "trade_closed", return_value=True),
        patch.object(b.jupiter, "price_sol", return_value=158.0),
    ):
        pos = b.open_trade("SOL/USDC", "long", 10.0, 144.0, 136.0, 158.0)
        result = b.close_trade(pos.trade_id, 158.0, "take_profit")
    assert result["signature"] == "sell"


def test_deposit_raises_in_live(tmp_path):
    b, _ = make_broker(tmp_path)
    with pytest.raises(RuntimeError):
        b.deposit(10.0)


def test_withdraw_raises_in_live(tmp_path):
    b, _ = make_broker(tmp_path)
    with pytest.raises(RuntimeError):
        b.withdraw(10.0)

