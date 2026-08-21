"""Tests del módulo de carteras on-chain (sin red, solo vectores de derivación)."""

from __future__ import annotations

import importlib.util

import pytest

from tokenscan.wallet import create_wallet, wallet_factory

HAS_WEB3 = importlib.util.find_spec("web3") is not None
HAS_SOLDERS = importlib.util.find_spec("solders") is not None

pytestmark = [
    pytest.mark.skipif(not HAS_WEB3, reason="Requiere pip install 'tokenscan[dex]'"),
    pytest.mark.skipif(not HAS_SOLDERS, reason="Requiere pip install 'tokenscan[solana]'"),
]


def test_evm_private_key_validation():
    """La clave privada EVM debe tener el prefijo 0x y 64 hex."""
    w = create_wallet("base", "https://mainnet.base.org")
    assert len(w.private_key) == 66
    assert w.private_key.startswith("0x")
    assert w.address.startswith("0x")
    assert len(w.address) == 42


def test_evm_known_derivation():
    """Vector oficial de Ethereum: la clave derivada debe coincidir."""
    w = wallet_factory("base", "", "0x4646464646464646464646464646464646464646464646464646464646464646")
    assert w.address.lower() == "0x9d8a62f656a8d1615c1294fd71e9cfb3e4855a4f"


def test_evm_derivation_is_deterministic():
    """Misma clave -> misma dirección (sin red)."""
    a = wallet_factory("evm", "", "0x" + "ab" * 32)
    b = wallet_factory("base", "", "0x" + "ab" * 32)
    assert a.address == b.address
    assert a.address.lower().startswith("0x")


def test_evm_different_keys_different_addresses():
    a = wallet_factory("base", "", "0x" + "11" * 32)
    b = wallet_factory("base", "", "0x" + "22" * 32)
    assert a.address != b.address


def test_solana_create_and_address_format():
    w = create_wallet("solana", "https://api.mainnet-beta.solana.com")
    assert len(w.address) == 44  # base58 pubkey
    assert w.private_key  # base58 clave completa
    assert w.address == str(w.keypair.pubkey())


def test_solana_derivation_roundtrip():
    w = create_wallet("solana", "")
    pk = w.private_key
    w2 = wallet_factory("solana", "", pk)
    assert w2.address == w.address


def test_solana_derivation_is_deterministic():
    w = create_wallet("solana", "")
    a = wallet_factory("solana", "", w.private_key)
    b = wallet_factory("solana", "", w.private_key)
    assert a.address == b.address


def test_unsupported_chain_raises():
    with pytest.raises(ValueError):
        wallet_factory("bitcoin", "", "")
    with pytest.raises(ValueError):
        create_wallet("foo", "")


def test_balance_returns_float():
    """Saldos con RPC público (red): devuelven float, no lanzan."""
    pytest.importorskip("web3")
    w = create_wallet("base", "https://mainnet.base.org")
    assert isinstance(w.get_native_balance(), float)
    assert isinstance(w.get_usdc_balance(), float)

    s = create_wallet("solana", "https://api.mainnet-beta.solana.com")
    assert isinstance(s.get_native_balance(), float)
    assert isinstance(s.get_usdc_balance(), float)
