"""Cartera EVM (Base, Ethereum, Polygon, etc.) usando web3."""

from __future__ import annotations

from ..utils.logger import setup_logger
from .types import Chain, TokenBalance, WalletInfo

log = setup_logger("tokenscan.wallet.evm")

USDC_BASE = "0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913"

ERC20_ABI = [
    {
        "constant": True,
        "inputs": [{"name": "_owner", "type": "address"}],
        "name": "balanceOf",
        "outputs": [{"name": "balance", "type": "uint256"}],
        "type": "function",
    },
    {
        "constant": True,
        "inputs": [],
        "name": "decimals",
        "outputs": [{"name": "", "type": "uint8"}],
        "type": "function",
    },
    {
        "constant": True,
        "inputs": [],
        "name": "symbol",
        "outputs": [{"name": "", "type": "string"}],
        "type": "function",
    },
]


class EVMWallet:
    def __init__(self, rpc_url: str, private_key: str | None = None):
        self.rpc_url = rpc_url
        self.private_key = private_key
        self._account = None
        self._w3 = None

    @property
    def w3(self):
        if self._w3 is None:
            from web3 import Web3
            self._w3 = Web3(Web3.HTTPProvider(self.rpc_url, request_kwargs={"timeout": 15}))
        return self._w3

    @property
    def account(self):
        if self._account is None and self.private_key:
            self._account = self.w3.eth.account.from_key(self.private_key)
        return self._account

    @property
    def address(self) -> str | None:
        return self.account.address if self.account else None

    @property
    def connected(self) -> bool:
        return self.w3.is_connected()

    @classmethod
    def create(cls, rpc_url: str) -> EVMWallet:
        from web3 import Web3
        w3 = Web3()
        account = w3.eth.account.create()
        return cls(rpc_url, "0x" + account.key.hex())

    def _checksum(self, address: str) -> str:
        from web3 import Web3
        return Web3.to_checksum_address(address)

    def get_native_balance(self) -> float:
        if not self.address:
            return 0.0
        try:
            wei = self.w3.eth.get_balance(self.address)
            return float(self.w3.from_wei(wei, "ether"))
        except Exception as e:  # noqa: BLE001
            log.warning("Error fetching native balance: %s", e)
            return 0.0

    def get_token_balance(self, token_address: str) -> float:
        if not self.address:
            return 0.0
        try:
            contract = self.w3.eth.contract(address=self._checksum(token_address), abi=ERC20_ABI)
            decimals = contract.functions.decimals().call()
            raw = contract.functions.balanceOf(self.address).call()
            return float(raw) / (10**decimals)
        except Exception as e:  # noqa: BLE001
            log.warning("Error fetching token balance %s: %s", token_address, e)
            return 0.0

    def get_usdc_balance(self) -> float:
        return self.get_token_balance(USDC_BASE)

    def get_info(self) -> WalletInfo:
        return WalletInfo(
            chain=Chain.BASE,
            address=self.address or "",
            private_key=self.private_key or "",
            native_balance=self.get_native_balance(),
            tokens=[
                TokenBalance(symbol="USDC", balance=self.get_usdc_balance(), decimals=6, address=USDC_BASE),
            ],
        )