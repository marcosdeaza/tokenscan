"""Agente de IA: loop de decisión con LLM configurable + fallback determinista.

El agente sigue el patrón de ai-hedge-fund (prompt + JSON parser) pero:
- Usa OpenAI SDK directo (sin LangChain, más ligero)
- Configurable: DeepSeek V4 Flash por defecto, cualquier API compatible con OpenAI
- Fallback automático a estrategia determinista (RSI) si no hay API key
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from ..config import Settings
from ..data.market import MarketData, NewsFeed, OnChainData
from ..execution.paper import PaperBroker
from ..quant.strategies import get_strategy
from ..storage.db import Database
from ..utils.logger import setup_logger

log = setup_logger("tokenscan.agent")


@dataclass
class Decision:
    action: str  # buy | sell | hold
    pair: str
    quantity: float
    confidence: float
    reasoning: str
    side: str = "long"


class LLMAgent:
    def __init__(self, settings: Settings, broker: PaperBroker, db: Database, market: MarketData):
        self.s = settings
        self.broker = broker
        self.db = db
        self.market = market
        self.news = NewsFeed(settings.agent.news_enabled)
        self.onchain = OnChainData(settings.agent.chain_enabled, settings.rpc_url)
        self._client = None
        self._cycle = 0
        self._fallback_strategy = get_strategy(
            settings.strategy.name,
            rsi_period=settings.strategy.rsi_period,
            oversold=settings.strategy.rsi_oversold,
            overbought=settings.strategy.rsi_overbought,
        )

    @property
    def client(self):
        if self._client is None and self.s.llm_api_key:
            from openai import OpenAI
            self._client = OpenAI(
                api_key=self.s.llm_api_key,
                base_url=self.s.llm_base_url,
            )
        return self._client

    @property
    def available(self) -> bool:
        return self.client is not None

    def run_cycle(self) -> list[Decision]:
        """Ciclo principal: recopila datos, decide, ejecuta, memoriza."""
        self._cycle += 1
        log.info("=== Ciclo agente #%d ===", self._cycle)

        if self.available:
            decisions = self._llm_decide()
        else:
            log.info("Sin LLM: usando fallback determinista (%s)", self._fallback_strategy.name)
            decisions = self._deterministic_decide()

        for d in decisions:
            self._execute_decision(d)

        self._log_cycle(decisions)
        return decisions

    # ── LLM decision ──────────────────────────────────────────

    def _llm_decide(self) -> list[Decision]:
        context = self._build_context()
        prompt = self._build_prompt(context)
        try:
            raw = self._call_llm(prompt)
            decisions = self._parse_llm_response(raw, context["prices"])
            log.info("LLM -> %s", json.dumps([d.action for d in decisions]))
            return decisions
        except Exception as e:  # noqa: BLE001
            log.warning("LLM error: %s — fallback determinista", e)
            return self._deterministic_decide()

    def _build_context(self) -> dict:
        prices: dict[str, float] = {}
        signals: dict[str, dict] = {}
        for pair in self.s.trading_pairs:
            try:
                df = self.market.fetch_ohlcv(pair, self.s.timeframe, 100)
                df = self._fallback_strategy.compute_indicators(df)
                last = df.iloc[-1]
                prices[pair] = last["close"]
                signals[pair] = {
                    "rsi": round(last.get("rsi", 50), 1),
                    "ema_fast": round(last.get("ema_fast", 0), 2),
                    "ema_slow": round(last.get("ema_slow", 0), 2),
                    "macd_hist": round(last.get("macd_hist", 0), 4),
                    "atr": round(last.get("atr", 0), 4),
                    "volatility": round(last.get("vol_ann", 0), 4),
                }
            except Exception as e:  # noqa: BLE001
                log.warning("Error fetching %s: %s", pair, e)

        wallet = self.broker.to_snapshot()
        positions = self.broker.open_positions()
        memory = self.db.get_recent_decisions(self.s.agent.memory_entries)
        news = self.news.fetch_crypto_news(5)
        onchain = self.onchain.latest_block()

        return {
            "prices": prices,
            "signals": signals,
            "wallet": wallet,
            "positions": positions,
            "memory": [{"cycle": m["cycle"], "decision": m["decision"], "pnl": m["pnl_impact"]} for m in memory[-10:]],
            "news": news,
            "onchain": onchain,
        }

    def _build_prompt(self, ctx: dict) -> str:
        return f"""Eres un agente de trading cuantitativo con IA. Tu objetivo es hacer crecer el capital.

CONTEXTO ACTUAL:
Señales técnicas por par: {json.dumps(ctx['signals'], indent=2)}
Precios actuales: {json.dumps(ctx['prices'], indent=2)}
Cartera: {json.dumps(ctx['wallet'], indent=2)}
Posiciones abiertas: {json.dumps(ctx['positions'], indent=2)}
Últimas decisiones: {json.dumps(ctx['memory'], indent=2)}
Noticias: {json.dumps(ctx['news'], indent=2 if ctx['news'] else '')}
Datos on-chain: {json.dumps(ctx['onchain'], indent=2 if ctx['onchain'] else '')}

REGLAS:
- Solo puedes comprar si tienes efectivo disponible.
- Solo puedes vender si tienes posición abierta en ese par.
- Cantidad máxima por operación: 20% del capital total.
- Riesgo máximo por operación: 5% del capital.
- Sé conservador con la confianza: solo tradea si confianza > 60.
- Ten en cuenta las noticias y datos on-chain en tu razonamiento.

RESPONDE EXACTAMENTE EN ESTE FORMATO JSON (sin markdown):
{{"decisions": [
  {{"pair": "BTC/USDT", "action": "buy", "quantity": 0.01, "confidence": 75, "reasoning": "..."}}
]}}
Acciones válidas: buy, sell, hold.
"""

    def _call_llm(self, prompt: str) -> str:
        resp = self.client.chat.completions.create(
            model=self.s.llm_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=2000,
        )
        return resp.choices[0].message.content or ""

    def _parse_llm_response(self, raw: str, prices: dict[str, float]) -> list[Decision]:
        clean = raw.strip()
        if clean.startswith("```"):
            clean = clean.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
        data = json.loads(clean)
        decisions: list[Decision] = []
        for d in data.get("decisions", data if isinstance(data, list) else []):
            pair = d["pair"]
            if pair not in prices:
                continue
            qty = float(d.get("quantity", 0))
            conf = float(d.get("confidence", 0))
            if conf < 60:
                continue
            decisions.append(Decision(
                action=d["action"],
                pair=pair,
                quantity=qty,
                confidence=conf,
                reasoning=d.get("reasoning", ""),
                side="long",
            ))
        return decisions[:self.s.agent.max_trades_per_cycle]

    # ── Deterministic fallback ────────────────────────────────

    def _deterministic_decide(self) -> list[Decision]:
        decisions: list[Decision] = []
        open_pairs = {p["pair"] for p in self.broker.open_positions()}

        for pair in self.s.trading_pairs:
            if pair in open_pairs:
                continue
            try:
                df = self.market.fetch_ohlcv(pair, self.s.timeframe, 100)
                df = self._fallback_strategy.compute_indicators(df)
                last = df.iloc[-1]
                signal = self._fallback_strategy.entry_signal(df, last)
                if signal == "long":
                    price = last["close"]
                    stake = self.get_available_capital() * 0.2
                    qty = stake / price if price > 0 else 0
                    if qty > 0:
                        decisions.append(Decision("buy", pair, qty, 80, "RSI oversold reversion"))
            except Exception as e:  # noqa: BLE001
                log.warning("Fallback error %s: %s", pair, e)

        return decisions[:self.s.agent.max_trades_per_cycle]

    def get_available_capital(self) -> float:
        return self.broker.get_balance()

    # ── Execution ─────────────────────────────────────────────

    def _execute_decision(self, d: Decision) -> None:
        if d.action == "hold":
            return
        if d.action == "buy":
            price = self.broker.price(d.pair)
            if price <= 0:
                return
            stake = d.quantity * price
            if stake > self.broker.get_balance():
                stake = self.broker.get_balance() * 0.95
            if stake < 1:
                log.info("[AGENT] stake too small for %s", d.pair)
                return
            sl = price * (1 - self.s.risk.stop_loss_pct)
            tp = price * (1 + self.s.risk.take_profit_pct)
            pos = self.broker.open_trade(d.pair, "long", stake, price, sl, tp)
            log.info("[AGENT] BUY %s qty=%.4f stake=%.2f conf=%.0f — %s",
                     d.pair, d.quantity, stake, d.confidence, d.reasoning[:60])
        elif d.action == "sell":
            for pos in self.broker.positions.values():
                if pos.pair == d.pair:
                    price = self.broker.price(d.pair)
                    self.broker.close_trade(pos.trade_id, price, "agent_signal")
                    log.info("[AGENT] SELL %s — %s", d.pair, d.reasoning[:60])
                    break

    def _log_cycle(self, decisions: list[Decision]) -> None:
        summary = "; ".join(f"{d.action} {d.pair} ({d.confidence:.0f})" for d in decisions)
        self.db.save_decision(self._cycle, summary, "", 0.0)
        self.db.log_pnl(self.broker.wallet_id, self.broker.get_equity(), 0.0)