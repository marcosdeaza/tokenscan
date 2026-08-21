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
from ..utils.notifier import TelegramNotifier

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
        self.onchain = OnChainData(
            settings.agent.chain_enabled,
            settings.wallet_rpc,
            chain=settings.chain,
            private_key=settings.wallet_private_key,
        )
        self._client = None
        self._cycle = 0
        self.notifier = TelegramNotifier(settings)
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

    def _supported_pairs(self) -> list[str]:
        if self.s.mode == "live":
            from ..execution.jupiter import LIVE_PAIRS
            return [p for p in self.s.trading_pairs if p in LIVE_PAIRS]
        return list(self.s.trading_pairs)

    def run_cycle(self) -> list[Decision]:
        """Ciclo principal: recopila datos, evalúa SL/TP, decide, ejecuta, memoriza."""
        self._cycle += 1
        log.info("=== Ciclo agente #%d ===", self._cycle)

        self._auto_fund_if_live()

        prices = self._refresh_prices()
        exits = self.broker.check_exits(prices)
        for e in exits:
            log.info("[AGENT] Salida automática: %s — pnl=%.4f", e.get("exit_reason", "?"), e.get("pnl_abs", 0))
            if self.s.mode != "live":
                self.notifier.trade_closed(
                    e.get("pair", "?"), e.get("exit_reason", "?"),
                    e.get("pnl_abs", 0.0), e.get("pnl_ratio", 0.0),
                    e.get("signature"),
                )

        if self._risk_gate_blocks_new_trades():
            log.info("[AGENT] Risk gate activo: no se abren nuevas posiciones (solo gestión de las abiertas).")
            decisions: list[Decision] = []
        elif self.available:
            decisions = self._llm_decide()
        else:
            log.info("Sin LLM: usando fallback determinista (%s)", self._fallback_strategy.name)
            decisions = self._deterministic_decide()

        for d in decisions:
            try:
                self._execute_decision(d)
            except Exception as e:  # noqa: BLE001
                log.warning("[AGENT] Error ejecutando %s %s: %s", d.action, d.pair, e)

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

    def _auto_fund_if_live(self) -> None:
        """En modo live convierte el SOL sobrante a USDC cada ciclo (autónomo).

        El auto-fund solo se dispara al arrancar o al abrir operación; si el
        agente decide hold, nunca se ejecutaba y el SOL quedaba parado.
        Al correrlo en cada ciclo, en cuanto llegue SOL se convierte solo.
        """
        if self.s.mode != "live":
            return
        try:
            from ..execution.live import LiveBroker
            if isinstance(self.broker, LiveBroker):
                self.broker._auto_fund()
        except Exception as e:  # noqa: BLE001
            log.warning("[AGENT] Auto-fund en ciclo falló: %s", e)

    def _risk_gate_blocks_new_trades(self) -> bool:
        """Bloquea abrir nuevas posiciones si se superó la pérdida diaria máxima o
        si un stop-loss reciente deja al sistema en cooldown."""
        from datetime import datetime, timedelta, timezone
        equity = self.broker.get_equity()
        daily = self.db.daily_pnl(self.broker.wallet_id)
        if equity > 0 and daily <= -abs(self.s.risk.max_daily_loss_pct) * equity:
            log.warning("[AGENT] Pérdida diaria %+.2f supera el límite (%.0f%%) — halt",
                        daily, self.s.risk.max_daily_loss_pct * 100)
            return True
        last_sl = self.db.last_stop_loss_time(self.broker.wallet_id)
        if last_sl:
            cooldown_end = last_sl + timedelta(minutes=self.s.risk.cooldown_minutes)
            if datetime.now(timezone.utc) < cooldown_end:
                log.info("[AGENT] Cooldown tras stop-loss hasta %s", cooldown_end.isoformat())
                return True
        return False

    def _refresh_prices(self) -> dict[str, float]:
        """Actualiza precios en el broker y devuelve {pair: precio}."""
        prices: dict[str, float] = {}
        for pair in self.s.trading_pairs:
            try:
                df = self.market.fetch_ohlcv(pair, self.s.timeframe, 2)
                price = float(df.iloc[-1]["close"])
                prices[pair] = price
                self.broker.update_price(pair, price)
            except Exception as e:  # noqa: BLE001
                log.warning("Error fetching %s: %s", pair, e)
        return prices

    def _build_context(self) -> dict:
        prices: dict[str, float] = {}
        signals: dict[str, dict] = {}
        regimes: dict[str, dict] = {}
        scores: dict[str, dict] = {}
        for pair in self.s.trading_pairs:
            try:
                df = self.market.fetch_ohlcv(pair, self.s.timeframe, 200)
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
                    "adx": round(last.get("adx", 0), 1),
                    "kaufman_er": round(last.get("kaufman_er", 0), 3),
                    "bb_pct_b": round(last.get("bb_pct_b", 0.5), 3),
                    "stoch_k": round(last.get("stoch_k", 50), 1),
                    "roc": round(last.get("roc", 0), 4),
                    "vol_ratio": round(last.get("vol_ratio", 1), 2),
                }
                from ..quant.regime import detect_regime
                from ..quant.scorer import composite_score
                regimes[pair] = detect_regime(df).as_dict()
                scores[pair] = composite_score(df)
            except Exception as e:  # noqa: BLE001
                log.warning("Error fetching %s: %s", pair, e)

        wallet = self.broker.to_snapshot()
        positions = self.broker.open_positions()
        memory = self.db.get_recent_decisions(self.s.agent.memory_entries)
        news = self.news.fetch_crypto_news(5)
        onchain = self.onchain.latest_block()

        for pair, price in prices.items():
            self.broker.update_price(pair, price)

        return {
            "prices": prices,
            "signals": signals,
            "regimes": regimes,
            "scores": scores,
            "wallet": wallet,
            "positions": positions,
            "memory": [{"cycle": m["cycle"], "decision": m["decision"], "pnl": m["pnl_impact"]} for m in memory[-10:]],
            "news": news,
            "onchain": onchain,
        }

    def _build_prompt(self, ctx: dict) -> str:
        equity = ctx.get("wallet", {}).get("balance", 0) or 0
        max_pct = self.s.effective_max_pct(equity)
        sizing_rule = (
            f"usa hasta ~{int(max_pct * 100)}% del capital en una sola operación "
            f"(modo {'micro/auto (cuenta pequeña)' if max_pct >= 0.9 else 'standard'})"
        )
        return f"""Eres un agente de trading cuantitativo con IA. Tu objetivo es hacer crecer el capital.

CONTEXTO ACTUAL:
Señales técnicas por par: {json.dumps(ctx['signals'], indent=2)}
Régimen de mercado por par: {json.dumps(ctx['regimes'], indent=2)}
Score compuesto (ensemble) por par: {json.dumps(ctx['scores'], indent=2)}
Precios actuales: {json.dumps(ctx['prices'], indent=2)}
Cartera: {json.dumps(ctx['wallet'], indent=2)}
Posiciones abiertas: {json.dumps(ctx['positions'], indent=2)}
Últimas decisiones: {json.dumps(ctx['memory'], indent=2)}
Noticias: {json.dumps(ctx['news'], indent=2 if ctx['news'] else '')}
Datos on-chain: {json.dumps(ctx['onchain'], indent=2 if ctx['onchain'] else '')}

REGLAS:
- Solo puedes comprar si tienes efectivo disponible.
- Solo puedes vender si tienes posición abierta en ese par.
- Tamaño de posición: {sizing_rule}.
- Riesgo máximo por operación: 1% del capital (sizing por volatilidad/ATR).
- Respeta el régimen: en 'trend_up' prioriza compras, en 'trend_down' evita comprar,
  en 'ranging' busca reversión (RSI/Bollinger).
- El score compuesto en [-1, 1] es la convicción del ensemble: úsalo como
  señal objetiva, no solo tu intuición.
- Sé conservador con la confianza: solo tradea si confianza > 60.
- Sé BREVE en el razonamiento (máx. 20 palabras por decisión).

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
            max_tokens=16000,
        )
        content = resp.choices[0].message.content or ""
        if not content:
            log.warning("LLM returned empty content, finish_reason=%s reasoning=%s",
                        resp.choices[0].finish_reason,
                        resp.usage.completion_tokens_details.reasoning_tokens if resp.usage else "?")
        return content

    def _parse_llm_response(self, raw: str, prices: dict[str, float]) -> list[Decision]:
        clean = raw.strip()
        if clean.startswith("```"):
            clean = clean.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
        data = json.loads(clean)
        decisions: list[Decision] = []
        supported = set(self._supported_pairs())
        for d in data.get("decisions", data if isinstance(data, list) else []):
            pair = d["pair"]
            if pair not in prices or pair not in supported:
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

        from ..quant.regime import detect_regime
        from ..quant.scorer import composite_score

        for pair in self._supported_pairs():
            if pair in open_pairs:
                continue
            try:
                df = self.market.fetch_ohlcv(pair, self.s.timeframe, 200)
                df = self._fallback_strategy.compute_indicators(df)
                last = df.iloc[-1]

                score = composite_score(df)
                regime = detect_regime(df)
                price = float(last["close"])

                # Decisión determinista: ensemble + régimen + ATR
                signal = None
                if regime.regime.value in ("trend_up", "ranging") and score["decision"] == "long":
                    signal = "long"
                elif regime.regime.value == "trend_down" and score["decision"] == "short":
                    signal = "short"

                if signal:
                    self.broker.update_price(pair, price)
                    atr_value = float(last.get("atr", 0.0))
                    stake = self._position_size(price, atr_value)
                    qty = stake / price if price > 0 else 0
                    if qty > 0:
                        side = "buy" if signal == "long" else "sell"
                        reasoning = (
                            f"ensemble={score['score']:.2f} "
                            f"regime={regime.regime.value} "
                            f"adx={regime.adx:.0f} conf={score['confidence']:.0f}"
                        )
                        decisions.append(Decision(side, pair, qty, 80, reasoning, side=signal))
            except Exception as e:  # noqa: BLE001
                log.warning("Fallback error %s: %s", pair, e)

        return decisions[:self.s.agent.max_trades_per_cycle]

    def _position_size(self, price: float, atr_value: float) -> float:
        """Sizing por volatilidad (reglas Turtle): riesgo fijo % por ATR.

        Con capital pequeño el sizing ATR puede quedar por debajo del mínimo
        operativo; en ese caso usa el % máximo por posición para que el bot
        siga operando con stakes razonables (p. ej. ~20% de $12).
        """
        from ..quant.risk import position_size_atr
        equity = self.get_available_capital()
        min_stake = self.s.jupiter.min_trade_usd
        # El tier (micro/standard/auto) decide cuánto capital exponer:
        # micro despliega casi todo, standard es conservador.
        max_pct = self.s.effective_max_pct(equity)
        if atr_value and atr_value > 0 and price > 0:
            stake = position_size_atr(
                self.s.risk, equity, atr_value, price,
                risk_pct=self.s.risk.risk_per_trade_pct,
                atr_multiplier=self.s.risk.atr_sl_multiplier,
                open_trades=len(self.broker.open_positions()),
            )
            if stake < min_stake:
                stake = equity * max_pct
            return max(0.0, stake)
        return equity * max_pct

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
            if stake < self.s.jupiter.min_trade_usd:
                log.info("[AGENT] stake too small for %s (%.2f < min %.2f)",
                         d.pair, stake, self.s.jupiter.min_trade_usd)
                return
            sl, tp = self._stops(d.pair, "long", price)
            pos = self.broker.open_trade(d.pair, "long", stake, price, sl, tp)
            log.info("[AGENT] BUY %s qty=%.4f stake=%.2f conf=%.0f — %s",
                     d.pair, d.quantity, stake, d.confidence, d.reasoning[:60])
            if self.s.mode != "live":
                self.notifier.trade_opened(
                    d.pair, "long", pos.amount, pos.open_price, stake,
                    d.confidence, d.reasoning, pos.signature,
                )
        elif d.action == "sell":
            for pos in self.broker.positions.values():
                if pos.pair == d.pair:
                    price = self.broker.price(d.pair)
                    self.broker.close_trade(pos.trade_id, price, "agent_signal")
                    log.info("[AGENT] SELL %s — %s", d.pair, d.reasoning[:60])
                    break

    def _stops(self, pair: str, side: str, price: float) -> tuple[float, float]:
        """Stop-loss/take-profit: dinámicos por ATR si hay datos, si no por pct fijo."""
        from ..quant.risk import stop_price_atr, take_profit_price_atr
        atr_value = 0.0
        try:
            df = self.market.fetch_ohlcv(pair, self.s.timeframe, 60)
            atr_value = float(df["atr"].iloc[-1]) if "atr" in df.columns else 0.0
        except Exception as e:  # noqa: BLE001
            log.warning("ATR fetch error %s: %s", pair, e)
        if atr_value and atr_value > 0 and price > 0:
            sl = stop_price_atr(side, price, atr_value, self.s.risk.atr_sl_multiplier)
            tp = take_profit_price_atr(side, price, atr_value, self.s.risk.atr_tp_multiplier)
        else:
            sl = price * (1 - self.s.risk.stop_loss_pct)
            tp = price * (1 + self.s.risk.take_profit_pct)
        return sl, tp

    def _log_cycle(self, decisions: list[Decision]) -> None:
        summary = "; ".join(f"{d.action} {d.pair} ({d.confidence:.0f})" for d in decisions)
        self.db.save_decision(self._cycle, summary, "", 0.0)
        self.db.log_pnl(self.broker.wallet_id, self.broker.get_equity(), self.db.daily_pnl(self.broker.wallet_id))