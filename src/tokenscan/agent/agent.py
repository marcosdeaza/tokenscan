"""Agente de IA: loop de decisión con LLM configurable + fallback determinista.

El agente sigue el patrón de ai-hedge-fund (prompt + JSON parser) pero:
- Usa OpenAI SDK directo (sin LangChain, más ligero)
- Configurable: DeepSeek V4 Flash por defecto, cualquier API compatible con OpenAI
- Fallback automático a estrategia determinista (RSI) si no hay API key
"""

from __future__ import annotations

import itertools
import json
from dataclasses import dataclass

import pandas as pd

from ..config import Settings
from ..data.market import MarketData, NewsFeed, OnChainData
from ..execution.paper import PaperBroker
from ..quant.strategies import MacroGate, get_strategy
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
        name = settings.strategy.name
        kwargs: dict = {}
        if name == "rsi_reversion":
            kwargs = {
                "rsi_period": settings.strategy.rsi_period,
                "oversold": settings.strategy.rsi_oversold,
                "overbought": settings.strategy.rsi_overbought,
            }
        elif name in ("trend_following", "macro_gate"):
            kwargs = {"fast": settings.strategy.ema_fast, "slow": settings.strategy.ema_slow}
            if name == "macro_gate":
                kwargs["ema_macro"] = settings.strategy.ema_macro
                kwargs["min_kaufman_er"] = settings.strategy.min_kaufman_er
        self._fallback_strategy = get_strategy(name, **kwargs)

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

    def _inject_macro_daily(self) -> None:
        """Inyecta el histórico diario completo en la estrategia para que el gate
        macro (EMA diaria) sea real: se computa sobre velas 1d, no resampleadas."""
        if not isinstance(self._fallback_strategy, MacroGate):
            return
        macro_daily: dict[str, pd.Series] = {}
        for pair in self._supported_pairs():
            try:
                df = self.market.fetch_ohlcv(pair, "1d", 1000)
                macro_daily[pair] = df["close"]
            except Exception as e:  # noqa: BLE001
                log.warning("Macro daily %s: %s", pair, e)
        if macro_daily:
            self._fallback_strategy.macro_daily = macro_daily

    def run_cycle(self) -> list[Decision]:
        """Ciclo principal: recopila datos, evalúa SL/TP, decide, ejecuta, memoriza."""
        self._cycle += 1
        log.info("=== Ciclo agente #%d ===", self._cycle)

        self._auto_fund_if_live()
        self._inject_macro_daily()

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

        self._gate_exit()

        if self._risk_gate_blocks_new_trades():
            log.info("[AGENT] Risk gate activo: no se abren nuevas posiciones (solo gestión de las abiertas).")
            decisions: list[Decision] = []
        elif self.available:
            decisions = self._llm_decide()
        else:
            log.info("Sin LLM: usando fallback determinista (%s)", self._fallback_strategy.name)
            decisions = self._deterministic_decide()

        decisions = self._apply_gate_veto(decisions)

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
                entry = self._fallback_strategy.entry_signal(df, last)
                entry_signal = entry if entry else "none"
                signals[pair] = {
                    "rsi": round(last.get("rsi", 50), 1),
                    "ema_fast": round(last.get("ema_fast", 0), 2),
                    "ema_slow": round(last.get("ema_slow", 0), 2),
                    "ema_macro": round(last.get("ema_macro", 0), 2),
                    "macd_hist": round(last.get("macd_hist", 0), 4),
                    "atr": round(last.get("atr", 0), 4),
                    "volatility": round(last.get("vol_ann", 0), 4),
                    "adx": round(last.get("adx", 0), 1),
                    "kaufman_er": round(last.get("kaufman_er", 0), 3),
                    "bb_pct_b": round(last.get("bb_pct_b", 0.5), 3),
                    "stoch_k": round(last.get("stoch_k", 50), 1),
                    "roc": round(last.get("roc", 0), 4),
                    "vol_ratio": round(last.get("vol_ratio", 1), 2),
                    "macro_gate": entry_signal,
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

        wallet["daily_pnl"] = self.db.daily_pnl(self.broker.wallet_id)

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
- EL GATE MACRO ES LA LEY: el campo 'macro_gate' en Señales dice 'long' o 'none'.
  NUNCA compres si dice 'none' (el precio está por debajo de su EMA diaria larga o
  sin tendencia). Es el filtro de régimen validado en backtest: en bear market no se
  compra, y en tendencia se aguanta la posición.
- Con posición abierta en tendencia alcista (macro_gate='long'), aguanta el hold:
  no vendas por nervios ni por pequeñas caídas. Deja correr el crecimiento.
- LA CANTIDAD LA DECIDE LA MATEMÁTICA, NO TÚ: el sistema recalcula el tamaño de
  posición con ATR + vol-targeting. Pon 'quantity' aproximada; se ignora y se
  recalcula de forma determinista. Tú decides dirección (buy/sell/hold) y par.
- Riesgo máximo por operación: 1% del capital (sizing por volatilidad/ATR).
- Respeta el régimen: en 'trend_up' prioriza compras, en 'trend_down' evita comprar,
  en 'ranging' busca reversión (RSI/Bollinger).
- El score compuesto en [-1, 1] es la convicción del ensemble: úsalo como
  señal objetiva, no solo tu intuición.
- Sé conservador con la confianza: solo tradea si confianza > 60.
- PnL del día en cartera: si es negativo y grande, prefiere hold o cerrar; no apuestes
  a recuperar pérdidas con operaciones más grandes.
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

    def _gate_verdict(self, pair: str) -> tuple[str, float]:
        """Veredicto del gate macro para un par: (signal|none, ema_macro)."""
        try:
            df = self.market.fetch_ohlcv(pair, self.s.timeframe, 200)
            df = self._fallback_strategy.compute_indicators(df)
            last = df.iloc[-1]
            sig = self._fallback_strategy.entry_signal(df, last)
            return (sig or "none", float(last.get("ema_macro", 0) or 0))
        except Exception as e:  # noqa: BLE001
            log.warning("Gate verdict %s: %s", pair, e)
            return "none", 0.0

    def _gate_exit(self) -> None:
        """Cierra posiciones cuya señal de salida del gate macro se activó.

        En el backtest el gate sale cuando el precio cruza por debajo de su EMA
        diaria o la tendencia se debilita. En live, sin esto, el bot solo
        dependería del SL/TP automático y se quedaría dentro de la caída.
        """
        if not isinstance(self._fallback_strategy, MacroGate):
            return
        open_positions = self.broker.open_positions()
        for pos in open_positions:
            pair = pos["pair"]
            try:
                df = self.market.fetch_ohlcv(pair, self.s.timeframe, 200)
                df = self._fallback_strategy.compute_indicators(df)
                last = df.iloc[-1]
                if self._fallback_strategy.exit_signal(df, last, "long"):
                    price = float(last["close"])
                    self.broker.update_price(pair, price)
                    self.broker.close_trade(pos["id"], price, "macro_gate_exit")
                    log.info("[GATE] Salida %s: tendencia agotada (macro_gate_exit)", pair)
            except Exception as e:  # noqa: BLE001
                log.warning("Gate exit %s: %s", pair, e)

    def _apply_gate_veto(self, decisions: list[Decision]) -> list[Decision]:
        """Veto duro del gate macro (la estrategia validada es la ley).

        - Compra: solo si el gate dice 'long' (precio > EMA diaria + tendencia).
        - Venta: no se vende mientras la posición siga en tendencia alcista
          (hold: dejar correr los ganadores). El SL/TP automático ya gestiona el riesgo.
        """
        if not isinstance(self._fallback_strategy, MacroGate):
            return decisions
        out: list[Decision] = []
        for d in decisions:
            if d.action == "buy":
                sig, _ = self._gate_verdict(d.pair)
                if sig != "long":
                    log.info("[GATE] Veto compra %s: gate=%s", d.pair, sig)
                    continue
            elif d.action == "sell":
                pos = next((p for p in self.broker.open_positions() if p["pair"] == d.pair), None)
                if pos:
                    px = self.broker.price(d.pair)
                    pnl = (px / float(pos["open_price"]) - 1) if px and pos.get("open_price") else 0.0
                    if pnl > 0:
                        sig, _ = self._gate_verdict(d.pair)
                        if sig == "long":
                            log.info("[GATE] Hold %s: en tendencia alcista, no se corta el ganador", d.pair)
                            continue
            out.append(d)
        return out

    def _deterministic_decide(self) -> list[Decision]:
        decisions: list[Decision] = []
        open_pairs = {p["pair"] for p in self.broker.open_positions()}

        for pair in self._supported_pairs():
            if pair in open_pairs:
                continue
            try:
                sig, ema = self._gate_verdict(pair)
                if sig != "long":
                    continue
                df = self.market.fetch_ohlcv(pair, self.s.timeframe, 200)
                last = df.iloc[-1]
                price = float(last["close"])
                atr_value = float(last.get("atr", 0.0))
                self.broker.update_price(pair, price)
                stake = self._position_size(price, atr_value)
                qty = stake / price if price > 0 else 0
                if qty > 0:
                    decisions.append(Decision(
                        "buy", pair, qty, 85,
                        f"macro_gate long, ema_macro={ema:.2f}",
                        side="long",
                    ))
            except Exception as e:  # noqa: BLE001
                log.warning("Fallback error %s: %s", pair, e)

        return decisions[:self.s.agent.max_trades_per_cycle]

    def _position_size(self, price: float, atr_value: float) -> float:
        """Sizing por volatilidad (reglas Turtle): riesgo fijo % por ATR.

        Con capital pequeño el sizing ATR puede quedar por debajo del mínimo
        operativo; en ese caso usa el % máximo por posición para que el bot
        siga operando con stakes razonables (p. ej. ~20% de $12).

        Vol-targeting (patrón de llm-quant): el % máximo por posición se escala
        por (vol_objetivo / vol_realizada). Si la volatilidad sube, la exposición
        baja sola y viceversa. Así el riesgo por posición se mantiene constante.
        """
        from ..quant.risk import portfolio_vol_target, position_size_atr
        equity = self.get_available_capital()
        min_stake = self.s.jupiter.min_trade_usd
        max_pct = self.s.effective_max_pct(equity)
        target_vol = self.s.risk.vol_target_annual
        if atr_value and atr_value > 0 and price > 0:
            stake = position_size_atr(
                self.s.risk, equity, atr_value, price,
                risk_pct=self.s.risk.risk_per_trade_pct,
                atr_multiplier=self.s.risk.atr_sl_multiplier,
                open_trades=len(self.broker.open_positions()),
            )
            if stake < min_stake:
                scaled = portfolio_vol_target(
                    equity, self._daily_returns(), target_vol=target_vol,
                    max_leverage=self.s.risk.vol_max_leverage,
                )
                stake = scaled * max_pct
            return max(0.0, min(stake, equity * max_pct))
        return equity * max_pct

    def _daily_returns(self) -> list[float]:
        """Returns diarios de la cartera para el vol-targeting (máx. 30)."""
        try:
            rows = self.db.equity_curve(self.broker.wallet_id)[-30:]
            out = []
            for prev, cur in itertools.pairwise(rows):
                if prev["equity"] > 0:
                    out.append(cur["equity"] / prev["equity"] - 1)
            return out
        except Exception:  # noqa: BLE001
            return []

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
            # El tamaño de posición es SIEMPRE determinista (matemática manda):
            # el LLM propone dirección, nunca la cantidad. Vol-targeting + ATR.
            equity = self.broker.get_balance()
            atr_value = 0.0
            try:
                df = self.market.fetch_ohlcv(d.pair, self.s.timeframe, 200)
                atr_value = float(df["atr"].iloc[-1]) if "atr" in df.columns else 0.0
            except Exception as e:  # noqa: BLE001
                log.warning("ATR fetch %s: %s", d.pair, e)
            stake = self._position_size(price, atr_value)
            stake = max(0.0, min(stake, equity * 0.95))
            if stake < self.s.jupiter.min_trade_usd:
                log.info("[AGENT] stake too small for %s (%.2f < min %.2f)",
                         d.pair, stake, self.s.jupiter.min_trade_usd)
                return
            d.quantity = stake / price
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