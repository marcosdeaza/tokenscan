# TokenScan

![TokenScan](assets/hero.png)

Agente de trading de criptomonedas con interfaz de Telegram. Paper trading por
defecto, backtesting, y un modelo de lenguaje configurable (DeepSeek por defecto)
que decide operaciones sobre indicadores técnicos y datos de mercado — con un
fallback determinista (RSI) cuando no hay API key configurada.

Código original bajo MIT. Las fórmulas y métricas son estándar del análisis
técnico y la gestión de riesgo (ver [docs/formulas.md](docs/formulas.md)).

---

## Índice

- [Quick start](#quick-start)
- [Cómo funciona](#cómo-funciona)
- [Arquitectura](#arquitectura)
- [Configurar la IA](#configurar-la-ia)
- [Uso con Telegram](#uso-con-telegram)
- [Deploy en VPS](#deploy-en-vps)
- [Documentación](#documentación)
- [Licencia](#licencia)

---

## Quick start

```bash
git clone https://github.com/marcosdeaza/tokenscan.git
cd tokenscan
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

cp .env.example .env
cp config.yaml.example config.yaml

python -m tokenscan wallet    # verificar que arranca
python -m tokenscan backtest  # backtest de la estrategia configurada
python -m tokenscan run       # loop del agente en paper trading
```

CLI disponible: `run`, `telegram`, `backtest`, `wallet`.

---

## Cómo funciona

Cada ciclo (5 minutos por defecto), el agente:

1. Recopila precios, velas e indicadores (RSI, EMA, ATR, MACD, Bollinger).
2. Decide con el LLM si comprar, vender o esperar. Sin LLM, usa la estrategia `macro_gate`.
3. Ejecuta en el broker virtual (o real) con gestión de riesgo: stop-loss,
   take-profit y trailing stop.
4. Guarda la decisión y el resultado en SQLite.

### Estrategia por defecto: `macro_gate`

Filtro macro defensivo long-only. Solo abre posición si el precio cotiza por
encima de una EMA diaria larga (150 días) y la tendencia rápida es alcista; cierra
si cruza por debajo o la tendencia se debilita. En mercados bajistas se queda en
cash, que es lo único que protege el capital de forma honesta.

Backtest de 90 días con velas 4h (BTC/ETH/SOL, comisión 0.1% + slippage 0.05%):

| Capital | Resultado | Trades | Win rate | PF | DD máx |
|---------|-----------|--------|----------|----|--------|
| 5€  | +28.4% | 10 | 100% | ∞ | 1.2% |
| 50€ | +6.0%  | 10 | 100% | ∞ | 0.3% |
| 500€| +4.4%  | 10 | 100% | ∞ | 0.1% |

Los periodos por encima de la EMA diaria (tendencia alcista) concentran las
ganancias; los bajistas quedan fuera del mercado. La validación walk-forward de
60 días confirma el patrón: ~50% de ventanas positivas en todos los regímenes y
cero trades en mercados sin tendencia.

Las matemáticas detrás de cada módulo están documentadas en
[docs/formulas.md](docs/formulas.md).

![Backtest 5/50/500€](results/backtest.png)

---

## Arquitectura

![Arquitectura](assets/architecture.png)

```
src/tokenscan/
├── quant/      → indicadores + gestión de riesgo (Kelly, stop-loss, trailing)
├── execution/  → broker paper (virtual) + cliente ccxt (modo real)
├── data/       → feeds de mercado, noticias, on-chain
├── agent/      → loop del agente: LLM configurable + fallback determinista
├── wallet/     → carteras blockchain reales: EVM/Base (web3) y Solana (solders)
├── storage/    → SQLite (wallets, trades, órdenes, memoria, PnL)
├── telegram/   → bot de Telegram
├── backtest/   → motor de backtesting con métricas (Sharpe, Sortino, DD…)
└── config.py   → configuración tipada (pydantic)
```

---

## Carteras blockchain reales

TokenScan puede crear y consultar **carteras reales en cadena** (no solo la
virtual del paper trading):

| Cadena | Dependencia | RPC por defecto |
|--------|-------------|-----------------|
| Base (EVM) | `pip install "tokenscan[dex]"` | `https://mainnet.base.org` |
| Solana | `pip install "tokenscan[solana]"` | `https://api.mainnet-beta.solana.com` |

Para usarlas:

```bash
pip install -e ".[dex,solana]"

# En el .env:
CHAIN=base                    # base | solana
WALLET_PRIVATE_KEY=tu-clave   # déjala vacía para crearla con el bot
```

Crea una cartera nueva directamente desde Telegram con `/create_wallet base` (o
`solana`): el bot genera las claves, te muestra la dirección y la clave privada
una sola vez. Guárdala a buen recaudo: quien tenga la clave controla la cartera.

Consulta dirección y saldos reales en cualquier momento con `/wallet_onchain`.

> **Importante**: las carteras on-chain son de **solo lectura** (dirección y
> saldos). No envían ni operan fondos automáticamente. La ejecución de operaciones
> sigue siendo la del broker configurado (paper por defecto).

---

## Configurar la IA

TokenScan usa **DeepSeek** por defecto (API compatible con OpenAI, económica).
Configura en `.env`:

```ini
LLM_API_KEY=tu-clave
LLM_BASE_URL=https://api.deepseek.com/v1
LLM_MODEL=deepseek-chat
```

Cualquier proveedor con API compatible con OpenAI funciona: cambia
`LLM_BASE_URL` y `LLM_MODEL` (OpenRouter, Groq, etc.). Sin API key, el agente usa
la estrategia determinista RSI.

---

## Uso con Telegram

![Telegram](assets/telegram-demo.png)

| Comando | Descripción |
|---------|-------------|
| `/wallet` | Cartera virtual |
| `/wallet_onchain` | Ver cartera blockchain real (dirección y saldos) |
| `/create_wallet base\|solana` | Crear una cartera blockchain nueva |
| `/deposit 20` | Ingresar fondos (paper) |
| `/withdraw 10` | Retirar (paper) |
| `/balance` | Saldo, equity y PnL |
| `/positions` | Posiciones abiertas |
| `/trade BTC/USDT buy 20` | Operación manual |
| `/agent_start` / `/agent_stop` | Arrancar/parar el agente |
| `/status` | Estado del sistema |

Para arrancar el bot:

```bash
python -m tokenscan telegram
```

Requiere `TELEGRAM_BOT_TOKEN` (con [@BotFather](https://t.me/botfather)) y tu
`TELEGRAM_CHAT_ID` en `.env`.

---

## Deploy en VPS

```bash
./scripts/deploy.sh usuario@ip-vps
```

El script sincroniza el código, crea `config.yaml` si no existe y levanta el
contenedor con Docker Compose. Manualmente:

```bash
docker compose up -d --build
docker compose logs --tail 50
```

---

## Documentación

- [docs/guide.md](docs/guide.md) — cómo funciona cada módulo, en detalle
- [docs/formulas.md](docs/formulas.md) — fórmulas, explicaciones y referencias

## Licencia

[MIT](LICENSE) © 2026 Marcos de Aza. Dependencias y créditos en
[CREDITS.md](CREDITS.md).

---

> **Aviso de riesgo**: trading e inversión en criptoactivos conllevan un riesgo
> elevado de pérdida de capital. TokenScan es una herramienta educativa: no
> constituye asesoramiento financiero, y el rendimiento pasado no garantiza
> resultados futuros. Empieza en modo paper y asume la responsabilidad de tus
> operaciones con dinero real.
