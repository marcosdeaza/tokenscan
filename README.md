# TokenScan 🔭

**Simbiosis entre inteligencia artificial y mercados blockchain.**

TokenScan es un *kit* educativo-funcional para construir tu propio agente de
trading de criptomonedas: un bot de Telegram que crea carteras, recibe fondos y
deja que una IA navegue por los mercados, la blockchain e Internet, aplicando
juicio + matemáticas con un único objetivo: **hacer crecer el capital**.

Escrito 100% desde cero (MIT), basado en los patrones y las matemáticas probadas
de la comunidad (freqtrade, jesse, ccxt, ai-hedge-fund) — créditos en
[CREDITS.md](CREDITS.md).

> ⚠️ **Aviso**: proyecto educativo. Cripto puede perder todo tu dinero.
> Arranca en modo **paper**, y el dinero real queda bajo tu responsabilidad.

---

## ✨ Qué puedes hacer

| Comando de Telegram | Descripción |
|---------------------|-------------|
| `/wallet` | Ver tu cartera virtual |
| `/deposit 20` | Ingresar fondos (paper) |
| `/withdraw 10` | Retirar (paper) |
| `/balance` | Saldo, equity y PnL |
| `/positions` | Posiciones abiertas |
| `/trade BTC/USDT buy 20` | Operar manualmente |
| `/agent_start` / `/agent_stop` | Arrancar/parar la IA |
| `/status` | Estado del sistema |

CLI: `python -m tokenscan run | telegram | backtest | wallet`

---

## 🧠 Arquitectura

```
tokenscan/
├── tokenscan/
│   ├── quant/      → indicadores (RSI/EMA/ATR/MACD/Bollinger) + riesgo (Kelly, SL/TP, trailing)
│   ├── execution/  → broker paper (virtual) + cliente ccxt (modo real)
│   ├── data/       → feeds de mercado, noticias, on-chain
│   ├── agent/      → loop del agente: LLM (DeepSeek por defecto) + fallback RSI
│   ├── storage/    → SQLite (wallets, trades, órdenes, memoria, PnL)
│   ├── telegram/   → bot de Telegram (python-telegram-bot)
│   └── backtest/   → motor de backtesting con métricas (Sharpe, Sortino, DD…)
├── config.yaml     → configuración (copia config.yaml.example)
├── .env            → secretos (copia .env.example)
├── Dockerfile      → deploy en VPS con un comando
└── results/        → donde publicar los resultados de tus tests
```

---

## 🚀 Instalación local (5 min)

```bash
git clone https://github.com/marcosdeaza/tokenscan.git
cd tokenscan
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

cp .env.example .env        # → rellena tu token de Telegram
cp config.yaml.example config.yaml

python -m tokenscan wallet   # comprueba que arranca
python -m tokenscan run      # bucle del agente en paper trading
python -m tokenscan telegram # bot de Telegram
python -m tokenscan backtest # backtest de la estrategia
```

---

## 🤖 Configurar la IA (2 min)

TokenScan usa **DeepSeek** por defecto (barato y rápido, API compatible con
OpenAI). Solo tienes que poner tu clave en `.env`:

```ini
LLM_API_KEY=tu-clave
LLM_BASE_URL=https://api.deepseek.com/v1
LLM_MODEL=deepseek-chat
```

Sin clave, el agente cae automáticamente a una **estrategia determinista (RSI)**.
¿Quieres usar otro proveedor? Cambia `LLM_BASE_URL` y `LLM_MODEL`: cualquier API
compatible con OpenAI (OpenRouter, Groq, etc.) funciona sin tocar código.

---

## 🧪 Paper trading primero

El modo por defecto es **paper**: dinero virtual, riesgo cero, mismas matemáticas.
Cuando te sientas cómodo, prueba con **15–20 €** reales:

1. Cambia `mode: live` en `config.yaml`.
2. Rellena `EXCHANGE_NAME`, `EXCHANGE_API_KEY`, `EXCHANGE_API_SECRET` en `.env`
   (crea la key SOLO con permisos de trading spot, **sin retiros**).
3. `python -m tokenscan run`.

Registra tus resultados en `results/` y súbelos al repo: **la transparencia es
la mejor marca personal.** 📈

---

## 🖥️ Deploy en VPS (Docker)

```bash
# en tu máquina local
./scripts/deploy.sh usuario@ip-vps

# o manual
docker compose up -d --build
docker compose logs -f tokenscan
```

> En el VPS: `curl -fsSL https://get.docker.com | sh` si no tienes Docker.

---

## 📚 Documentación educativa

- [docs/guide.md](docs/guide.md) — La guía completa: cómo funciona cada pieza y por qué
- `tokenscan/quant/indicators.py` — las fórmulas, comentadas
- `tokenscan/quant/risk.py` — gestión de riesgo explicada

---

## 🔐 Seguridad

- **Nunca** subas `.env` ni claves a GitHub (`.gitignore` lo cubre).
- API keys de exchange: solo trading spot, sin permisos de retiro.
- La clave privada de wallet on-chain se usa solo en tu propio VPS.

---

## 🧾 Licencia

[MIT](LICENSE) © 2026 [Marcos de Aza](https://github.com/marcosdeaza).
Créditos e inspiración en [CREDITS.md](CREDITS.md).
