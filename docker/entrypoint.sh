#!/bin/sh
# Entrypoint para el contenedor de TokenScan.
# Arranca el bot de Telegram, que inicia el loop del agente automáticamente.
# Sin TELEGRAM_BOT_TOKEN no hay control remoto; el agente se ejecuta solo.

set -e

CONFIG_DIR=/app/config
CONFIG_FILE=$CONFIG_DIR/config.yaml

if [ ! -f "$CONFIG_FILE" ]; then
    echo "config.yaml no existe en $CONFIG_DIR, copiando ejemplo..."
    cp /app/config.yaml.example "$CONFIG_FILE"
fi

MODE=$(grep '^mode:' "$CONFIG_FILE" | head -1 | awk '{print $2}' || echo paper)
echo "Arrancando TokenScan (modo: $MODE)"

if [ -n "$TELEGRAM_BOT_TOKEN" ] && [ "$TELEGRAM_BOT_TOKEN" != "123456789:TU-TOKEN-AQUI" ]; then
    echo "Bot de Telegram habilitado (agente autónomo activo)."
    exec python -m tokenscan telegram --config "$CONFIG_FILE"
else
    echo "TELEGRAM_BOT_TOKEN no configurado: arrancando solo el agente."
    exec python -m tokenscan run --config "$CONFIG_FILE"
fi
