#!/bin/sh
# Entrypoint para el contenedor de TokenScan.
# Ejecuta el loop del agente y el bot de Telegram. Si el bot no está
# configurado (sin TELEGRAM_BOT_TOKEN) el agente sigue corriendo.

set -e

CONFIG_DIR=/app/config
CONFIG_FILE=$CONFIG_DIR/config.yaml

if [ ! -f "$CONFIG_FILE" ]; then
    echo "config.yaml no existe en $CONFIG_DIR, copiando ejemplo..."
    cp /app/config.yaml.example "$CONFIG_FILE"
fi

MODE=$(grep '^mode:' "$CONFIG_FILE" | head -1 | awk '{print $2}' || echo paper)
echo "Arrancando TokenScan (modo: $MODE)"

python -m tokenscan run --config "$CONFIG_FILE" &

if [ -n "$TELEGRAM_BOT_TOKEN" ] && [ "$TELEGRAM_BOT_TOKEN" != "123456789:TU-TOKEN-AQUI" ]; then
    echo "Bot de Telegram habilitado."
    python -m tokenscan telegram --config "$CONFIG_FILE" &
else
    echo "TELEGRAM_BOT_TOKEN no configurado: el bot de Telegram no arrancará (el agente sigue activo)."
fi

wait
