#!/bin/sh
# Entrypoint para el contenedor de TokenScan.
# El volumen host: ~/tokenscan/config/ → /app/config/
# Si no existe config.yaml, se copia el ejemplo.

set -e

CONFIG_DIR=/app/config
CONFIG_FILE=$CONFIG_DIR/config.yaml

if [ ! -f "$CONFIG_FILE" ]; then
    echo "config.yaml no existe en $CONFIG_DIR, copiando ejemplo..."
    cp /app/config.yaml.example "$CONFIG_FILE"
fi

echo "Arrancando TokenScan (modo: $(grep '^mode:' "$CONFIG_FILE" | head -1 || echo paper))"
exec python -m tokenscan run --config "$CONFIG_FILE"
