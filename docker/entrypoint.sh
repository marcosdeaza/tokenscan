#!/bin/sh
# Entrypoint para el contenedor de TokenScan.
# Copia config.yaml.example a config.yaml si no existe.

set -e

if [ ! -f /app/config.yaml ]; then
    echo "config.yaml no existe, copiando ejemplo..."
    cp /app/config.yaml.example /app/config.yaml
fi

echo "Arrancando TokenScan (modo: $(grep '^mode:' /app/config.yaml | head -1 || echo paper))"
exec python -m tokenscan run
