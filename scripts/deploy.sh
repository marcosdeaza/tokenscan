#!/usr/bin/env bash
# Despliegue de TokenScan en un VPS con Docker.
# Uso: ./scripts/deploy.sh [usuario@host]

set -euo pipefail

HOST="${1:-}"
if [ -z "$HOST" ]; then
    echo "Uso: ./scripts/deploy.sh usuario@vps-ip"
    exit 1
fi

echo "==> Comprobando docker en $HOST..."
ssh "$HOST" "docker --version && docker compose version" || {
    echo "Docker no instalado en el VPS. Ejecuta este one-liner:";
    echo "  curl -fsSL https://get.docker.com | sh";
    exit 1;
}

echo "==> Subiendo TokenScan..."
rsync -av --exclude '.venv' --exclude '__pycache__' --exclude 'data' --exclude '.git' \
    ./ "$HOST:~/tokenscan/"

echo "==> Configurando .env en el VPS (si no existe)..."
ssh "$HOST" "cd ~/tokenscan && ([ -f .env ] || cp .env.example .env) && echo 'Edita ~/tokenscan/.env con tus claves.'"

echo "==> Creando config.yaml en el VPS (si no existe)..."
ssh "$HOST" "cd ~/tokenscan && mkdir -p config && ([ -f config/config.yaml ] || cp config.yaml.example config/config.yaml) && echo 'config.yaml listo.'"

echo "==> Levantando contenedor..."
ssh "$HOST" "cd ~/tokenscan && docker compose up -d --build"

echo "==> Listo. Logs (tail 30):"
ssh "$HOST" "cd ~/tokenscan && docker compose logs --tail 30"
