#!/usr/bin/env bash

set -euo pipefail

# Resolve the script's directory to locate docker-compose.env.yml
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMPOSE_FILE="${SCRIPT_DIR}/docker-compose.env.yml"

echo "=========================================="
echo "  Docker Compose Environment Cleaner"
echo "=========================================="
echo ""
echo "This script will stop all services and remove all containers and named volumes."
echo "WARNING: This action is IRREVERSIBLE and will destroy ALL DATA (databases, caches, etc.)."
echo ""

# Prompt for confirmation
read -r -p "Type 'yes' to confirm and proceed: " CONFIRM
if [ "$CONFIRM" != "yes" ]; then
    echo "Aborted by user. No changes were made."
    exit 0
fi

echo ""
echo "Stopping services and removing containers..."
docker compose -f "$COMPOSE_FILE" down -v --remove-orphans

echo ""
echo "=========================================="
echo "Cleanup finished successfully."
echo "To restart services, run: docker compose -f docker-compose.env.yml up -d"
echo "=========================================="
