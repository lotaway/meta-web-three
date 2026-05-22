#!/bin/bash
# Start Digital Twin Service in dev mode
# Requires: JDK 17+, Maven 3.8+

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SERVER_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

echo "=== Building Digital Twin Service ==="
cd "$SERVER_DIR"

mvn clean package -pl factory-domain/digital-twin-service -am -DskipTests -q

echo ""
echo "=== Starting Digital Twin Service (port 10102) ==="
echo "  REST API : http://localhost:10102/api/digital-twin/devices"
echo "  WebSocket: ws://localhost:10102/ws/digital-twin"
echo ""

java -jar "$SCRIPT_DIR/target/digital-twin-service-1.0.0.jar" \
    --spring.profiles.active=dev
