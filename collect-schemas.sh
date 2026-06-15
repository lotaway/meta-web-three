#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
TEMP_DIR="$SCRIPT_DIR/temp"

rm -rf "$TEMP_DIR"
mkdir -p "$TEMP_DIR"

find "$SCRIPT_DIR/server" -name "schema.sql" -not -path "*/target/*" | sort | while read -r f; do
  service_name=$(echo "$f" | sed -n 's|.*/\([^/]*-service\)/.*|\1|p')
  if [ -z "$service_name" ]; then
    service_name=$(basename "$(dirname "$f")")
  fi
  cp "$f" "$TEMP_DIR/${service_name}_schema.sql"
  echo "  ${service_name}_schema.sql"
done

echo "Done: $(ls -1 "$TEMP_DIR"/*.sql 2>/dev/null | wc -l) files collected"
