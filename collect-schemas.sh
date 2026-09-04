#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
TEMP_DIR="$SCRIPT_DIR/temp"

rm -rf "$TEMP_DIR"
mkdir -p "$TEMP_DIR"

find "$SCRIPT_DIR/server" -name "schema.sql" -not -path "*/target/*" | sort | while read -r f; do
  service_name=$(echo "$f" | awk -F/src/ '{print $1}' | xargs basename)
  cp "$f" "$TEMP_DIR/${service_name}_schema.sql"
  echo "  ${service_name}_schema.sql"
done

echo "Done: $(ls -1 "$TEMP_DIR"/*.sql 2>/dev/null | wc -l) files collected"
