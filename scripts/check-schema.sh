#!/usr/bin/env bash
# =============================================================================
# Database schema validation gate.
#
# There is no Flyway/Liquibase runner in this repository, so every service ships
# its own `src/main/resources/db/schema.sql`. This script validates that those
# files are present, non-empty, and free of duplicate table definitions, and
# fails the build when they are not.
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
STATUS=0

echo "Validating service schema.sql files..."

while IFS= read -r -d '' file; do
  rel="${file#"$PROJECT_ROOT"/}"
  if [ ! -s "$file" ]; then
    echo "ERROR: schema file is empty: $rel"
    STATUS=1
    continue
  fi

  tables=$(grep -oE 'CREATE TABLE IF NOT EXISTS[[:space:]]+[A-Za-z0-9_.]+' "$file" | awk '{print $NF}')
  count=$(printf '%s\n' "$tables" | sed '/^$/d' | wc -l | tr -d ' ')
  if [ "$count" -eq 0 ]; then
    echo "ERROR: no CREATE TABLE statements found in $rel"
    STATUS=1
    continue
  fi

  duplicates=$(printf '%s\n' "$tables" | sort | uniq -d | tr '\n' ' ')
  if [ -n "$duplicates" ]; then
    echo "ERROR: duplicate table definitions in $rel: $duplicates"
    STATUS=1
  fi

  echo "OK: $rel ($count tables)"
done < <(find "$PROJECT_ROOT/server" -path '*/src/main/resources/db/schema.sql' -print0 | sort -z)

if [ "$STATUS" -ne 0 ]; then
  echo "Schema validation FAILED"
  exit 1
fi
echo "Schema validation passed"
