#!/bin/bash
# Collect all schema.sql files into a flat temp/ directory

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TARGET_DIR="$SCRIPT_DIR/temp"

rm -rf "$TARGET_DIR"
mkdir -p "$TARGET_DIR"

SQL_FILES=$(find "$SCRIPT_DIR" -name "schema.sql" -not -path "*/target/*" -not -path "*/node_modules/*")

if [ -z "$SQL_FILES" ]; then
    echo "No schema.sql files found."
    exit 0
fi

for sql_file in $SQL_FILES; do
    rel_path="${sql_file#$SCRIPT_DIR/}"
    flat_name=$(echo "$rel_path" | tr '/' '-')
    cp "$sql_file" "$TARGET_DIR/$flat_name"
    echo "Copied: $rel_path -> temp/$flat_name"
done

echo "Done. $(ls -1 "$TARGET_DIR" | wc -l) files collected into temp/"
