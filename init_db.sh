#!/bin/bash

# Parse -y flag to skip confirmation
SKIP_CONFIRM=false
while getopts "y" opt; do
    case $opt in
        y) SKIP_CONFIRM=true ;;
        *) echo "Usage: $0 [-y]"; exit 1 ;;
    esac
done

# Load environment variables from .env file and strip carriage returns (\r)
if [ -f .env ]; then
    # More robust way to load .env, handling CRLF and spaces
    export $(grep -v '^#' .env | sed 's/\r$//' | xargs)
fi

# Set default values if environment variables are not set
# Note: Stripping \r from variables just in case they were set elsewhere with CRLF
DB_USER=$(echo "${POSTGRES_USER:-""}" | sed 's/\r$//')
DB_PASS=$(echo "${POSTGRES_PASSWORD:-""}" | sed 's/\r$//')
DB_HOST=$(echo "${POSTGRES_HOST:-"localhost"}" | sed 's/\r$//')
DB_PORT=$(echo "${POSTGRES_PORT:-"5432"}" | sed 's/\r$//')
DB_NAME=$(echo "${POSTGRES_DB:-"metawebthree"}" | sed 's/\r$//')

DB_URL="postgresql://${DB_USER}:${DB_PASS}@${DB_HOST}:${DB_PORT}/${DB_NAME}"

CLICKHOUSE_HOST=$(echo "${CLICKHOUSE_HOST:-"localhost"}" | sed 's/\r$//')
CLICKHOUSE_PORT=$(echo "${CLICKHOUSE_PORT:-"8123"}" | sed 's/\r$//')

echo "=========================================="
echo "   MetaWebThree Database Initializer"
echo "=========================================="
echo "PostgreSQL: ${DB_HOST}:${DB_PORT}/${DB_NAME}"
echo "ClickHouse: ${CLICKHOUSE_HOST}:${CLICKHOUSE_PORT}"

# ------------------------------------------------------------------
# PostgreSQL Initialization
# ------------------------------------------------------------------
if command -v psql &> /dev/null; then
    # Ensure target database exists (connect to 'postgres' to create target DB)
    if [ "$DB_NAME" != "postgres" ]; then
        ADMIN_DB_URL="postgresql://${DB_USER}:${DB_PASS}@${DB_HOST}:${DB_PORT}/postgres"
        echo "Checking if database '$DB_NAME' exists..."
        DB_EXISTS=$(psql "$ADMIN_DB_URL" -tAc "SELECT 1 FROM pg_database WHERE datname='$DB_NAME'")
        if [ "$DB_EXISTS" != "1" ]; then
            echo "Creating database '$DB_NAME'..."
            psql "$ADMIN_DB_URL" -c "CREATE DATABASE \"$DB_NAME\""
            if [ $? -ne 0 ]; then
                echo "Error: Failed to create database '$DB_NAME'."
                exit 1
            fi
            echo "Database '$DB_NAME' created successfully."
        else
            echo "Database '$DB_NAME' already exists."
        fi
    fi

    # Confirm destructive operation unless -y is passed
    if [ "$SKIP_CONFIRM" = false ]; then
        echo ""
        echo "⚠️ WARNING: This will DROP ALL TABLES in database '$DB_NAME' and re-create them."
        read -p "Are you sure? (yes/no): " CONFIRM
        if [ "$CONFIRM" != "yes" ]; then
            echo "Aborted."
            exit 0
        fi
    fi

    # Clean existing schema (drop all tables) to allow idempotent re-run
    echo "Cleaning existing database schema..."
    psql "$DB_URL" -c "DROP SCHEMA IF EXISTS public CASCADE; CREATE SCHEMA public;"
    if [ $? -eq 0 ]; then
        echo "Schema cleaned ✅"
    else
        echo "Warning: Schema cleanup failed, tables may already exist."
    fi

    # Execute PostgreSQL schema files
    for sql_file in $(ls temp/*_schema.sql 2>/dev/null); do
        if grep -q "ENGINE\s*=" "$sql_file" 2>/dev/null; then
            continue
        fi
        echo "------------------------------------------"
        echo "Importing: $sql_file"
        psql "$DB_URL" -f "$sql_file" --set ON_ERROR_STOP=1
        if [ $? -eq 0 ]; then
            echo "Success ✅"
        else
            echo "Failed ❌"
        fi
    done
else
    echo "psql not found, skipping PostgreSQL initialization."
fi

# ------------------------------------------------------------------
# ClickHouse Initialization
# ------------------------------------------------------------------
echo ""
echo "=========================================="
echo "   ClickHouse Initialization"
echo "=========================================="

if command -v curl &> /dev/null; then
    for sql_file in $(ls temp/*_schema.sql 2>/dev/null); do
        if ! grep -q "ENGINE\s*=" "$sql_file" 2>/dev/null; then
            continue
        fi
        echo "------------------------------------------"
        echo "Executing: $sql_file"
        if curl -s -X POST "http://${CLICKHOUSE_HOST}:${CLICKHOUSE_PORT}/" \
            --data-binary @"$sql_file" > /dev/null 2>&1; then
            echo "Success ✅"
        else
            echo "Failed ❌ (is ClickHouse running?)"
            echo "  To run manually: curl -X POST 'http://${CLICKHOUSE_HOST}:${CLICKHOUSE_PORT}/' --data-binary @${sql_file}"
        fi
    done
else
    echo "curl not found, skipping ClickHouse initialization."
fi

echo "=========================================="
echo "Initialization finished."
