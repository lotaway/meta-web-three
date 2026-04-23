#!/bin/bash

# Configuration
DEFAULT_HOST="localhost"
DEFAULT_PORT="10089"
ENDPOINT="/excel/import/url"

# Load environment variables if .env exists (for custom host/port)
if [ -f .env ]; then
    # Look for MEDIA_SERVICE_HOST or similar if defined, else use defaults
    MEDIA_HOST=$(grep MEDIA_SERVICE_HOST .env | cut -d '=' -f2 | sed 's/\r$//')
    MEDIA_PORT=$(grep MEDIA_SERVICE_PORT .env | cut -d '=' -f2 | sed 's/\r$//')
fi

HOST=${MEDIA_HOST:-$DEFAULT_HOST}
PORT=${MEDIA_PORT:-$DEFAULT_PORT}

# Usage help
if [ -z "$1" ]; then
    echo "Usage: $0 <excel_url>"
    echo "Example: $0 https://example.com/artwork_data.xlsx"
    exit 1
fi

EXCEL_URL="$1"

echo "------------------------------------------"
echo "   Media Service Excel Importer"
echo "------------------------------------------"
echo "Target: http://$HOST:$PORT$ENDPOINT"
echo "URL to import: $EXCEL_URL"
echo "------------------------------------------"

# Use curl to call the POST endpoint
# --data-urlencode ensures the URL parameter is properly escaped
response=$(curl -s -X POST "http://$HOST:$PORT$ENDPOINT" \
     --data-urlencode "excelUrl=$EXCEL_URL" \
     -H "Accept: application/json")

# Check if curl succeeded
if [ $? -eq 0 ]; then
    echo "Response from server:"
    echo "$response"
    echo "------------------------------------------"
    echo "Success ✅"
else
    echo "Error: Failed to connect to Media Service at http://$HOST:$PORT"
    exit 1
fi
