#!/bin/bash
# UNIA OS Web Simulation Runner
# This script runs the web-based simulation of UNIA OS

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WEB_DIR="$SCRIPT_DIR/src/boot/web"

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}======================================${NC}"
echo -e "${BLUE}    UNIA OS Web Simulation Runner     ${NC}"
echo -e "${BLUE}======================================${NC}"

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    if ! command -v python &> /dev/null; then
        echo -e "${RED}Error: Python is not installed.${NC}"
        echo -e "${YELLOW}Please install Python before continuing.${NC}"
        exit 1
    else
        PYTHON_CMD="python"
    fi
else
    PYTHON_CMD="python3"
fi

# Check if the web directory exists
if [ ! -d "$WEB_DIR" ]; then
    echo -e "${RED}Error: Web directory not found at: $WEB_DIR${NC}"
    exit 1
fi

# Navigate to web directory
cd "$WEB_DIR"

# Get a free port (default to 8000 if not possible)
PORT=8000
if command -v lsof &> /dev/null; then
    while lsof -Pi :$PORT -sTCP:LISTEN -t >/dev/null ; do
        PORT=$((PORT+1))
    done
fi

echo -e "${GREEN}Starting UNIA OS web simulation on port $PORT...${NC}"
echo -e "${YELLOW}Open your browser to http://localhost:$PORT${NC}"
echo -e "${YELLOW}Press Ctrl+C to stop the server${NC}"

# Start the HTTP server
$PYTHON_CMD -m http.server $PORT
