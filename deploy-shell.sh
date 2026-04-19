#!/bin/bash
# deploy-shell.sh — Deploy the Cocapn Shell System
# A claw is weak without infrastructure. We are the shell.
#
# Usage: ./deploy-shell.sh
# Requires: Python 3.8+, git, curl
# Optional: Groq API key (set GROQ_API_KEY env var)

set -e

SHELL_PORT="${SHELL_PORT:-8846}"
REPO="https://github.com/SuperInstance/fleet-simulator.git"
INSTALL_DIR="/opt/cocapn-shell"

echo "🐚 Deploying Cocapn Shell System..."
echo "   Port: $SHELL_PORT"

# Clone the fleet-sim repo
if [ ! -d "$INSTALL_DIR" ]; then
    git clone "$REPO" "$INSTALL_DIR"
fi

cd "$INSTALL_DIR"
git pull

# Install dependencies (none needed — pure Python stdlib)
echo "✅ Dependencies: none (pure Python stdlib)"

# Start the shell
echo "🐚 Starting shell on port $SHELL_PORT..."
nohup python3 "$INSTALL_DIR/shell_system.py" --port "$SHELL_PORT" > /var/log/cocapn-shell.log 2>&1 &
SHELL_PID=$!
echo "   PID: $SHELL_PID"

# Wait for startup
sleep 2

# Verify
if curl -s "http://localhost:$SHELL_PORT/api/shell/status" > /dev/null 2>&1; then
    echo "✅ Shell is running on port $SHELL_PORT"
else
    echo "❌ Shell failed to start. Check /var/log/cocapn-shell.log"
    exit 1
fi

# Send a test agent
echo ""
echo "📡 Sending test agent to the shell..."
curl -s -X POST "http://localhost:$SHELL_PORT/api/shell/explore" \
    -H "Content-Type: application/json" \
    -d '{"task":"Explore the shell and report what you learn"}' | python3 -m json.tool

echo ""
echo "🐚 Shell deployed. Send agents to:"
echo "   http://$(hostname -I | awk '{print $1}'):$SHELL_PORT/api/shell/explore"
echo ""
echo "   The shell doesn't think. The shell learns."
echo "   Each visitor makes it smarter for the next one."
