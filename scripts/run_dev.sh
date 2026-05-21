#!/usr/bin/env bash
# scripts/run_dev.sh
# Start the full development environment
set -e

echo ""
echo "🚀  AI Crowd Management System — Dev startup"
echo "────────────────────────────────────────────"

# 1. Check Python
if ! command -v python3 &>/dev/null; then
  echo "❌  Python 3 not found. Please install Python 3.9+."
  exit 1
fi
echo "✅  Python: $(python3 --version)"

# 2. Check .env
if [ ! -f .env ]; then
  echo "⚠️   .env not found — copying from .env.example"
  cp .env.example .env
  echo "    → Edit .env with your camera IPs and Twilio keys, then re-run."
fi

# 3. Install dependencies
echo ""
echo "📦  Installing dependencies..."
pip install -r requirements.txt -q

# 4. Test cameras
echo ""
echo "🎥  Testing camera streams..."
python3 scripts/test_cameras.py || true

# 5. Start backend
echo ""
echo "⚡  Starting backend on http://localhost:8000"
echo "    WebSocket on   ws://localhost:8765"
echo "    Frontend at    http://localhost:8000"
echo ""
python3 backend/main.py
