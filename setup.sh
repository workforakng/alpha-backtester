#!/bin/bash
# Alpha Backtester — Termux Setup (clang + system-site-packages)
set -e

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV_DIR="$PROJECT_DIR/venv"

echo "============================================"
echo "  Alpha Backtester — Termux Setup"
echo "============================================"

echo "[1/6] Installing clang compiler + precompiled science packages ..."
pkg update -y 2>/dev/null || true
pkg install -y python python-numpy python-scipy clang libffi

echo "[2/6] Installing pandas + lxml via tur-repo ..."
pkg install -y tur-repo 2>/dev/null || true
pkg install -y python-pandas python-lxml 2>/dev/null || true

echo "[3/6] Creating venv with --system-site-packages ..."
rm -rf "$VENV_DIR"
python3 -m venv "$VENV_DIR" --system-site-packages

source "$VENV_DIR/bin/activate"

echo "[4/6] Upgrading pip ..."
pip install --upgrade pip --quiet

echo "[5/6] Installing yfinance + pandas_ta ..."
# Now that clang is present, cffi and curl_cffi can compile if needed
pip install yfinance pandas_ta --quiet

echo "[6/6] Verifying imports ..."
python3 -c "
import numpy;    print(f'  numpy      : {numpy.__version__}')
import scipy;    print(f'  scipy      : {scipy.__version__}')
import pandas;   print(f'  pandas     : {pandas.__version__}')
import lxml;     print(f'  lxml       : {lxml.__version__}')
import yfinance; print(f'  yfinance   : {yfinance.__version__}')
import pandas_ta; print(f'  pandas_ta  : OK')
print()
print('  All packages verified!')
"

echo ""
echo "============================================"
echo "  Done! Run with:"
echo "    source venv/bin/activate"
echo "    python main.py"
echo "============================================"
