#!/usr/bin/env bash
# Quickstart script for Aiqon ETA & Delay Propagation Engine

set -e

echo "=========================================================="
echo "🚢 Aiqon AI Logistics Platform | ETA & Delay Propagation"
echo "=========================================================="

# Check for virtual environment
if [ -d ".venv" ]; then
    echo "Activating virtual environment (.venv)..."
    source .venv/bin/activate
elif [ -d "venv" ]; then
    echo "Activating virtual environment (venv)..."
    source venv/bin/activate
else
    echo "Creating virtual environment..."
    python3 -m venv .venv
    source .venv/bin/activate
    echo "Installing dependencies..."
    pip install -r requirements.txt
fi

# Ensure dataset exists
if [ ! -f "data/shipments_dataset.csv" ]; then
    echo "Generating initial synthetic shipment dataset..."
    python data/generator.py
fi

echo "Launching Streamlit Frontend on http://localhost:8501 ..."
streamlit run app.py
