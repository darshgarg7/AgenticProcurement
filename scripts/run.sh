#!/bin/bash
set -e

echo "Setting up Agentic Procurement Environment..."

# 1. Create a virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment (venv)..."
    python3 -m venv venv
fi

# 2. Activate the virtual environment
source venv/bin/activate

# 3. Ensure the required packages are installed
echo "Checking dependencies..."
python3 -m pip install --upgrade pip --quiet
python3 -m pip install -r requirements.txt --quiet
echo "All dependencies installed."

# 4. Run the main program
echo ""
echo "========================================="
echo " Starting The Main Simulation Program    "
echo "========================================="
python3 main.py

echo ""
echo "Simulation Finished Successfully."
