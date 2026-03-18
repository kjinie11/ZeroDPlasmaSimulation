#!/bin/bash

echo "Setting up Plasma Simulation for Linux..."

# Check if python3 is installed
if ! command -v python3 &> /dev/null; then
    echo "Python3 is not installed. Please install it."
    exit 1
fi

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip and install wheel for better compatibility with old packages
echo "Upgrading pip..."
pip install --upgrade pip
pip install wheel

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Run the simulation
echo "Starting Global Model simulation..."
python3 global_model_sim.py

echo "Starting 1D Plasma simulation..."
python3 plasma_sim.py
