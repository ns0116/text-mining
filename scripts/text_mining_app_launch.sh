#!/bin/bash
# Move to the project root directory
cd "$(dirname "$0")/.."

echo "======================================================="
echo "Text Mining Application Launcher for macOS"
echo "======================================================="
echo ""

# Check if .venv exists
if [ ! -d ".venv" ]; then
    echo "Virtual environment (.venv) not found. Creating it..."
    python3 -m venv .venv
    if [ $? -ne 0 ]; then
        echo "Error: Failed to create virtual environment."
        exit 1
    fi
fi

# Activate virtual environment
echo "Activating virtual environment..."
source .venv/bin/activate

# Check if required packages are installed
echo "Checking dependencies..."
python3 -c "import streamlit, spacy, networkx, openpyxl, pandas, plotly, wordcloud, sklearn, scipy" &>/dev/null
if [ $? -ne 0 ]; then
    echo "Installing or updating required libraries from requirements.txt..."
    pip install -r requirements.txt
    if [ $? -ne 0 ]; then
        echo "Error: Failed to install dependencies."
        exit 1
    fi
fi

# Check for GiNZA model
python3 -c "import spacy; spacy.load('ja_ginza')" &>/dev/null
if [ $? -ne 0 ]; then
    echo "Downloading Japanese SpaCy model (ja_ginza)..."
    python3 -m spacy download ja_ginza
fi

echo "Starting Streamlit..."
echo ""
streamlit run src/text_mining_app.py
