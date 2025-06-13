#!/bin/bash

# setup.sh - Setup script for Ready Tensor RAG Assistant

echo "Setting up Ready Tensor RAG Assistant with Ollama..."
echo "============================================================"

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "Python 3 is not installed. Please install Python 3.8 or higher."
    exit 1
fi

# Check if Ollama is installed
if ! command -v ollama &> /dev/null; then
    echo "Ollama is not installed."
    echo "Please install Ollama from: https://ollama.com/download"
    echo "Then run this setup script again."
    exit 1
fi

echo "Python and Ollama found!"

# Create virtual environment
echo "Creating virtual environment..."
python3 -m venv rag_env
source rag_env/bin/activate

# Install requirements
echo "Installing Python dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Check if Ollama is running
echo "Checking Ollama service..."
if ! curl -s http://localhost:11434/api/tags &> /dev/null; then
    echo "Ollama server is not running. Starting Ollama..."
    echo "Please run 'ollama serve' in another terminal, then continue."
    read -p "Press Enter when Ollama is running..."
fi

# Pull required models
echo "Pulling required Ollama models..."
echo "This may take a few minutes depending on your internet connection..."

echo "Pulling llama3.2:3b (LLM model)..."
ollama pull llama3.2:3b

echo "Pulling nomic-embed-text (embedding model)..."
ollama pull nomic-embed-text

# Verify models are available
echo "Verifying models..."
ollama list

echo ""
echo "Setup complete!"
echo "============================================================"
echo "To run the RAG Assistant:"
echo "1. Activate the virtual environment: source rag_env/bin/activate"
echo "2. Make sure you have project_1_publications.json in the current directory"
echo "3. Run: python rag_assistant.py"
echo ""
echo "Alternative models you can try:"
echo "- LLM models: llama3.1:8b, llama3.2:1b, qwen2.5:3b"
echo "- Embedding models: mxbai-embed-large, all-minilm"
echo ""
echo "To use different models, modify the model names in the Python script."
echo "============================================================"