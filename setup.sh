#!/bin/bash

# simple_setup.sh - Simplified local setup (no virtual environment)

echo "Setting up Ready Tensor RAG Assistant (Local Installation)..."
echo "================================================================"

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

# Install dependencies globally
echo "Installing Python dependencies globally..."
pip install --upgrade pip
pip install langchain==0.0.352 chromadb==0.4.18 ollama==0.1.7

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
echo "================================================================"
echo "To run the RAG Assistant:"
echo "1. Make sure you have project_1_publications.json in the current directory"
echo "2. Generate the data file: python create_data_file.py"
echo "3. Run the assistant: python rag_assistant.py"
echo ""
echo "No virtual environment needed - everything is installed globally!"
echo "================================================================"