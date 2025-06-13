# Ready Tensor RAG Assistant

A Retrieval-Augmented Generation (RAG) assistant that answers questions about Ready Tensor publications using Ollama for local LLM inference.

## What This Project Does

This is a RAG-based assistant for Ready Tensor publications that allows users to ask natural language questions about AI/ML research papers and articles. The system retrieves relevant content from a knowledge base of publications and generates informative answers using a local language model.

The assistant can answer questions about:
- Machine learning techniques and algorithms
- Software tools and frameworks (like UV, pip, poetry)
- Best practices for AI/ML projects
- Repository organization and documentation
- Computer vision and time series analysis
- Specific implementation details from publications

## How It Works

### Architecture

The system follows a standard RAG (Retrieval-Augmented Generation) pipeline:

1. **Document Processing**: Publications are loaded from JSON format and split into manageable text chunks
2. **Embedding Generation**: Text chunks are converted to vector embeddings using Ollama's nomic-embed-text model
3. **Vector Storage**: Embeddings are stored in ChromaDB for fast similarity search
4. **Query Processing**: User questions are embedded and used to retrieve most relevant document chunks
5. **Answer Generation**: Retrieved context is combined with the user question and sent to Ollama's LLM for response generation

### Models Used

- **LLM**: llama3.2:3b (default) - for generating responses
- **Embeddings**: nomic-embed-text - for creating vector representations
- **Vector Database**: ChromaDB - for storing and searching embeddings

### Retrieval Method

The system uses semantic similarity search where user queries are embedded and compared against stored document embeddings. The top 4 most similar chunks are retrieved and used as context for generating answers.

## Setup Instructions

### Prerequisites

- Python 3.8 or higher
- Ollama installed (download from ollama.com)
- At least 4GB RAM for running models locally

### Installation Steps

#### Option 1: Virtual Environment Setup (Recommended)

```bash
# Make setup script executable
chmod +x setup.sh

# Run automated setup
./setup.sh

# Activate the virtual environment
source rag_env/bin/activate
```

#### Option 2: Global Installation

```bash
# Make setup script executable
chmod +x globalSetup.sh

# Run global setup
./globalSetup.sh
```

#### Option 3: Manual Installation

```bash
# Install Python dependencies
pip install chromadb==0.4.24 requests==2.31.0 numpy==1.26.4

# Start Ollama server (in separate terminal)
ollama serve

# Pull required models
ollama pull llama3.2:3b
ollama pull nomic-embed-text
```

### Environment Setup

The system requires:
- Ollama server running on localhost:11434
- ChromaDB for vector storage (automatically configured)
- project_1_publications.json file in the working directory

## How to Run

### Command Line Interface

Start the interactive assistant:

```bash
python rag_assistant.py
```

The assistant will:
1. Initialize the vector database (first run may take a few minutes)
2. Run example queries to test the system
3. Start an interactive session where you can ask questions

### Programmatic Usage

You can also use the assistant programmatically:

```python
from rag_assistant import ReadyTensorRAGAssistant

# Initialize the assistant
assistant = ReadyTensorRAGAssistant()
assistant.initialize("project_1_publications.json")

# Query the assistant
result = assistant.query("What is UV and how does it compare to pip?")
print(result["answer"])
print("Sources:", result["sources"])
```

### Configuration Options

You can customize the models used:

```python
assistant = ReadyTensorRAGAssistant(
    llm_model="llama3.1:8b",        # Use larger model for better quality
    embedding_model="mxbai-embed-large"  # Use better embeddings
)
```

## Sample Input/Output Examples

### Example 1: Technical Comparison

**Input:**
```
What is UV and how does it compare to pip?
```

**Output:**
```
Answer: UV is a next-generation Python package manager built in Rust that offers 
significant performance improvements over traditional tools like pip. According to 
the Ready Tensor publications, UV is 10-100x faster than pip, poetry, and conda. 

Key differences include:
- Speed: UV uses Rust implementation for much faster package installation
- Integration: Combines virtual environment creation and dependency management
- Compatibility: Maintains compatibility with existing Python packaging standards
- Performance: Dramatically reduces installation times compared to pip

UV is designed to address common challenges in the Python ecosystem such as lengthy 
installation times, dependency conflicts, and complexity of managing environments.

Sources (1 publications):
  1. UV: The Next Generation Python Package Manager Built for Speed (by aryan_patil)
```

### Example 2: Implementation Guide

**Input:**
```
How do you add memory to RAG applications?
```

**Output:**
```
Answer: Adding memory to RAG applications involves creating a system that can 
reference previous conversations and questions. Based on the Ready Tensor 
publications, here's the approach:

1. Database Storage: Create a database to store user questions, AI responses, 
   chat IDs, and user information with timestamps

2. Question Retrieval: Implement a function that retrieves the user's previous 
   questions (typically last 3-5) whenever a new question is asked

3. Contextual Processing: Use an LLM to determine if the current question relates 
   to previous questions. If related, create a standalone question that includes 
   necessary context from the conversation history

4. Implementation: Use MongoDB or similar database with fields like question, 
   response, chatId, userId, and timestamps

The key is creating "standalone questions" that can be understood without prior 
conversation context, then embedding these for vector similarity search.

Sources (1 publications):
  1. How to Add Memory to RAG Applications and AI Agents (by 3rdson)
```

### Example 3: Best Practices

**Input:**
```
What are the best practices for repository organization?
```

**Output:**
```
Answer: According to the Ready Tensor publications, repository organization 
follows a three-tiered framework: Essential, Professional, and Elite levels.

Key practices include:

Essential Level:
- Clear directory structure with logical file separation
- Descriptive naming conventions for files and directories
- Proper .gitignore configuration
- Clearly identifiable entry points

Professional Level:
- Dedicated src/ directories with submodules
- Separate directories for data, configuration, tests, and documentation
- Maintain under 15 files per directory and max 5 levels deep
- Proper environment configuration isolation

The framework emphasizes that good repository structure serves as a visual map 
of your project's architecture, making it easier for users to navigate, 
understand, and contribute to your code.

Sources (1 publications):
  1. The Open Source Repository Guide: Best Practices for Sharing Your AI/ML and Data Science Projects (by ready-tensor)
```

## Troubleshooting

### Common Issues

**Ollama not running:**
```bash
ollama serve
```

**Models not available:**
```bash
ollama list
ollama pull llama3.2:3b
ollama pull nomic-embed-text
```

**Import errors:**
```bash
pip install chromadb requests numpy
```

**Permission issues with setup scripts:**
```bash
chmod +x setup.sh
# or run with: bash setup.sh
```

### Performance Notes

- First run takes longer as it builds the vector database
- Vector database is persisted and reloaded on subsequent runs
- Using smaller models (llama3.2:1b) will be faster but less accurate
- Using larger models (llama3.1:8b) will be slower but more detailed

## Project Structure

```
ready_tensor_rag/
├── rag_assistant.py              # Main application
├── requirements.txt              # Python dependencies
├── setup.sh                     # Virtual environment setup
├── globalSetup.sh               # Global installation setup
├── project_1_publications.json  # Knowledge base
├── chroma_db/                   # Vector database (auto-created)
└── rag_env/                     # Virtual environment (if using setup.sh)
```

## Technical Notes

This implementation avoids LangChain dependencies to prevent SQLAlchemy compatibility issues common in Anaconda environments. It uses direct HTTP calls to Ollama and ChromaDB for reliable operation across different Python setups.