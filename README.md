# Ready Tensor RAG Assistant 

A Retrieval-Augmented Generation (RAG) assistant that answers questions about Ready Tensor publications using **Ollama** for local LLM inference.

## Project Overview

This is Project 1 of the Agentic AI Developer Certification Program. It demonstrates:
- **RAG pipeline** using direct Ollama integration
- **Local LLM inference** with Ollama (no API costs!)
- **Vector storage** with ChromaDB
- **Document processing** and retrieval
- **Interactive chat interface**

## Quick Start

### Prerequisites

1. **Python 3.8+** installed
2. **Ollama** installed ([Download here](https://ollama.com/download))
3. **Your existing project_1_publications.json file**

### Setup Instructions

#### Option 1: Global Installation (Recommended for simplicity)
```bash
# Run the global setup script
chmod +x globalSetup.sh
./globalSetup.sh
```

#### Option 2: Virtual Environment (Recommended for isolation)
```bash
# Run the local environment setup script
chmod +x setup.sh
./setup.sh
```

#### Option 3: Manual Setup
```bash
# Install dependencies
pip install chromadb requests numpy

# Start Ollama (in another terminal)
ollama serve

# Pull required models
ollama pull llama3.2:3b
ollama pull nomic-embed-text

# Run the assistant
python rag_assistant.py
```

## How It Works

### Architecture
```
Publications (JSON) → Text Splitting → Embeddings → Vector Store (ChromaDB)
                                                                         ↓
User Question → Similarity Search → Context Retrieval → LLM (Ollama) → Answer
```

### Key Components

1. **Document Processing:**
   - Loads Ready Tensor publications from your existing JSON file
   - Splits content into manageable chunks
   - Creates embeddings using Ollama's nomic-embed-text model

2. **Vector Storage:**
   - Uses ChromaDB for fast similarity search
   - Persists data locally (no cloud dependencies)
   - Retrieves most relevant content for queries

3. **Question Answering:**
   - Uses Ollama's llama3.2:3b for response generation
   - Combines retrieved context with user questions
   - Provides source attribution

## Usage Examples

### Interactive Mode
```
Ready Tensor Publications Assistant
======================================
Your question: What is UV and how does it compare to pip?

Searching publications...

Answer: UV is a next-generation Python package manager built in Rust that's 10-100x faster than pip. It combines virtual environment creation and dependency management in one tool...

Sources (1 publications):
  1. UV: The Next Generation Python Package Manager Built for Speed (by aryan_patil)
```

### Programmatic Usage
```python
from rag_assistant import ReadyTensorRAGAssistant

# Initialize
assistant = ReadyTensorRAGAssistant()
assistant.initialize("project_1_publications.json")

# Query
result = assistant.query("How do you add memory to RAG applications?")
print(result["answer"])
```

## Configuration

### Model Options

You can customize the models used by modifying the initialization:

```python
assistant = ReadyTensorRAGAssistant(
    llm_model="llama3.1:8b",        # Larger, more capable model
    embedding_model="mxbai-embed-large"  # Better embeddings
)
```

**Available LLM Models:**
- `llama3.2:1b` - Fastest, basic responses
- `llama3.2:3b` - Balanced (default)
- `llama3.1:8b` - More capable, slower
- `qwen2.5:3b` - Alternative option

**Available Embedding Models:**
- `nomic-embed-text` - Fast, good quality (default)
- `mxbai-embed-large` - Higher quality, larger
- `all-minilm` - Lightweight option

## Example Queries to Try

1. **Technical Comparisons:**
   - "How does UV compare to pip and poetry?"
   - "What are the differences between CNN and RNN models?"

2. **Best Practices:**
   - "What are the best practices for repository organization?"
   - "How should I structure my AI/ML project?"

3. **Implementation Details:**
   - "How do you add memory to RAG applications?"
   - "What models performed best in time series classification?"

4. **Tools and Frameworks:**
   - "What computer vision models were discussed?"
   - "What are the key features of UV package manager?"

## Troubleshooting

### Common Issues

1. **Ollama not running:**
```bash
# Start Ollama server
ollama serve
```

2. **Models not found:**
```bash
# Check available models
ollama list

# Pull required models
ollama pull llama3.2:3b
ollama pull nomic-embed-text
```

3. **Memory issues:**
```bash
# Use smaller models
ollama pull llama3.2:1b  # Instead of 3b
```

4. **ChromaDB/Dependencies errors:**
```bash
# Reinstall dependencies
pip install --upgrade chromadb requests numpy
```

5. **SQLAlchemy/LangChain conflicts (Anaconda users):**
   - This version bypasses LangChain entirely to avoid conflicts
   - Uses direct HTTP calls to Ollama instead

### Performance Optimization

- **For faster responses:** Use smaller models (`llama3.2:1b`)
- **For better quality:** Use larger models (`llama3.1:8b`)
- **For less memory:** Reduce chunk size in the code
- **For persistence:** Vector store is automatically saved and reloaded

## Project Structure

```
ready_tensor_rag/
├── rag_assistant.py              # Main application
├── project_1_publications.json  # Your knowledge base (existing)
├── setup.sh                     # Local environment setup
├── globalSetup.sh              # Global installation setup
├── requirements.txt            # Dependencies list
├── chroma_db/                  # Vector database (auto-created)
└── rag_env/                    # Virtual environment (if using setup.sh)
```

## Learning Outcomes

By completing this project, you've learned:

**RAG Implementation:** Built a complete retrieval-augmented generation system  
**Local LLM Usage:** Used Ollama for cost-free, private AI inference  
**Vector D# Ready Tensor RAG Assistant 

A Retrieval-Augmented Generation (RAG) assistant that answers questions about Ready Tensor publications using **Ollama** for local LLM inference.

## Project Overview

This is Project 1 of the Agentic AI Developer Certification Program. It demonstrates:
- **RAG pipeline** using LangChain
- **Local LLM inference** with Ollama (no API costs!)
- **Vector storage** with ChromaDB
- **Document processing** and retrieval
- **Interactive chat interface**

## Quick Start

### Prerequisites

1. **Python 3.8+** installed
2. **Ollama** installed ([Download here](https://ollama.com/download))

### Setup Instructions

1. **Clone or create the project directory:**
```bash
mkdir ready_tensor_rag
cd ready_tensor_rag
```

2. **Save the provided files:**
   - Save the main code as `rag_assistant.py`
   - Save the requirements as `requirements.txt`
   - Save the publication data as `project_1_publications.json`

3. **Create the JSON data file:**
```bash
# Create project_1_publications.json with the publication data
# (Copy the JSON data provided in the project description)
```

4. **Run the automated setup:**
```bash
chmod +x setup.sh
./setup.sh
```

Or **manual setup:**
```bash
# Create virtual environment
python3 -m venv rag_env
source rag_env/bin/activate

# Install dependencies
pip install -r requirements.txt

# Start Ollama (in another terminal)
ollama serve

# Pull required models
ollama pull llama3.2:3b
ollama pull nomic-embed-text
```

5. **Run the assistant:**
```bash
python rag_assistant.py
```

## How It Works

### Architecture
```
Publications (JSON) → Text Splitting → Embeddings → Vector Store (Chroma)
                                                                         ↓
User Question → Similarity Search → Context Retrieval → LLM (Ollama) → Answer
```

### Key Components

1. **Document Processing:**
   - Loads Ready Tensor publications from JSON
   - Splits content into manageable chunks
   - Creates embeddings using Ollama

2. **Vector Storage:**
   - Uses ChromaDB for fast similarity search
   - Persists data locally (no cloud dependencies)
   - Retrieves most relevant content for queries

3. **Question Answering:**
   - Uses Ollama's llama3.2:3b for response generation
   - Combines retrieved context with user questions
   - Provides source attribution

## Usage Examples

### Interactive Mode
```
Ready Tensor Publications Assistant
======================================
Your question: What is UV and how does it compare to pip?

Searching publications...

Answer: UV is a next-generation Python package manager built in Rust that's 10-100x faster than pip. It combines virtual environment creation and dependency management in one tool...

Sources (1 publications):
  1. UV: The Next Generation Python Package Manager Built for Speed (by aryan_patil)
```

### Programmatic Usage
```python
from rag_assistant import ReadyTensorRAGAssistant

# Initialize
assistant = ReadyTensorRAGAssistant()
assistant.initialize("project_1_publications.json")

# Query
result = assistant.query("How do you add memory to RAG applications?")
print(result["answer"])
```

## Configuration

### Model Options

You can customize the models used by modifying the initialization:

```python
assistant = ReadyTensorRAGAssistant(
    llm_model="llama3.1:8b",        # Larger, more capable model
    embedding_model="mxbai-embed-large"  # Better embeddings
)
```

**Available LLM Models:**
- `llama3.2:1b` - Fastest, basic responses
- `llama3.2:3b` - Balanced (default)
- `llama3.1:8b` - More capable, slower
- `qwen2.5:3b` - Alternative option

**Available Embedding Models:**
- `nomic-embed-text` - Fast, good quality (default)
- `mxbai-embed-large` - Higher quality, larger
- `all-minilm` - Lightweight option

### Performance Tuning

```python
# Adjust chunk sizes for different trade-offs
assistant.create_vector_store(
    documents, 
    chunk_size=1000,    # Smaller chunks = more precise
    chunk_overlap=100   # Less overlap = faster processing
)

# Adjust retrieval parameters
retriever = vectorstore.as_retriever(
    search_kwargs={"k": 6}  # Retrieve more context
)
```

## Example Queries to Try

1. **Technical Comparisons:**
   - "How does UV compare to pip and poetry?"
   - "What are the differences between CNN and RNN models?"

2. **Best Practices:**
   - "What are the best practices for repository organization?"
   - "How should I structure my AI/ML project?"

3. **Implementation Details:**
   - "How do you add memory to RAG applications?"
   - "What models performed best in time series classification?"

4. **Tools and Frameworks:**
   - "What computer vision models were discussed?"
   - "What are the key features of UV package manager?"

## Troubleshooting

### Common Issues

1. **Ollama not running:**
```bash
# Start Ollama server
ollama serve
```

2. **Models not found:**
```bash
# Check available models
ollama list

# Pull required models
ollama pull llama3.2:3b
ollama pull nomic-embed-text
```

3. **Memory issues:**
```bash
# Use smaller models
ollama pull llama3.2:1b  # Instead of 3b
```

4. **Import errors:**
```bash
# Reinstall dependencies
pip install --upgrade langchain chromadb
```

### Performance Optimization

- **For faster responses:** Use smaller models (`llama3.2:1b`)
- **For better quality:** Use larger models (`llama3.1:8b`)
- **For less memory:** Reduce chunk size and retrieval count
- **For persistence:** Vector store is automatically saved and reloaded

## Project Structure

```
ready_tensor_rag/
├── rag_assistant.py              # Main application
├── requirements.txt              # Python dependencies
├── setup.sh                     # Automated setup script
├── project_1_publications.json  # Knowledge base
├── chroma_db/                   # Vector database (auto-created)
└── rag_env/                     # Virtual environment (auto-created)
```

## Learning Outcomes

By completing this project, you've learned:

**RAG Implementation:** Built a complete retrieval-augmented generation system  
**Local LLM Usage:** Used Ollama for cost-free, private AI inference  
**Vector Databases:** Implemented semantic search with ChromaDB  
**Document Processing:** Chunked and embedded text documents  
**LangChain Integration:** Connected retrieval and generation components  
**Production Practices:** Added logging, error handling, and persistence  

##  Next Steps

**Enhancements you could add:**
- Web interface using Streamlit or Gradio
- Advanced conversation memory
- Multi-modal support (images, PDFs)
- Custom embedding fine-tuning
- API endpoint creation
- Advanced retrieval strategies

**For Module 2:** You'll build on this foundation to create more sophisticated agentic workflows with reasoning and tool usage.

## License

This project is part of the Agentic AI Developer Certification Program. Use for educational purposes.

---

**Need help?** Check the troubleshooting section or review the Ollama documentation at [ollama.com](https://ollama.com).