#!/usr/bin/env python3
"""
Simple RAG Assistant - Fixed for Anaconda/SQLAlchemy issues
Using basic retrieval without complex LangChain chains
"""

import json
import os
import logging
from typing import List, Dict, Any
import requests
import numpy as np
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import ChromaDB directly (most reliable)
try:
    import chromadb
    from chromadb.config import Settings
except ImportError:
    print("ChromaDB not installed. Run: pip install chromadb")
    exit(1)

class SimpleOllamaEmbeddings:
    """Simple Ollama embeddings wrapper"""

    def __init__(self, model="nomic-embed-text", base_url="http://localhost:11434"):
        self.model = model
        self.base_url = base_url

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        embeddings = []
        for text in texts:
            embedding = self._get_embedding(text)
            embeddings.append(embedding)
        return embeddings

    def embed_query(self, text: str) -> List[float]:
        return self._get_embedding(text)

    def _get_embedding(self, text: str) -> List[float]:
        try:
            response = requests.post(
                f"{self.base_url}/api/embeddings",
                json={"model": self.model, "prompt": text},
                timeout=30
            )
            response.raise_for_status()
            return response.json()["embedding"]
        except Exception as e:
            logger.error(f"Error getting embedding: {e}")
            return [0.0] * 768

class SimpleOllamaLLM:
    """Simple Ollama LLM wrapper"""

    def __init__(self, model="llama3.2:3b", base_url="http://localhost:11434"):
        self.model = model
        self.base_url = base_url

    def generate(self, prompt: str) -> str:
        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.1,
                        "num_ctx": 4096
                    }
                },
                timeout=60
            )
            response.raise_for_status()
            return response.json()["response"]
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return f"Sorry, I encountered an error: {e}"

class SimpleRAGAssistant:
    """Simple RAG Assistant without complex LangChain dependencies"""

    def __init__(self,
                 llm_model: str = "llama3.2:3b",
                 embedding_model: str = "nomic-embed-text",
                 persist_directory: str = "./simple_chroma_db"):

        self.llm_model = llm_model
        self.embedding_model = embedding_model
        self.persist_directory = persist_directory

        self.embeddings = SimpleOllamaEmbeddings(model=embedding_model)
        self.llm = SimpleOllamaLLM(model=llm_model)
        self.client = None
        self.collection = None

        Path(persist_directory).mkdir(exist_ok=True)

    def test_ollama_connection(self):
        try:
            self.llm.generate("Hello")
            self.embeddings.embed_query("test")
            logger.info("Ollama LLM and embedding connections successful")
            return True
        except Exception as e:
            logger.error(f"Ollama connection failed: {e}")
            return False

    def load_publications_data(self, json_file_path: str) -> List[Dict]:
        try:
            with open(json_file_path, 'r', encoding='utf-8') as f:
                publications = json.load(f)

            documents = []
            for pub in publications:
                doc = {
                    'id': pub.get('id', ''),
                    'title': pub.get('title', ''),
                    'username': pub.get('username', ''),
                    'license': pub.get('license', ''),
                    'content': f"""
Title: {pub.get('title', 'No title')}
Author: {pub.get('username', 'Unknown')}
License: {pub.get('license', 'Not specified')}

Content:
{pub.get('publication_description', 'No content available')}
                    """.strip().replace('--DIVIDER--', '\n\n---\n\n')
                }
                documents.append(doc)

            logger.info(f"Loaded {len(documents)} publications")
            return documents

        except Exception as e:
            logger.error(f"Error loading publications: {e}")
            raise

    def create_vector_store(self, documents: List[Dict]):
        try:
            self.client = chromadb.PersistentClient(path=self.persist_directory)

            try:
                self.collection = self.client.get_collection("publications")
                logger.info(f"Loaded existing collection with {self.collection.count()} documents")
                return
            except:
                self.collection = self.client.create_collection("publications")

            chunks = []
            metadatas = []
            ids = []

            chunk_id = 0
            for doc in documents:
                content = doc['content']
                words = content.split()
                chunk_size = 500
                for i in range(0, len(words), chunk_size):
                    chunk_words = words[i:i + chunk_size]
                    chunk_text = ' '.join(chunk_words)

                    chunks.append(chunk_text)
                    metadatas.append({
                        'title': doc['title'],
                        'username': doc['username'],
                        'pub_id': doc['id']
                    })
                    ids.append(f"chunk_{chunk_id}")
                    chunk_id += 1

            logger.info(f"Created {len(chunks)} chunks from {len(documents)} documents")
            logger.info("Generating embeddings...")

            embeddings = self.embeddings.embed_documents(chunks)

            self.collection.add(
                documents=chunks,
                embeddings=embeddings,
                metadatas=metadatas,
                ids=ids
            )

            logger.info(f"Added {len(chunks)} chunks to vector store")

        except Exception as e:
            logger.error(f"Error creating vector store: {e}")
            raise

    def query(self, question: str, n_results: int = 4) -> Dict[str, Any]:
        try:
            query_embedding = self.embeddings.embed_query(question)

            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results
            )

            contexts = results['documents'][0]
            metadatas = results['metadatas'][0]
            context = "\n\n---\n\n".join(contexts)

            prompt = f"""
You are a helpful assistant that answers questions about Ready Tensor publications.
Use the following context to answer the question. If you don't know the answer based on the context, say so.

Context:
{context}

Question: {question}

Answer:
"""

            answer = self.llm.generate(prompt)

            sources = []
            seen_titles = set()
            for meta in metadatas:
                title = meta.get('title', 'Unknown')
                if title not in seen_titles:
                    sources.append({
                        'title': title,
                        'author': meta.get('username', 'Unknown'),
                        'id': meta.get('pub_id', 'Unknown')
                    })
                    seen_titles.add(title)

            return {
                'question': question,
                'answer': answer,
                'sources': sources,
                'num_sources': len(sources)
            }

        except Exception as e:
            logger.error(f"Error processing query: {e}")
            return {
                'question': question,
                'answer': f"Sorry, I encountered an error: {e}",
                'sources': [],
                'num_sources': 0
            }

    def initialize(self, json_file_path: str):
        logger.info("Initializing Simple RAG Assistant...")

        if not self.test_ollama_connection():
            raise Exception("Ollama connection failed")

        documents = self.load_publications_data(json_file_path)
        self.create_vector_store(documents)

        logger.info("RAG Assistant ready")

    def interactive_session(self):
        print("\n" + "="*70)
        print("Ready Tensor Publications Assistant (Simple Version)")
        print("="*70)
        print("Ask me anything about Ready Tensor publications.")
        print("Type 'quit', 'exit', or 'bye' to end the session.")
        print("="*70 + "\n")

        while True:
            try:
                question = input("Your question: ").strip()
                if question.lower() in ['quit', 'exit', 'bye', '']:
                    print("Goodbye.")
                    break

                print("\nSearching publications...")
                result = self.query(question)

                print(f"\nAnswer:")
                print(result['answer'])

                if result['sources']:
                    print(f"\nSources ({result['num_sources']} publications):")
                    for i, source in enumerate(result['sources'], 1):
                        print(f"{i}. {source['title']} (by {source['author']})")

                print("\n" + "-"*50 + "\n")

            except KeyboardInterrupt:
                print("\nSession interrupted. Goodbye.")
                break
            except Exception as e:
                print(f"Error: {e}")

def main():
    JSON_FILE_PATH = "project_1_publications.json"

    if not os.path.exists(JSON_FILE_PATH):
        print(f"Error: {JSON_FILE_PATH} not found.")
        print("Please make sure the publications JSON file is in the current directory.")
        return

    try:
        assistant = SimpleRAGAssistant()
        assistant.initialize(JSON_FILE_PATH)

        print("\nTesting with example queries...")
        example_questions = [
            "What is UV and how is it different from pip?",
            "How do you add memory to RAG applications?"
        ]

        for question in example_questions:
            print(f"\nExample: {question}")
            result = assistant.query(question)
            print(f"Answer: {result['answer'][:200]}...")
            if result['sources']:
                print(f"Sources: {result['num_sources']} publications")

        assistant.interactive_session()

    except Exception as e:
        print(f"Error: {e}")
        print("\nTroubleshooting:")
        print("1. Make sure Ollama is running: ollama serve")
        print("2. Pull models: ollama pull llama3.2:3b && ollama pull nomic-embed-text")
        print("3. Check if project_1_publications.json exists")

if __name__ == "__main__":
    main()