# RAG Chatbot with Ollama and Qdrant

A Python-based Retrieval-Augmented Generation (RAG) chatbot that uses Ollama for embeddings and chat, with Qdrant as the vector database for efficient document retrieval.

## Features

- **Document Ingestion**: Load data from files, directories, or raw text
- **Vector Storage**: Efficient document storage and retrieval using Qdrant
- **Conversational Memory**: Maintains context across multiple interactions
- **Local LLM**: Uses Ollama for both embeddings and chat generation
- **Flexible Data Sources**: Support for single files, directories, or direct text input

## Architecture

This implementation consists of two main flows:

### Data Ingestion Flow
1. Load documents from various sources
2. Split documents into chunks using RecursiveCharacterTextSplitter
3. Generate embeddings using Ollama
4. Store embeddings in Qdrant vector database

### RAG Chatbot Flow
1. Receive user message
2. Retrieve relevant documents from Qdrant
3. Generate context-aware response using Ollama
4. Maintain conversation history with memory

## Prerequisites

- Python 3.8+
- [Ollama](https://ollama.ai/) installed and running
- [Qdrant](https://qdrant.tech/) vector database running

## Installation

1. Clone the repository:
```bash
git clone https://github.com/decagondev/ingest-chat-flow.git
cd ingest-chat-flow
```

2. Install dependencies:
```bash
pip install langchain langchain-community qdrant-client ollama
```

3. Start Ollama (if not already running):
```bash
ollama serve
```

4. Start Qdrant using Docker:
```bash
docker run -p 6333:6333 qdrant/qdrant
```

## Configuration

Update the following constants in the code to match your setup:

```python
OLLAMA_BASE_URL = "http://localhost:11434"
QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "ollama_data"
EMBEDDING_MODEL = "ollama1"  # Replace with your Ollama embedding model
CHAT_MODEL = "ollama1"       # Replace with your Ollama chat model
```

## Usage

### Basic Example

```python
from rag_chatbot import RAGChatbot

rag_bot = RAGChatbot()

# Ingest data
rag_bot.ingest_data(text_data="Your document content here...")

# Setup chat
rag_bot.setup_chat()

# Chat with the bot
response = rag_bot.chat("What is this document about?")
print(response['answer'])
```

### Ingest from File

```python
rag_bot.ingest_data(file_path="path/to/your/document.txt")
```

### Ingest from Directory

```python
rag_bot.ingest_data(directory_path="./documents")
```

### Reset Conversation Memory

```python
rag_bot.reset_memory()
```

## API Reference

### `RAGChatbot`

#### `__init__()`
Initialize the RAG chatbot with Ollama and Qdrant connections.

#### `ingest_data(file_path=None, directory_path=None, text_data=None)`
Ingest data from various sources and store in Qdrant vector store.

**Parameters:**
- `file_path` (str, optional): Path to a single text file
- `directory_path` (str, optional): Path to directory containing text files
- `text_data` (str, optional): Raw text data to ingest directly

**Returns:**
- `int`: Number of document chunks created and stored

#### `setup_chat()`
Initialize the conversational retrieval chain with vector store and memory.

#### `chat(user_message)`
Process a user message and generate a context-aware response.

**Parameters:**
- `user_message` (str): The user's input message/question

**Returns:**
- `dict`: Dictionary containing:
  - `answer` (str): The AI-generated response
  - `source_documents` (list): Retrieved source documents

#### `reset_memory()`
Clear the conversation history from memory.

## Project Structure

```
ingest-chat-flow/
├── rag_chatbot.py       # Main implementation
├── README.md            # This file
└── requirements.txt     # Python dependencies
```

## Configuration Options

### Text Splitter Settings

Adjust chunk size and overlap in the `RecursiveCharacterTextSplitter`:

```python
self.text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,      # Size of each chunk
    chunk_overlap=200     # Overlap between chunks
)
```

### Retrieval Settings

Modify the number of documents retrieved:

```python
retriever=self.vector_store.as_retriever(search_kwargs={"k": 3})  # Retrieve top 3 documents
```

### Temperature Setting

Adjust the LLM creativity:

```python
self.llm = ChatOllama(
    model=CHAT_MODEL,
    base_url=OLLAMA_BASE_URL,
    temperature=0.7  # Range: 0.0 (deterministic) to 1.0 (creative)
)
```

## Troubleshooting

### Ollama Connection Issues
Ensure Ollama is running:
```bash
ollama list  # Check available models
ollama serve # Start Ollama server
```

### Qdrant Connection Issues
Verify Qdrant is accessible:
```bash
curl http://localhost:6333/collections
```

### Model Not Found
Pull the required Ollama models:
```bash
ollama pull llama2  # Or your preferred model
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is open source and available under the MIT License.

## Acknowledgments

- [LangChain](https://github.com/langchain-ai/langchain) for the RAG framework
- [Ollama](https://ollama.ai/) for local LLM capabilities
- [Qdrant](https://qdrant.tech/) for vector database functionality

## Contact

For questions or support, please open an issue on the [GitHub repository](https://github.com/decagondev/ingest-chat-flow).
