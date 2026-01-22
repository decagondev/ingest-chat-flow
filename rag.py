import os
from langchain_community.vectorstores import Qdrant
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.chat_models import ChatOllama
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from qdrant_client import QdrantClient

OLLAMA_BASE_URL = "http://localhost:11434"
QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "ollama_data"
EMBEDDING_MODEL = "ollama1"
CHAT_MODEL = "ollama1"

class RAGChatbot:
    def __init__(self):
        self.embeddings = OllamaEmbeddings(
            model=EMBEDDING_MODEL,
            base_url=OLLAMA_BASE_URL
        )
        
        self.qdrant_client = QdrantClient(url=QDRANT_URL)
        
        self.vector_store = None
        
        self.llm = ChatOllama(
            model=CHAT_MODEL,
            base_url=OLLAMA_BASE_URL,
            temperature=0.7
        )
        
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="answer"
        )
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        
        self.qa_chain = None
    
    def ingest_data(self, file_path=None, directory_path=None, text_data=None):
        """
        Ingest data from various sources and store in Qdrant vector store.
        
        Args:
            file_path (str, optional): Path to a single text file to ingest
            directory_path (str, optional): Path to directory containing text files
            text_data (str, optional): Raw text data to ingest directly
            
        Returns:
            int: Number of document chunks created and stored
            
        Raises:
            ValueError: If no data source is provided
        """
        documents = []
        
        if file_path:
            loader = TextLoader(file_path)
            documents.extend(loader.load())
        
        if directory_path:
            loader = DirectoryLoader(directory_path, glob="**/*.txt")
            documents.extend(loader.load())
        
        if text_data:
            from langchain.schema import Document
            documents.append(Document(page_content=text_data))
        
        if not documents:
            raise ValueError("No data provided for ingestion")
        
        splits = self.text_splitter.split_documents(documents)
        
        print(f"Split {len(documents)} documents into {len(splits)} chunks")
        
        self.vector_store = Qdrant.from_documents(
            splits,
            self.embeddings,
            url=QDRANT_URL,
            collection_name=COLLECTION_NAME,
            force_recreate=False
        )
        
        print(f"Successfully ingested {len(splits)} chunks into Qdrant")
        return len(splits)
    
    def setup_chat(self):
        """
        Initialize the conversational retrieval chain with vector store and memory.
        
        Connects to existing Qdrant collection if vector store is not already initialized,
        then creates a conversational chain that can retrieve relevant context and
        maintain conversation history.
        """
        if self.vector_store is None:
            self.vector_store = Qdrant(
                client=self.qdrant_client,
                collection_name=COLLECTION_NAME,
                embeddings=self.embeddings
            )
        
        self.qa_chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=self.vector_store.as_retriever(search_kwargs={"k": 3}),
            memory=self.memory,
            return_source_documents=True,
            verbose=True
        )
        
        print("Chat system ready!")
    
    def chat(self, user_message):
        """
        Process a user message and generate a context-aware response.
        
        Args:
            user_message (str): The user's input message/question
            
        Returns:
            dict: Dictionary containing:
                - answer (str): The AI-generated response
                - source_documents (list): List of retrieved source documents used for context
        """
        if self.qa_chain is None:
            self.setup_chat()
        
        response = self.qa_chain({"question": user_message})
        
        return {
            "answer": response["answer"],
            "source_documents": response.get("source_documents", [])
        }
    
    def reset_memory(self):
        """
        Clear the conversation history from memory.
        
        Useful for starting a fresh conversation or clearing context
        from previous interactions.
        """
        self.memory.clear()
        print("Conversation memory cleared")


if __name__ == "__main__":
    rag_bot = RAGChatbot()
    
    print("=== Starting Data Ingestion ===")
    
    sample_text = """
    This is sample data about AI and machine learning.
    Large Language Models are trained on vast amounts of text data.
    They can generate human-like responses and understand context.
    """
    rag_bot.ingest_data(text_data=sample_text)
    
    print("\n=== Starting Chat ===")
    
    rag_bot.setup_chat()
    
    questions = [
        "What is this document about?",
        "Tell me about Large Language Models",
        "What did you just tell me?"
    ]
    
    for question in questions:
        print(f"\nUser: {question}")
        response = rag_bot.chat(question)
        print(f"AI: {response['answer']}")
        
        if response['source_documents']:
            print(f"\n[Sources: {len(response['source_documents'])} documents retrieved]")
