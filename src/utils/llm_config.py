"""
LLM Configuration Module for Flexible Model Support

Supports both OpenAI and Ollama models with easy switching via environment variables.
"""

import os
from typing import Any, Optional

def get_llm_instance(
    model_name: Optional[str] = None,
    temperature: float = 0.0,
    max_tokens: Optional[int] = None,
    **kwargs
) -> Any:
    """
    Get LLM instance based on model configuration.
    
    Supports:
    - OpenAI models (gpt-4, gpt-3.5-turbo, etc.)
    - Ollama models (llama3.1:8b, mistral:7b, etc.)
    
    Args:
        model_name: Model name (if None, uses CHAT_MODEL from env)
        temperature: Generation temperature
        max_tokens: Maximum tokens to generate
        **kwargs: Additional model-specific parameters
    
    Returns:
        LLM instance (ChatOpenAI or ChatOllama)
    """
    
    # Get model from env if not provided
    if model_name is None:
        model_name = os.getenv('CHAT_MODEL', 'gpt-4o-mini')
    
    # Clean model name
    model_name = model_name.strip('"')
    
    print(f"🤖 Initializing LLM: {model_name}")
    
    # Check if it's an Ollama model (contains ':' or is in known ollama models)
    ollama_indicators = [
        ':',  # llama3.1:8b, mistral:7b
        'llama', 'mistral', 'codellama', 'mixtral', 'phi3',
        'gemma', 'qwen', 'vicuna', 'orca'
    ]
    
    is_ollama = any(indicator in model_name.lower() for indicator in ollama_indicators)
    
    if is_ollama:
        # Use Ollama
        try:
            from langchain_ollama import ChatOllama
            
            print(f"🦙 Using Ollama model: {model_name}")
            
            llm = ChatOllama(
                model=model_name,
                temperature=temperature,
                base_url=os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434'),
                **kwargs
            )
            
            # Test connection
            try:
                response = llm.invoke("Hello")
                print(f"✅ Ollama connection successful")
            except Exception as e:
                print(f"⚠️  Ollama connection test failed: {e}")
                print("   Make sure Ollama is running: 'ollama serve'")
            
            return llm
            
        except ImportError:
            print("❌ langchain-ollama not installed. Install with: pip install langchain-ollama")
            raise
            
    else:
        # Use OpenAI
        try:
            from langchain_openai import ChatOpenAI
            
            print(f"🔑 Using OpenAI model: {model_name}")
            
            # Set max_tokens based on model if not provided
            if max_tokens is None:
                if 'gpt-4' in model_name:
                    max_tokens = 4096
                else:
                    max_tokens = 2048
            
            llm = ChatOpenAI(
                model_name=model_name,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs
            )
            
            return llm
            
        except Exception as e:
            print(f"❌ OpenAI initialization failed: {e}")
            raise

def get_embedding_model(model_name: Optional[str] = None) -> Any:
    """
    Get embedding model instance.
    
    Args:
        model_name: Embedding model name (if None, uses EMBEDDING_MODEL from env)
    
    Returns:
        Embedding model instance
    """
    
    if model_name is None:
        model_name = os.getenv('EMBEDDING_MODEL', 'text-embedding-3-small')
    
    model_name = model_name.strip('"')
    
    print(f"🔍 Initializing embedding model: {model_name}")
    
    # Check if it's an Ollama embedding model
    if ':' in model_name or any(x in model_name.lower() for x in ['nomic', 'mxbai']):
        try:
            from langchain_ollama import OllamaEmbeddings
            
            return OllamaEmbeddings(
                model=model_name,
                base_url=os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434')
            )
        except ImportError:
            print("❌ langchain-ollama not installed for embeddings")
            raise
    else:
        # Use OpenAI embeddings
        from langchain_openai import OpenAIEmbeddings
        
        return OpenAIEmbeddings(model=model_name)

# Convenience functions for common configurations
def get_factory_llm() -> Any:
    """Get LLM optimized for factory/manufacturing queries."""
    return get_llm_instance(temperature=0.1)

def get_classification_llm() -> Any:
    """Get LLM optimized for classification tasks."""
    return get_llm_instance(temperature=0.0)

def get_creative_llm() -> Any:
    """Get LLM optimized for creative/generation tasks."""
    return get_llm_instance(temperature=0.7)

# Model recommendations
RECOMMENDED_MODELS = {
    'fast': {
        'openai': 'gpt-3.5-turbo',
        'ollama': 'phi3:mini'
    },
    'balanced': {
        'openai': 'gpt-4o-mini',
        'ollama': 'llama3.1:8b'
    },
    'powerful': {
        'openai': 'gpt-4o',
        'ollama': 'mixtral:8x7b'
    },
    'technical': {
        'openai': 'gpt-4o',
        'ollama': 'codellama:7b'
    }
}

def print_model_recommendations():
    """Print recommended models for different use cases."""
    print("\n🎯 Recommended Models:")
    print("=" * 50)
    
    for use_case, models in RECOMMENDED_MODELS.items():
        print(f"\n{use_case.upper()}:")
        print(f"  OpenAI: {models['openai']}")
        print(f"  Ollama: {models['ollama']}")
    
    print("\n💡 Tips:")
    print("  • For testing: Use phi3:mini or llama3.1:8b")
    print("  • For production: Use gpt-4o-mini or mistral:7b")
    print("  • For technical docs: Use codellama:7b")
    print("  • Set CHAT_MODEL in .env to switch models")

if __name__ == "__main__":
    print_model_recommendations()
