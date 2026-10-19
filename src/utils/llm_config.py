"""
LLM Configuration Module for Flexible Model Support

Supports OpenAI, OpenAI-compatible (e.g. IDUN) and Ollama models with easy
switching via environment variables.
"""

import os
import re
from typing import Any, Optional, Tuple

# Tag names reasoning models may leak into `content`
# The tags are built from these names so the patterns stay in sync.
_REASONING_TAG_NAMES = ("think", "thinking", "reasoning", "thought")
_REASONING_BLOCK_RE = re.compile(
    "<(?:" + "|".join(_REASONING_TAG_NAMES) + ")>"
    ".*?"
    "</(?:" + "|".join(_REASONING_TAG_NAMES) + ")>",
    re.DOTALL,
)
_REASONING_OPEN_TAGS = tuple(f"<{name}>" for name in _REASONING_TAG_NAMES)


def get_temperature() -> float:
    """
    Parse TEMPERATURE once, as a float

    All LLM instances in the system use this value.
    """
    try:
        return float(os.getenv("TEMPERATURE", "0.0"))
    except (TypeError, ValueError):
        return 0.0


def get_max_tokens() -> int:
    """
    Single output-token cap for every LLM instance

    Reasoning models can spend a small cap entirely on reasoning and return
    empty content, so the default is generous.
    """
    try:
        return int(os.getenv("LLM_MAX_TOKENS", "16384"))
    except (TypeError, ValueError):
        return 16384


def strip_reasoning(text: Optional[str]) -> Tuple[str, bool]:
    """
    Remove leaked reasoning blocks (e.g.  trimmed-to-fit reasoning tags)
    from model output before it is parsed or returned

    Handles closed blocks, unclosed opening tags, and passes clean text
    through unchanged.

    Returns:
        (clean_text, leaked): the cleaned string and whether anything was removed.
    """
    if not text:
        return text, False
    original = text
    cleaned = _REASONING_BLOCK_RE.sub("", text)
    # Unclosed opening tag: keep only what precedes it.
    for tag in _REASONING_OPEN_TAGS:
        if tag in cleaned:
            cleaned = cleaned.split(tag, 1)[0]
    cleaned = cleaned.strip()
    return cleaned, (cleaned != original)


def get_llm_instance(
    model_name: Optional[str] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    **kwargs
) -> Any:
    """
    Get LLM instance based on model configuration.

    Supports:
    - LLM_PROVIDER=openai_compatible: any OpenAI-compatible endpoint
      (e.g. IDUN) via LLM_BASE_URL + IDUN_API_KEY
    - OpenAI models (gpt-4, gpt-3.5-turbo, etc.) when LLM_PROVIDER is unset
    - Ollama models (llama3.1:8b, mistral:7b, etc.) when LLM_PROVIDER is unset

    Args:
        model_name: Model name (if None, uses CHAT_MODEL from env)
        temperature: Generation temperature (if None, uses TEMPERATURE from env)
        max_tokens: Maximum tokens to generate (if None, uses LLM_MAX_TOKENS from env)
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

    # Normalize temperature/max_tokens
    if temperature is None or isinstance(temperature, str):
        temperature = get_temperature()
    if max_tokens is None:
        max_tokens = get_max_tokens()

    # C1: explicit provider takes precedence. Name guessing is kept
    # only for backward compatibility when LLM_PROVIDER is unset.
    provider = os.getenv('LLM_PROVIDER', '').strip('"').lower()
    if provider == 'openai_compatible':
        from langchain_openai import ChatOpenAI

        print(f"🔑 Using OpenAI-compatible model: {model_name} "
              f"at {os.getenv('LLM_BASE_URL', '')}")

        return ChatOpenAI(
            model=model_name,
            base_url=os.getenv('LLM_BASE_URL'),
            api_key=os.getenv('IDUN_API_KEY'),
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=float(os.getenv('LLM_TIMEOUT_S', '300')),
            max_retries=0,  # retries are done and counted by the harness, never hidden here
        )

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

            # max_tokens already normalized to LLM_MAX_TOKENS above

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
