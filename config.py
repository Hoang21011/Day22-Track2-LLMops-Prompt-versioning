import os
from dotenv import load_dotenv

# Load .env file
load_dotenv()

def get_config():
    """
    Returns the configuration from environment variables.
    """
    config = {
        "LANGCHAIN_TRACING_V2": os.getenv("LANGCHAIN_TRACING_V2", "true"),
        "LANGCHAIN_API_KEY": os.getenv("LANGCHAIN_API_KEY"),
        "LANGCHAIN_PROJECT": os.getenv("LANGCHAIN_PROJECT", "day22-lab-rag"),
        "LANGCHAIN_ENDPOINT": os.getenv("LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com"),
        
        "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY"),
        "OPENAI_BASE_URL": os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
        "OPENAI_MODEL_NAME": os.getenv("OPENAI_MODEL_NAME", "gpt-4o"),
        
        "EMBEDDING_MODEL_NAME": os.getenv("EMBEDDING_MODEL_NAME", "text-embedding-3-small"),
    }
    return config

if __name__ == "__main__":
    conf = get_config()
    print("="*40)
    print("📋 Current Configuration")
    print("="*40)
    for k, v in conf.items():
        if "KEY" in k and v:
            print(f"{k:25}: {'*'*len(v[:10])}...{v[-4:]}")
        else:
            print(f"{k:25}: {v}")
    
    missing = [k for k, v in conf.items() if v is None and "KEY" in k]
    if missing:
        print("\n❌ Missing required environment variables:", missing)
        print("Please add them to your .env file.")
    else:
        print("\n✅ Config loaded successfully")
