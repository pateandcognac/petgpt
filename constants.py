import os

# OpenAI API Key
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

# ChromaDB Settings
CHROMADB_PERSIST_DIR = "chromadb"
CHROMADB_IMPL = "duckdb+parquet"

# Model Names
CHAT_MODEL = "gpt-4-0613"
SEARCH_MODEL = "gpt-3.5-turbo-0613"
SUMMARIZE_MODEL = "gpt-4"

# File Paths
SYSTEM_MSG_PATH = "agents/system_messages/"
CHAT_LOGS_PATH = "chat_logs/"
API_LOGS_PATH = "api_logs/"

# File Names
CHAT_SYS = "chat.txt"
SUMMARIZE_SYS = "summarize.txt"
SEARCH = "search.txt"

# Other Constants
MAX_RETRIES = 7
MAX_CONTEXT_LENGTH = 1000
MAX_SUMMARY_WORDS = 1000
