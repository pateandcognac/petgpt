import openai
import chromadb
import json
from time import time
from chromadb.config import Settings
from utils import exponential_backoff_on_fail, save_yaml
from constants import OPENAI_API_KEY, CHROMADB_PERSIST_DIR, CHROMADB_IMPL, SEARCH_MODEL, SYSTEM_MSG_PATH, API_LOGS_PATH

# Initialize ChromaDB client
chroma_client = chromadb.Client(
    Settings(persist_directory=CHROMADB_PERSIST_DIR, chroma_db_impl=CHROMADB_IMPL))

# Set OpenAI API key
openai.api_key = OPENAI_API_KEY

# Define function descriptor
function_descriptor = {
    "name": "search_database",
    "description": "searches target database for few shot code examples and other information relevant to the task.",
    "parameters": {
            "type": "object",
            "properties": {
                "search_terms": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "search_term": {"type": "string"},
                            'database': {'type': 'string', 'enum': [collection.name for collection in chroma_client.list_collections()]},
                        },
                        "required": ["search_term", "database"]
                    },
                    "minItems": 1
                }
            },
        "required": ["search_terms"]
    }
}


def search_agent(user_message):
    @exponential_backoff_on_fail(max_retries=5)  # Adjust max_retries as needed
    def call_openai_api():
        return openai.ChatCompletion.create(
            model=SEARCH_MODEL,
            messages=[
                {"role": "system", "content": open(SYSTEM_MSG_PATH + "search.txt").read().replace(
                    "<<COLLECTIONS>>", ", ".join([collection.name for collection in chroma_client.list_collections()]))},
                {"role": "user", "content": user_message}
            ],
            functions=[function_descriptor],
            function_call={"name": "search_database"},
        )

    # Call the decorated function
    response = call_openai_api()

    # Parse API response
    function_call = json.loads(
        response['choices'][0]['message']['function_call']['arguments'])
    database_search_term_pairs = function_call['search_terms']

    # Perform searches and collect results
    search_results = []
    for pair in database_search_term_pairs:
        database = pair['database']
        term = pair['search_term']
        collection = chroma_client.get_or_create_collection(name=database)
        results = collection.query(query_texts=[term], n_results=3)
        search_results.append({
            'database': database,
            'term': term,
            'results': results['documents'][0]
        })

    print(search_results)

    # Log API call and response
    save_yaml(API_LOGS_PATH + "search-" + str(time()) + ".yaml",
              {"api_call": function_call, "response": search_results})

    return search_results
