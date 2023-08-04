# PETGPT - Python and OpenAI Language Model-Based Chatbot

PETGPT is a Python-based chatbot that uses OpenAI's language model and ChromaDB vector embeddings search. It consists of three interacting agents: the Chat Agent, Summarize Agent, and Search Agent.

## Project Structure

- `main.py`: The main script.
- `utils.py`: Contains utility functions.
- `constants.py`: Holds all constant values.
- `/agents`: Holds all agents scripts.
    - `search.py`: The Search Agent script.
    - `summarize.py`: The Summarize Agent script.
    - `chat.py`: The Chat Agent script.
    - `/system_messages`: Contains system message files (agent_name.txt).
- `/chat_logs`: Holds chat session logs.
- `/api_logs`: Stores API call logs and responses.

## Program Flow

1. The chat session date-time is logged.
2. Input is processed, checking for control commands: `QUIT`, `NEWCHAT` or input errors.
3. Few-shot examples are retrieved via the Search Agent based on the current conversation.
4. Chat Agent is called with examples, summary, scratchpad, live conversation including search results, and new user input.
5. Chat Agent's response is displayed and processed. If response includes "SEARCH AGENT", Search Agent retrieves information and adds it to the conversation.
6. If the live conversation exceeds size limitations, old entries are sent to the Summarize Agent, and the new summary is generated.
7. The process loops back to take new user input.

## Agents

- **Search Agent**: Uses the `gpt-3.5-turbo-0613` model, invoking OpenAI's function-calling feature and a vector database called ChromaDB. It parses inputs from the Chat Agent, executes multiple searches on ChromaDB, and returns results as a list of dictionaries.
- **Summarize Agent**: Gets automatically invoked when the conversation grows too large. It uses the `gpt-3.5-turbo` model to summarize the chat history, taking conversation snippets and search results, and replacing the current summary with a condensed version.
- **Chat Agent**: Acts as the primary user interface using the `gpt-4-0613` model. The Chat Agent leverages a system message, few-shot examples provided by the Search Agent, a summary of the conversation, the user's scratchpad, and the live conversation as inputs to interact with the user.

## User Interface

Uses `prompt_toolkit` for user interaction, offering rich input capabilities and color-coding for different chat components. The user interface presents the current summary, live conversation, custom search results, and the newest user input box. The Chat Agent serves as the primary interface.

## Error Handling

All agents feature basic error handling and fail gracefully, logging their API calls and responses to `api_logs/agent_name-session-date-time.json`. API requests implement exponential back-off.

## Dependencies

- Python 3.7+
- OpenAI API
- ChromaDB
- prompt_toolkit
- termcolor
- tokenizers
