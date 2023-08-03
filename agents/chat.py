import openai
from utils import open_file, log_api_call, exponential_backoff_on_fail
from constants import OPENAI_API_KEY, CHAT_MODEL, SYSTEM_MSG_PATH, CHAT_SYS, CHAT_LOGS_PATH

openai.api_key = OPENAI_API_KEY


@exponential_backoff_on_fail(max_retries=5)
def chat_agent(system_messages, scratchpad, conversation):
    # Prepare the message input for the chat model
    message_stack = system_messages + \
        [{"role": "user", "content": scratchpad}] + conversation

    # Call the OpenAI API
    response = openai.ChatCompletion.create(
        model=CHAT_MODEL,
        messages=message_stack,
        max_tokens=2048,
        temperature=0.05,
        # log_level="info",
    )

    # Log the API call
    log_api_call("chat", message_stack, response)

    # Extract the assistant's reply
    assistant_reply = response['choices'][0]['message']['content']

    # Return the assistant's reply
    return assistant_reply
