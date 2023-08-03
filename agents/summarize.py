import openai
from utils import exponential_backoff_on_fail, log_api_call, open_file
from constants import OPENAI_API_KEY, SUMMARIZE_MODEL, SUMMARIZE_SYS, SYSTEM_MSG_PATH, MAX_RETRIES
import tiktoken
import tokenizers
from tokenizers import Tokenizer
from tokenizers import CharBPETokenizer

openai.api_key = OPENAI_API_KEY
tokenizer = Tokenizer.from_pretrained("bert-base-cased")


def get_word_count(text):
    tokenizer = Tokenizer.from_pretrained("bert-base-cased")
    encoding = tokenizer.encode(text)
    tokens = encoding.tokens
    return len(tokens)


@exponential_backoff_on_fail(max_retries=MAX_RETRIES)
def call_summarize_api(system, conversation):
    # Include the system message as part of the conversation
    conversation = [{'role': 'system', 'content': system}] + conversation
    response = openai.ChatCompletion.create(
        model=SUMMARIZE_MODEL,
        messages=conversation
    )
    return response


def format_conversation(conversation):
    formatted_conversation = ""
    for message in conversation:
        formatted_conversation += f"{message['role'].upper()}: {message['content']}\n"
    return formatted_conversation


def summarize_agent(conversation, current_summary):
    # Load system message
    system_message = open_file(SYSTEM_MSG_PATH + SUMMARIZE_SYS)

    # Calculate word count
    word_count = get_word_count(current_summary)

    # Update system message with word count and current summary
    system_message = system_message.replace("<<TOKENS>>", str(word_count))
    system_message = system_message.replace("<<SUMMARY>>", current_summary)

    # Format conversation
    formatted_conversation = format_conversation(conversation)

    # Call API
    response = call_summarize_api(system_message, [{'role': 'system', 'content': system_message}, {
                                  'role': 'user', 'content': formatted_conversation}])

    # Log API call
    log_api_call("summarize", {"system": system_message,
                 "conversation": conversation}, response)

    # Return new summary
    return response['choices'][0]['message']['content']
