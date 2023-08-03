import yaml
import time
import os
import json
from datetime import datetime
from functools import wraps
import tokenizers
from tokenizers import Tokenizer
from tokenizers import CharBPETokenizer


def save_yaml(filepath, data):
    with open(filepath, 'w', encoding='utf-8') as file:
        yaml.dump(data, file, allow_unicode=True)


def save_file(filepath, content):
    with open(filepath, 'w', encoding='utf-8') as outfile:
        outfile.write(content)


def open_file(filepath):
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as infile:
        return infile.read()


def get_chat_session_datetime():
    return datetime.now().strftime("%Y-%m-%d-%H-%M-%S")


def log_api_call(agent_name, api_call, response):
    session_datetime = get_chat_session_datetime()
    log_data = {
        "api_call": api_call,
        "response": response
    }
    log_file_path = f"api_logs/{agent_name}-{session_datetime}.json"
    with open(log_file_path, 'w') as log_file:
        json.dump(log_data, log_file)


def count_tokens(text):
    tokenizer = Tokenizer.from_pretrained("bert-base-cased")
    encoding = tokenizer.encode(text)
    tokens = encoding.tokens
    return len(tokens)


def exponential_backoff_on_fail(max_retries):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            retry = 0
            while retry < max_retries:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    print(f"Error: {e}")
                    retry += 1
                    time.sleep(2 ** retry)
            raise Exception(f"Failed after {max_retries} retries")
        return wrapper
    return decorator
