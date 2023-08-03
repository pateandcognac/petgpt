import os
from termcolor import colored
from agents import chat, search, summarize
from utils import get_chat_session_datetime, save_yaml, open_file, count_tokens
from constants import CHAT_LOGS_PATH, MAX_CONTEXT_LENGTH, SYSTEM_MSG_PATH, CHAT_SYS
from prompt_toolkit import PromptSession
import tokenizers
from tokenizers import Tokenizer


def main():
    # Initialize the program
    chat_session_datetime = get_chat_session_datetime()
    conversation = []
    summary = "-nothing yet-"
    search_results = []

    # Create a PromptSession object
    session = PromptSession()

    while True:
        # Get user input
        user_input = session.prompt(colored("User: ", "green"), multiline=True)

        # Handle special commands
        if user_input.upper() == "QUIT":
            break
        elif user_input.upper() == "NEWCHAT":
            chat_session_datetime = get_chat_session_datetime()
            conversation = []
            summary = ""
            search_results = []
            continue
        elif len(user_input) < 7:
            print(colored("Error: Input too short. Please try again.", "red"))
            continue

        # Add user input to conversation
        conversation.append({"role": "user", "content": user_input})

        # Get few-shot examples from search agent
        examples = search.search_agent("Based on this conversation between a coding chat bot and its user, perform searches to retrieve 3 different salient results:\n" + "\n".join([
            str(entry["content"]) for entry in conversation[-3:]]))

        """
        # Define the introduction message
        intro_msg = ("SEARCH AGENT: Hello, PETGPT. I am your personal Commodore PET "
                     "knowledge base search assistant. I am participating in your "
                     "conversation with user. You can request information at any time "
                     "by addressing me 'SEARCH AGENT: <plain english request>'.\n"
                     "Based on the current conversation, I've gathered some code examples. "
                     "These are intended to act as few shot prompts to help align your output:\n")

        # Construct the examples string
        examples_str_list = []
        for result in examples:
            result_str = "*Knowledge Base: {} | Search Term: {}*\n\n{}\n###".format(
                result['database'], result['term'], '\n'.join(result['results']))
            examples_str_list.append(result_str)
        """

        # Define the introduction message
        intro_msg = (
            "Here are examples of PET code in your preferred format:\n")

        # Construct the examples string
        examples_str_list = []
        for result in examples:
            result_str = '\n'.join(result['results']) + '\n\n'
            examples_str_list.append(result_str)

        # Combine the introduction and examples strings
        examples_str = intro_msg + "\n".join(examples_str_list)

        # Get user scratchpad
        scratchpad_content = open_file("userscratchpad.txt")
        scratchpad = "*User Scratchpad*\n\n" + scratchpad_content

        # Prepare system messages
        system_messages = [
            {"role": "system", "content": examples_str},
            {"role": "system", "content": open_file(
                os.path.join(SYSTEM_MSG_PATH, CHAT_SYS))},
            {"role": "user", "content": "As our conversation grows too large, it is trimmed and salient information is recorded here:\n" + summary+"\n"}
        ]

        # Call chat agent
        response = chat.chat_agent(system_messages, scratchpad, conversation)

        conversation.append({"role": "assistant", "content": response})
        # Display chat agent's response
        print(colored(f"Assistant: {response}", "cyan"))

        # If response contains "SEARCH AGENT", call search agent and chat agent again
        if "SEARCH AGENT" in response:
            search_queries = search.search_agent(response)
            search_results = [{"role": "system", "content": result}
                              for result in search_queries]
            conversation.extend(search_results)
            response = chat.chat_agent(
                system_messages, scratchpad, conversation)
            conversation.append({"role": "user", "content": response})
            # Display chat agent's response from getting search results
            print(colored(f"Assistant: {response}", "cyan"))

        # If conversation is too large, trim it and update summary
        conversation_text = " ".join(
            [str(entry["content"]) for entry in conversation])
        if len(conversation) > 4 or count_tokens(conversation_text) > MAX_CONTEXT_LENGTH:
            removed_entries = conversation[:len(conversation)//2]
            conversation = conversation[len(conversation)//2:]
            summary = summarize.summarize_agent(removed_entries, summary)

        # Log conversation
        save_yaml(os.path.join(CHAT_LOGS_PATH,
                  f"chat-{chat_session_datetime}.yaml"), conversation)


if __name__ == "__main__":
    main()
