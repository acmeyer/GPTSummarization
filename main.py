import argparse
import openai
from dataclasses import asdict
from models import Message
import os
import shutil
import tiktoken
import requests
from unstructured.partition.auto import partition
from bs4 import BeautifulSoup
import dotenv
dotenv.load_dotenv('.env')

openai.api_key = os.getenv("OPENAI_API_KEY")
MODEL_TEMPERATURE = 0.0
OPENAI_EMBEDDING_ENCODING = "cl100k_base" # this the encoding for text-embedding-ada-002
MAX_TOKENS = 2048 # this is the maximum number of tokens for text chunk

def count_tokens(text: str) -> int:
    """Returns the number of tokens in the given text."""
    encoding = tiktoken.get_encoding(OPENAI_EMBEDDING_ENCODING)
    tokens = encoding.encode(text)
    num_tokens = len(tokens)
    return num_tokens, tokens

def chunks(text, n, tokenizer):
    tokens = tokenizer.encode(text)
    """Yield successive n-sized chunks from text."""
    i = 0
    while i < len(tokens):
        # Find the nearest end of sentence within a range of 0.5 * n and 1.5 * n tokens
        j = min(i + int(1.5 * n), len(tokens))
        while j > i + int(0.5 * n):
            # Decode the tokens and check for full stop or newline
            chunk = tokenizer.decode(tokens[i:j])
            if chunk.endswith(".") or chunk.endswith("\n"):
                break
            j -= 1
        # If no end of sentence found, use n tokens as the chunk size
        if j == i + int(0.5 * n):
            j = min(i + n, len(tokens))
        yield tokens[i:j]
        i = j

def split_text(text: str) -> list[str]:
    """Split incoming text and return chunks."""
    tokenizer = tiktoken.get_encoding(OPENAI_EMBEDDING_ENCODING)
    token_chunks = list(chunks(text, MAX_TOKENS, tokenizer))
    text_chunks = [tokenizer.decode(chunk) for chunk in token_chunks]
    return text_chunks

def create_new_file(filepath: str):
    file_name = filepath.split("/")[-1]
    if not os.path.exists("tmp"):
        os.mkdir("tmp")
    file_name = file_name.replace(" ", "_")
    new_file_path = f"tmp/{file_name}"
    # Copy the file to the new location
    shutil.copy(filepath, new_file_path)
    return new_file_path

def ask_gpt_chat(prompt: str, messages: list[Message], model: str = 'gpt-3.5-turbo'):
    """Returns ChatGPT's response to the given prompt."""
    system_message = [{"role": "system", "content": prompt}]
    message_dicts = [asdict(message) for message in messages]
    conversation_messages = system_message + message_dicts
    response = openai.ChatCompletion.create(
        model=model,
        messages=conversation_messages,
        temperature=MODEL_TEMPERATURE
    )
    return response.choices[0]['message']['content'].strip()

def setup_prompt() -> str:
    """Creates a prompt for gpt for generating a response."""
    with open('prompt.md') as f:
        prompt = f.read()
    return prompt

def get_summary(full_text: str, summary_type: str, model: str = 'gpt-3.5-turbo') -> str:
    prompt = setup_prompt()
    conversation_messages = []
    user_input = f"Here's the content you should summarize:\n\n{full_text}\n\n----\n\nI would like you to produce a {summary_type} summary of this content.\n\n----\n\nWhat is the summary of this information?"
    conversation_messages.append(Message(role="user", content=user_input))
    return ask_gpt_chat(prompt, conversation_messages, model=model)

def read_text_from_url(url: str) -> str:
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36'
        }
        response = requests.get(url, headers=headers)
        response.raise_for_status()  # Raise an exception if the request returned an HTTP error
        soup = BeautifulSoup(response.text, 'html.parser')
        text = soup.get_text()
        return text
    except requests.exceptions.RequestException as e:
        print(f"An error occurred while fetching the URL: {e}")
        return None
    
def write_to_file(text: str, filename: str):
    with open(filename, 'w') as f:
        f.write(text)
    print(f"Summary saved to {filename}")


def create_summary(url: str, filepath: str, summary_type: str, model: str) -> str:
    if filepath:
        elements = partition(filepath)
        full_text = "\n\n".join([str(el) for el in elements])
        num_tokens, _ = count_tokens(full_text)
        title = filepath.split("/")[-1]
        title = title.replace(" ", "_")
    elif url:
        full_text = read_text_from_url(url)
        num_tokens, _ = count_tokens(full_text)
        title = url.split("/")[-1]
        if not title or title == '' or title == 'index.html':
            title = url.split("/")[-2]

    if num_tokens > MAX_TOKENS:
        text_chunks = split_text(full_text)
        summaries = []
        for chunk in text_chunks:
            if len(chunk) < 140:
                continue
            summary = get_summary(chunk, summary_type=summary_type, model=model)
            summaries.append(summary)
        all_summaries = "\n\n".join(summaries)
        if summary_type == 'short':
            full_response = get_summary(all_summaries, summary_type=summary_type, model=model)
            write_to_file(full_response, f"summaries/{title}.txt")
            response = full_response
        else:
            write_to_file(all_summaries, f"summaries/{title}.txt")
            response = all_summaries
    else:
        response = get_summary(full_text, summary_type=summary_type, model=model)
        write_to_file(response, f"summaries/{title}.txt")

    return response


# To run directly on command line, run `python main.py`
if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("-u", "--url", type=str, help="URL to summarize")
    argparser.add_argument("-f", "--filepath", type=str, help="Path to file to summarize")
    argparser.add_argument("-m", "--model", type=str, help="Model to use", default="gpt-3.5-turbo")
    argparser.add_argument("-s", "--summary_type", type=str, help="Type of summary to generate", default="short")
    args = argparser.parse_args()
    url = args.url
    summary_type = args.summary_type
    filepath = args.filepath
    model = args.model
    response = create_summary(url, filepath, summary_type, model)
    print(response)
