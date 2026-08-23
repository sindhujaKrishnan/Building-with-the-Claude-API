
from dotenv import load_dotenv
from anthropic import Anthropic
import os
load_dotenv()
api_key = os.getenv("SINDHU_ANTHROPIC_KEY")
client = Anthropic(api_key=api_key)
model = "claude-sonnet-5"
# Helper functions
def add_user_message(messages, text):
    user_message = {
        "role": "user",
        "content": text
        }
    messages.append(user_message)
def add_assistant_message(messages, text):
    assistant_message = {
          "role": "assistant",
          "content": text
        }
    messages.append(assistant_message)
def chat(messages, system_prompt=None):
    params = {
            "model": model,
            "max_tokens": 1000,
            "messages": messages,
            #"temperature": temperature, 
            #not supported when extended thinking is enabled
        }
    if system_prompt:
         params["system"] = system_prompt
    full_text = ""
    # stream.text_stream yields only text deltas (thinking deltas, if any,
    # are excluded automatically), so we don't need to filter block types
    # here the way the non-streaming version had to.
    with client.messages.stream(**params) as stream:
        for text in stream.text_stream:
            print(text, end="", flush=True)
            full_text += text
    print()  # newline after the streamed response finishes
    return full_text
# -----------------------------
# Get system prompt from user
# -----------------------------
system_prompt = input("Enter system prompt: ")
print("\nSystem prompt set successfully.")
print("Type 'exit' to quit.\n")
# Conversation history
messages = []
# -----------------------------
# Chat loop
# -----------------------------
while True:
    user_input = input("You: ")
    if user_input.lower() == "exit":
        break
    add_user_message(messages, user_input)
    print("\nClaude:")
    answer = chat(messages, system_prompt)
    add_assistant_message(messages, answer)
    print("\n--------------------")
