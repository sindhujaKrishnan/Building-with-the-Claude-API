# Load env variables and create client
from dotenv import load_dotenv
from anthropic import Anthropic
import os
import json

load_dotenv()
api_key = os.getenv("SINDHU_ANTHROPIC_KEY")
client = Anthropic(api_key=api_key)
model = "claude-sonnet-5"

# Helper functions
def add_messages(messages, text, role="user"):
    message = {"role": role, "content": text}
    messages.append(message)


def chat(messages, system=None, temperature=1.0, stop_sequences=[]):
    params = {
        "model": model,
        "max_tokens": 1000,
        "messages": messages,
        "temperature": temperature,
        "stop_sequences": stop_sequences,
    }

    if system:
        params["system"] = system

    message = client.messages.create(**params)
    #return message.content[0].text
    # Filter for text blocks in case thinking blocks are present
    text_blocks = [block.text for block in message.content if block.type == "text"]
    return "\n".join(text_blocks)


def extract_json(text):
    """Strip markdown code fences if the model adds them despite instructions."""
    text = text.strip()
    if text.startswith("```"):
        # Remove opening fence (```json or ```) and closing fence
        text = text.split("\n", 1)[1] if "\n" in text else text
        if text.endswith("```"):
            text = text.rsplit("```", 1)[0]
    return text.strip()


def generate_dataset():
    prompt = """
Generate an evaluation dataset for a prompt evaluation. The dataset will be used to evaluate prompts
that generate Python, JSON, or Regex specifically for AWS-related tasks. Generate an array of JSON objects,
each representing a task that requires Python, JSON, or a Regex to complete.

Example output:
```json
[
    {
        "task": "Description of task"
    },
    ...additional
]
```

* Focus on tasks that can be solved by writing a single Python function, a single JSON object, or a regular expression.
* Focus on tasks that do not require writing much code
* Respond with ONLY the raw JSON array. Do not include markdown code fences, backticks, or any other text before or after the JSON.

Please generate 3 objects.
"""

    messages = []
    add_messages(messages, prompt, role="user")
    # Note: assistant message prefill (e.g. seeding a "```json" assistant turn)
    # is not supported by this model, so we rely on the prompt instructions
    # above plus a defensive fence-stripper instead.
    text = chat(messages)  # no stop_sequences needed now — nothing to prefill against
    cleaned = extract_json(text)
    return json.loads(cleaned)

dataset = generate_dataset()

# Save dataset to file
with open("dataset.json", "w") as f:
    json.dump(dataset, f, indent=4)

print("Dataset saved to dataset.json")

def run_prompt(test_case):
    """Merges the prompt and test case input, then returns the result"""
    prompt = f"""
Please solve the following task:

{test_case["task"]}
"""
    
    messages = []
    add_messages(messages, prompt, role="user")
    output = chat(messages)
    return output

def run_eval(dataset):
    results = []
    for test_case in dataset:
        result = run_prompt(test_case)
        results.append(result)
    return results

with open("dataset.json", "r") as f:
    dataset = json.load(f)

results = run_eval(dataset)

print(json.dumps(results, indent=4))