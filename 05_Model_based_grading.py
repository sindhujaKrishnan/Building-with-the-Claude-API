# Load env variables and create client
from dotenv import load_dotenv
from anthropic import Anthropic
import os
import json
from statistics import mean


load_dotenv()
api_key = os.getenv("SINDHU_ANTHROPIC_KEY")
client = Anthropic(api_key=api_key)
model = "claude-sonnet-5"


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------
def add_messages(messages, text, role="user"):
    message = {"role": role, "content": text}
    messages.append(message)


def chat(messages, system=None, temperature=None, stop_sequences=None, max_tokens=1000):
    if stop_sequences is None:
        stop_sequences = []

    params = {
        "model": model,
        "max_tokens": max_tokens,
        "messages": messages,
        "stop_sequences": stop_sequences,
    }

    if system:
        params["system"] = system

    # claude-sonnet-5 has deprecated `temperature` — only send it if the
    # caller explicitly asked for a non-default value.
    if temperature is not None:
        params["temperature"] = temperature

    message = client.messages.create(**params)
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


# ---------------------------------------------------------------------------
# Step 1: Generate the evaluation dataset
# ---------------------------------------------------------------------------
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
    text = chat(messages)
    cleaned = extract_json(text)
    return json.loads(cleaned)


# ---------------------------------------------------------------------------
# Step 2: Run each task through the "prompt under test"
# ---------------------------------------------------------------------------
def run_prompt(test_case):
    """Merges the prompt and test case input, then returns the model's solution."""
    prompt = f"""
Please solve the following task:

{test_case["task"]}
"""
    messages = []
    add_messages(messages, prompt, role="user")
    # Bump max_tokens here so code + explanation isn't cut off mid-response.
    output = chat(messages, max_tokens=1500)
    return output


# ---------------------------------------------------------------------------
# Step 3: Grade each output using a model-as-judge
# ---------------------------------------------------------------------------
def grade_by_model(test_case, output):
    eval_prompt = f"""
You are an expert AWS code reviewer. Your task is to evaluate the following AI-generated solution.

Original Task:
<task>
{test_case["task"]}
</task>

Solution to Evaluate:
<solution>
{output}
</solution>

Output Format
Provide your evaluation as a structured JSON object with the following fields, in this specific order:
- "strengths": An array of 1-3 key strengths
- "weaknesses": An array of 1-3 key areas for improvement
- "reasoning": A concise explanation of your overall assessment
- "score": A number between 1-10

Respond with ONLY the raw JSON object. Do not include markdown code fences, backticks, or any other text before or after the JSON.
"""

    messages = []
    add_messages(messages, eval_prompt, role="user")
    eval_text = chat(messages)
    cleaned = extract_json(eval_text)
    return json.loads(cleaned)


# ---------------------------------------------------------------------------
# Step 4: Tie it all together — run every task, then grade every result
# ---------------------------------------------------------------------------
def run_eval(dataset):
    results = []
    for test_case in dataset:
        output = run_prompt(test_case)
        grade = grade_by_model(test_case, output)
        results.append({
            "task": test_case["task"],
            "output": output,
            "grade": grade,
        })
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # 1. Generate (or reuse) the dataset
    if not os.path.exists("dataset.json"):
        dataset = generate_dataset()
        with open("dataset.json", "w") as f:
            json.dump(dataset, f, indent=4)
        print("Dataset saved to dataset.json")
    else:
        with open("dataset.json", "r") as f:
            dataset = json.load(f)
        print("Loaded existing dataset.json")

    # 2. Run + grade every task
    results = run_eval(dataset)

    # 3. Report
    average_score = mean(r["grade"]["score"] for r in results)
    print(f"\nAverage score: {average_score}\n")
    print(json.dumps(results, indent=2))