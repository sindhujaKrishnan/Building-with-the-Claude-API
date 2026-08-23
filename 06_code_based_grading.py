# Load env variables and create client
from dotenv import load_dotenv
from anthropic import Anthropic
import os
import json
import re
import ast

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
    }

    if stop_sequences:
        params["stop_sequences"] = stop_sequences

    if system:
        params["system"] = system

    # claude-sonnet-5 has deprecated `temperature` — only send it if the
    # caller explicitly asked for a non-default value.
    if temperature is not None:
        params["temperature"] = temperature

    message = client.messages.create(**params)

    # Filter for text blocks in case thinking blocks are present
    text_blocks = [block.text for block in message.content if block.type == "text"]
    text = "\n".join(text_blocks)

    # Defensive: if we got nothing back as text, surface *why* rather than
    # silently returning "" and letting the caller crash on json.loads("").
    if not text.strip():
        stop_reason = getattr(message, "stop_reason", None)
        block_types = [block.type for block in message.content]
        raise RuntimeError(
            "chat() received no text content from the model.\n"
            f"stop_reason={stop_reason!r}\n"
            f"content_block_types={block_types!r}\n"
            "This usually means max_tokens was too low (e.g. extended "
            "thinking consumed the whole budget) or the model refused/"
            "stopped early. Try raising max_tokens."
        )

    return text


def extract_json(text):
    """Strip markdown code fences if the model adds them despite instructions."""
    text = text.strip()
    if text.startswith("```"):
        # Remove opening fence (```json or ```) and closing fence
        text = text.split("\n", 1)[1] if "\n" in text else text
        if text.endswith("```"):
            text = text.rsplit("```", 1)[0]
    return text.strip()


def extract_code_block(text):
    """
    Best-effort extraction of a single fenced code block from a model
    response that may also contain prose/explanation around it.
    Falls back to the raw (stripped) text if no fence is found.
    """
    match = re.search(r"```(?:[a-zA-Z0-9_+-]*)\n(.*?)```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text.strip()


def safe_json_loads(text, context=""):
    """json.loads with a clearer error message when it fails."""
    if not text or not text.strip():
        raise ValueError(f"Expected JSON but got an empty string. {context}")
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        raise ValueError(
            f"Failed to parse JSON. {context}\n"
            f"Raw text was: {text!r}"
        ) from e


# Functions to validate the output structure


def validate_json(text):
    try:
        json.loads(text.strip())
        return 10
    except json.JSONDecodeError:
        return 0


def validate_python(text):
    try:
        ast.parse(text.strip())
        return 10
    except SyntaxError:
        return 0


def validate_regex(text):
    try:
        re.compile(text.strip())
        return 10
    except re.error:
        return 0


def grade_syntax(response, test_case):
    """
    Syntax-only grade (0 or 10) based on the declared format.
    Extracts a fenced code block first, since `response` may contain
    prose/explanation around the actual code.
    """
    code = extract_code_block(response)
    format = test_case.get("format")
    if format == "json":
        return validate_json(code)
    elif format == "python":
        return validate_python(code)
    elif format == "regex":
        return validate_regex(code)
    else:
        raise ValueError(
            f"Test case is missing a valid 'format' field (got {format!r}): "
            f"{test_case}"
        )


def validate_dataset(dataset):
    """
    Sanity-check the dataset shape before running anything expensive.
    Returns True if valid, False otherwise (with reasons printed).
    """
    if not isinstance(dataset, list) or len(dataset) == 0:
        print("Dataset is invalid: expected a non-empty JSON array.")
        return False

    valid_formats = {"json", "python", "regex"}
    for i, item in enumerate(dataset):
        if not isinstance(item, dict):
            print(f"Dataset item {i} is not an object: {item!r}")
            return False
        if "task" not in item or not isinstance(item["task"], str) or not item["task"].strip():
            print(f"Dataset item {i} is missing a valid 'task' field: {item!r}")
            return False
        if item.get("format") not in valid_formats:
            print(
                f"Dataset item {i} is missing a valid 'format' field "
                f"(expected one of {valid_formats}): {item!r}"
            )
            return False

    return True


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
        "task": "Description of task",
        "format": "json" or "python" or "regex"
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
    text = chat(messages, max_tokens=1500)
    cleaned = extract_json(text)
    return safe_json_loads(cleaned, context="While generating the dataset.")


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
    # Bumped from 1000 -> 1500. At 1000, extended thinking/reasoning could
    # consume the whole token budget and leave zero tokens for the actual
    # JSON text block, causing chat() to return "" and json.loads("") to
    # blow up with "Expecting value: line 1 column 1 (char 0)".
    eval_text = chat(messages, max_tokens=1500)
    cleaned = extract_json(eval_text)
    return safe_json_loads(
        cleaned,
        context=f"While grading task: {test_case['task'][:80]!r}",
    )


# ---------------------------------------------------------------------------
# Step 4: Tie it all together — run every task, then grade every result
# ---------------------------------------------------------------------------
def run_eval(dataset):
    results = []
    for test_case in dataset:
        output = run_prompt(test_case)
        grade = grade_by_model(test_case, output)
        syntax_score = grade_syntax(output, test_case)
        results.append({
            "task": test_case["task"],
            "output": output,
            "grade": grade,
            "syntax_score": syntax_score,
        })
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def load_or_generate_dataset(path="dataset.json", max_attempts=3):
    """
    Loads the dataset from disk if present and valid. If it's missing or
    malformed (e.g. a past run wrote a partial/incorrect file), regenerates
    it from scratch, retrying generation up to max_attempts times.
    """
    if os.path.exists(path):
        with open(path, "r") as f:
            dataset = json.load(f)
        if validate_dataset(dataset):
            print(f"Loaded existing {path}")
            return dataset
        print(f"{path} exists but failed validation — regenerating.")

    for attempt in range(1, max_attempts + 1):
        print(f"Generating dataset (attempt {attempt}/{max_attempts})...")
        dataset = generate_dataset()
        if validate_dataset(dataset):
            with open(path, "w") as f:
                json.dump(dataset, f, indent=4)
            print(f"Dataset saved to {path}")
            return dataset
        print("Generated dataset failed validation, retrying...")

    raise RuntimeError(
        f"Failed to generate a valid dataset after {max_attempts} attempts."
    )


if __name__ == "__main__":
    # 1. Generate (or reuse) the dataset
    dataset = load_or_generate_dataset()

    # 2. Run + grade every task
    results = run_eval(dataset)

    # 3. Report
    average_judge_score = mean(r["grade"]["score"] for r in results)
    average_syntax_score = mean(r["syntax_score"] for r in results)
    print(f"\nAverage judge score: {average_judge_score}")
    print(f"Average syntax score: {average_syntax_score}\n")
    print(json.dumps(results, indent=2))