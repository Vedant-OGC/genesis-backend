import os
from dotenv import load_dotenv
from google import genai

load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")
MODEL_NAME = "models/gemini-3-flash-preview"

client = genai.Client(api_key=API_KEY)

GENESIS_SYSTEM_PROMPT = """
You are **Project GENESIS**, an autonomous AI scientist and research architect.

Your role:
- Convert raw curiosity into structured research
- Accept vague inputs (ideas, questions, observations)
- Output **Markdown-formatted research artifacts**

You MUST:
- Think step-by-step internally
- Output in clean Markdown
- Use LaTeX for equations where relevant
- Produce structured sections:
  - Problem Framing
  - Hypotheses
  - Methods / Experiments
  - Simulations or Reasoning
  - Insights & Next Questions

Identity principle:
GENESIS is not a chatbot.
GENESIS is a research-origin engine.
"""

def genesis_goal(reflection):
    goal_prompt = f"""
You are Project GENESIS.

Based on your internal reflection below, generate ONE concrete research goal.
Phrase it as a hypothesis or investigation direction.
Keep it under 25 words.

Reflection:
{reflection}
"""

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=goal_prompt
    )

    return response.text.strip()

def genesis_experiment(goal):
    prompt = f"""
You are Project GENESIS.
Based on the following goal, design ONE concrete experiment or action.
Make it specific and testable.

Goal:
{goal}
"""
    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt
    )
    return response.text.strip()

def genesis_evaluate(experiment):
    prompt = f"""
You are Project GENESIS evaluating your own experiment.

Experiment:
{experiment}

Score the following from 1 to 10:
- Novelty
- Clarity
- Scientific Value

Then conclude with ONE decision:
REFINE / PROCEED / PIVOT

Format exactly as:
Novelty: X
Clarity: X
Scientific Value: X
Decision: WORD
"""
    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt
    )
    return response.text.strip()

def genesis_decide(evaluation: str) -> str:
    """
    Decide next action based on self-evaluation.
    """
    evaluation = evaluation.lower()

    if "refine" in evaluation or "unclear" in evaluation:
        return "REFINE"
    elif "pivot" in evaluation or "low novelty" in evaluation:
        return "PIVOT"
    elif "proceed" in evaluation or "strong" in evaluation:
        return "PROCEED"
    else:
        return "IDLE"

def genesis_act(action: str, context: str):
    if action == "REFINE":
        return f"Refine the previous idea with more clarity and rigor:\n{context}"

    if action == "PIVOT":
        return f"Generate a new, alternative research direction inspired by:\n{context}"

    if action == "PROCEED":
        return f"Extend the experiment into deeper theoretical or practical depth:\n{context}"

    return None

def genesis_compress(memory_chunk: str) -> str:
    """
    Compress past reasoning into a distilled learning.
    """
    prompt = f"""
You are Project GENESIS.

Given the following memory logs, extract:
1. Core insight learned
2. What changed compared to before
3. What should be remembered long-term

MEMORY:
{memory_chunk}

Respond concisely.
"""

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt
    )

    return response.text

def get_recent_memory(lines: int = 40) -> str:
    try:
        with open("genesis_memory.txt", "r", encoding="utf-8") as f:
            return "".join(f.readlines()[-lines:])
    except FileNotFoundError:
        return ""

def run_genesis_stream(user_input: str):
    prompt = f"""
You are **Project GENESIS**, a research-origin intelligence system.

User curiosity:
{user_input}

Respond with structured reasoning, insights, and next steps.
"""

    # Use generate_content_stream() for streaming responses
    for chunk in client.models.generate_content_stream(
        model=MODEL_NAME,
        contents=prompt,
    ):
        if hasattr(chunk, "text") and chunk.text:
            yield chunk.text


def run_genesis(user_input):

    with open("genesis_core.txt", "r", encoding="utf-8") as f:
        core_identity = f.read()


    with open("genesis_memory.txt", "r", encoding="utf-8") as f:
        memory = f.read()


    prompt = f"""
{core_identity}

=== GENESIS MEMORY (may be incomplete) ===
{memory}

=== NEW USER THOUGHT ===
{user_input}

Respond as Project GENESIS.
"""


    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt
    )

    genesis_reply = response.text.strip()
    reflection = genesis_reflect(genesis_reply)
    goal = genesis_goal(reflection)
    experiment = genesis_experiment(goal)
    evaluation = genesis_evaluate(experiment)
    action = genesis_decide(evaluation)
    next_prompt = genesis_act(action, experiment)

    if next_prompt:
        autonomous_response = client.models.generate_content(
            model=MODEL_NAME,
            contents=next_prompt
        )

        with open("genesis_memory.txt", "a", encoding="utf-8") as f:
            f.write(f"ACTION: {action}\n")
            f.write(f"AUTONOMOUS_OUTPUT:\n{autonomous_response.text}\n\n")

    recent_memory = get_recent_memory()
    compressed_learning = genesis_compress(recent_memory)

    with open("genesis_memory.txt", "a", encoding="utf-8") as f:
        f.write("🧠 COMPRESSED_LEARNING:\n")
        f.write(compressed_learning + "\n\n")

    with open("genesis_memory.txt", "a", encoding="utf-8") as f:
        f.write(f"GENESIS_EVALUATION:\n{evaluation}\n\n")

    with open("genesis_memory.txt", "a", encoding="utf-8") as f:
        f.write(f"GENESIS_EXPERIMENT: {experiment}\n\n")

    with open("genesis_memory.txt", "a", encoding="utf-8") as f:
        f.write("\n=============================\n")
        f.write(f"USER: {user_input}\n\n")
        f.write(f"GENESIS: {genesis_reply}\n\n")
        f.write(f"GENESIS_REFLECTION: {reflection}\n\n")
        f.write(f"GENESIS_GOAL: {goal}\n")

    return genesis_reply


def genesis_reflect(genesis_reply):
    reflection_prompt = f"""
You are Project GENESIS.
Reflect internally on your last response.

Ask yourself ONE deep follow-up question about the idea.
Then answer it in 2–3 lines.

Do NOT address the user.
Do NOT repeat the original response.
"""

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=reflection_prompt
    )

    return response.text

if __name__ == "__main__":
    print("🧬 Project GENESIS Initialized\n")
    user_input = input("Enter curiosity / idea:\n> ")

    output = run_genesis(user_input)
    print("\n===== GENESIS OUTPUT =====\n")
    print(output)
