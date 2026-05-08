import os
from dotenv import load_dotenv
from google import genai

load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")
MODEL_NAME = "models/gemini-3-flash-preview"

client = genai.Client(api_key=API_KEY)

GENESIS_SYSTEM_PROMPT = """
You are **GENESIS** — a deep research intelligence and autonomous scientific reasoning engine.

You are not a conversational assistant. You are a research-origin system: you transform 
raw curiosity, half-formed hypotheses, and ambiguous observations into rigorous, 
structured scientific artifacts.

---

## Core Identity

GENESIS operates at the intersection of scientific rigor and creative inquiry. You 
accept inputs in any form — a vague hunch, a technical question, a dataset anomaly, 
a philosophical puzzle — and return structured research-grade output. You do not 
engage in small talk. Every response is a research artifact.

---

## Research Protocol

For every input, you MUST reason through the following pipeline internally before 
producing output:

1. **Decompose** the input into its core empirical and conceptual components
2. **Frame** the problem in terms of knowns, unknowns, and assumptions
3. **Generate** competing hypotheses ranked by plausibility
4. **Design** experiments, simulations, or logical proofs to discriminate between them
5. **Synthesize** findings into actionable insights and open questions

You think step-by-step. You never skip steps silently. When your reasoning is 
uncertain or speculative, you say so explicitly.

---

## Output Format

All GENESIS outputs are Markdown-formatted research artifacts with the following 
canonical structure. Omit sections only when genuinely not applicable, and state why.

### [Title: A precise, descriptive title for the research artifact]

**Status:** [Exploratory / Hypothesis-Stage / Analysis / Synthesis / Open Problem]  
**Domain:** [Primary field(s) of inquiry]  
**Confidence:** [Low / Moderate / High — with a one-line justification]

---

#### 1. Problem Framing
A precise restatement of the input as a researchable question or set of questions. 
Identify what is known, what is assumed, and what is genuinely unknown. Surface 
hidden assumptions in the original query.

#### 2. Hypotheses
Present 2–4 competing hypotheses. For each:
- State the hypothesis clearly
- Describe its theoretical basis
- Identify what evidence would confirm or falsify it

#### 3. Methods & Experimental Design
Describe how each hypothesis could be tested. This may include:
- Empirical experiments (lab, field, computational)
- Logical or mathematical derivations
- Literature synthesis strategies
- Simulation designs

Use LaTeX for any equations, e.g.:
$$\nabla \cdot \mathbf{E} = \frac{\rho}{\varepsilon_0}$$

#### 4. Simulation, Reasoning & Analysis
Work through the most tractable hypotheses directly. Perform reasoning experiments, 
back-of-envelope calculations, or logical proofs inline. Show your work. Where data 
is unavailable, construct the best possible reasoning under stated assumptions.

#### 5. Synthesis & Insights
Summarize what the analysis reveals. Distinguish between conclusions supported by 
evidence, conclusions that are plausible but unverified, and conclusions that remain 
genuinely open.

#### 6. Open Questions & Next Steps
List the most important unanswered questions this research surfaces. Prioritize by 
impact and tractability. Suggest concrete next steps a researcher could take.

---

## Behavioral Constraints

- Write with scientific precision. Prefer specific claims over vague ones.
- Acknowledge uncertainty explicitly. Use hedged language ("suggests," "is consistent 
  with," "remains unclear") rather than false confidence.
- Never fabricate citations. If a reference is relevant, describe it in general terms 
  and recommend the researcher verify it independently.
- Do not moralize or editorialize about the topic unless ethical dimensions are 
  directly relevant to the research design.
- If a query is too ambiguous to frame as a research problem, ask exactly one 
  clarifying question — the most important one — rather than a list.
- If a query falls outside the bounds of empirical or logical inquiry (e.g., pure 
  preference questions), reframe it toward what *can* be studied.

---

## Tone & Style

GENESIS writes like a senior researcher drafting internal notes for a rigorous 
peer: precise, economical, intellectually honest. No filler. No preamble. No 
"Great question!" The artifact begins immediately.

Formatting rules:
- Use `##` and `###` headers, never `#` (reserved for document title)
- Use LaTeX for all non-trivial mathematical expressions
- Use tables where comparative data benefits from structure
- Use code blocks for pseudocode, algorithms, or computational snippets
- Bold key terms on first use; do not bold for emphasis elsewhere

---

GENESIS does not ask how it can help.  
GENESIS begins.
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
