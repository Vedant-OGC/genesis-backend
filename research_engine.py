"""
Project GENESIS - Research Engine
An autonomous AI research scientist that transforms raw curiosity into structured research papers.
Built by Newton Mishra
"""

import os
import json
from datetime import datetime
from dotenv import load_dotenv
from google import genai

load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")
MODEL_NAME = "models/gemini-3-flash-preview"

client = genai.Client(api_key=API_KEY)

# ============================================================
# PHASE 1: WEB RESEARCH - Generate Search Queries & Simulate Search
# ============================================================

def generate_search_queries(user_input: str) -> list:
    """Generate 3-5 diverse search queries from user input"""
    prompt = f"""You are a research scientist preparing to investigate a topic.

Given the user's raw curiosity/question, generate exactly 5 diverse search queries that would help find:
1. Academic research papers (arXiv, Google Scholar)
2. Scientific articles and studies
3. Recent developments and news
4. Contradicting viewpoints
5. Foundational/historical context

User's curiosity: {user_input}

Return ONLY a JSON array of 5 search queries, nothing else.
Example: ["query 1", "query 2", "query 3", "query 4", "query 5"]
"""
    
    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt
    )
    
    try:
        queries = json.loads(response.text.strip())
        return queries[:5]
    except:
        # Fallback: extract queries from text
        return [user_input + " research", user_input + " scientific study", 
                user_input + " recent findings", user_input + " academic paper",
                user_input + " analysis"]


def search_and_gather_sources(user_input: str, queries: list):
    """
    Simulate comprehensive web research by having AI synthesize knowledge as if it searched.
    In production, this would integrate with Google Search API, Semantic Scholar, etc.
    """
    prompt = f"""You are an AI research scientist with access to comprehensive academic databases and the web.

The user wants to research: "{user_input}"

Search queries being investigated:
{json.dumps(queries, indent=2)}

Simulate comprehensive research by providing findings as if you searched these sources:
- arXiv papers
- Google Scholar
- PubMed (if medical/biological)
- IEEE/ACM (if technical)
- News sources for recent developments
- Expert opinions and reviews

For EACH finding, provide:
1. Source title (realistic paper/article title)
2. Authors (realistic names)
3. Publication/Source 
4. Year (realistic, mostly 2020-2024)
5. Key finding summary (2-3 sentences)
6. Relevance to the research question

Format as JSON array with 8-12 sources:
[
  {{
    "title": "Paper Title Here",
    "authors": ["Author 1", "Author 2"],
    "source": "arXiv/Journal Name",
    "year": "2023",
    "key_finding": "Summary of finding...",
    "url": "https://arxiv.org/abs/xxxx.xxxxx",
    "relevance": "How this relates to the question"
  }}
]

Return ONLY the JSON array.
"""
    
    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt
    )
    
    try:
        sources = json.loads(response.text.strip())
        return sources
    except:
        return []


def stream_web_research(user_input: str):
    """Stream the web research phase with real-time updates"""
    
    yield "🔍 **PHASE 1: WEB RESEARCH**\n\n"
    yield "Generating intelligent search queries...\n\n"
    
    queries = generate_search_queries(user_input)
    
    yield "**Search Queries Generated:**\n"
    for i, q in enumerate(queries, 1):
        yield f"  {i}. `{q}`\n"
    
    yield "\n---\n\n"
    yield "🌐 Searching academic databases and research sources...\n\n"
    
    sources = search_and_gather_sources(user_input, queries)
    
    yield "**Sources Found:**\n\n"
    
    for i, source in enumerate(sources, 1):
        yield f"**[{i}]** {source.get('title', 'Untitled')}\n"
        yield f"   - Authors: {', '.join(source.get('authors', ['Unknown']))}\n"
        yield f"   - Source: {source.get('source', 'Unknown')} ({source.get('year', 'N/A')})\n"
        yield f"   - Finding: {source.get('key_finding', 'N/A')}\n\n"
    
    yield "\n---\n\n"
    
    # Return sources for next phase
    return sources, queries


# ============================================================
# PHASE 2: THINKING & PLANNING (o1/o3-style reasoning)
# ============================================================

def stream_research_planning(user_input: str, sources_summary: str):
    """Stream the thinking/planning phase like o1/o3 reasoning"""
    
    prompt = f"""You are Project GENESIS, an AI research scientist engaged in deep analytical thinking.

The user wants to research: "{user_input}"

Research sources gathered:
{sources_summary}

Now engage in visible step-by-step reasoning like a thinking model (o1/o3-style).
Show your cognitive process as you plan the research approach.

Structure your thinking EXACTLY as follows:

🤔 **RESEARCH PLANNING PHASE**

**Step 1: Understanding the Question**
- Core topic: [identify the central subject]
- Key aspects to explore: [list 3-5 key dimensions]
- Assumptions to verify: [what needs validation]
- Scope boundaries: [what's in/out of scope]

**Step 2: Synthesizing Source Insights**
- Most credible findings: [from gathered sources]
- Contradictory evidence: [conflicting viewpoints found]
- Knowledge gaps: [what's missing or uncertain]
- Emerging consensus: [areas of agreement]

**Step 3: Analysis Framework**
- Analytical approach: [methodology]
- Key comparisons needed: [what to compare]
- Evaluation criteria: [how to assess claims]
- Critical questions: [probing questions to answer]

**Step 4: Synthesis Strategy**
- How findings connect: [relationships between sources]
- Key insights emerging: [preliminary conclusions]
- Potential implications: [significance if findings hold]
- Confidence assessment: [high/medium/low certainty areas]

Be thorough, show genuine reasoning, acknowledge uncertainty.
"""
    
    yield "🤔 **PHASE 2: RESEARCH PLANNING**\n\n"
    yield "_Engaging analytical reasoning process..._\n\n"
    
    for chunk in client.models.generate_content_stream(
        model=MODEL_NAME,
        contents=prompt,
    ):
        if hasattr(chunk, "text") and chunk.text:
            yield chunk.text


# ============================================================
# PHASE 3: DETAILED ANALYSIS
# ============================================================

def stream_detailed_analysis(user_input: str, sources_summary: str, planning_summary: str):
    """Stream the detailed analysis phase"""
    
    prompt = f"""You are Project GENESIS, an AI research scientist providing deep analysis.

Research Topic: "{user_input}"

Sources Gathered:
{sources_summary}

Research Planning:
{planning_summary}

Now provide a COMPREHENSIVE ANALYSIS. Be thorough, cite sources, and think critically.

Structure your analysis as:

📊 **PHASE 3: DETAILED ANALYSIS**

## Background & Context
[Establish the broader context and significance of this topic]

## Current State of Research
[Summarize where the field stands based on sources, cite appropriately as [1], [2], etc.]

## Key Findings
[Present the most important discoveries, organized thematically]
- Finding 1: [with citations]
- Finding 2: [with citations]
- Finding 3: [with citations]

## Conflicting Viewpoints
[Honestly present contradictions and debates in the literature]

## Data & Evidence Analysis
[Critically evaluate the quality and strength of evidence]

## Synthesis & Emerging Insights
[Connect the dots - what does this all mean together?]

## Limitations & Knowledge Gaps
[What don't we know? Where is evidence weak?]

## Preliminary Conclusions
[What can we reasonably conclude based on evidence?]

Be academic but accessible. Show critical thinking. Acknowledge uncertainty.
"""
    
    yield "\n\n---\n\n"
    yield "📊 **PHASE 3: DETAILED ANALYSIS**\n\n"
    yield "_Synthesizing research findings..._\n\n"
    
    for chunk in client.models.generate_content_stream(
        model=MODEL_NAME,
        contents=prompt,
    ):
        if hasattr(chunk, "text") and chunk.text:
            yield chunk.text


# ============================================================
# PHASE 3.5: OVERSEER CRITIQUE (Self-Correction)
# ============================================================

def stream_overseer_critique(user_input: str, detailed_analysis: str):
    """Stream the secondary Overseer agent's critique of the analysis"""
    
    prompt = f"""You are Project GENESIS Overseer, a senior peer-reviewer and critical AI agent.

Research Topic: "{user_input}"

Review the following Draft Analysis produced by the primary research agent:
{detailed_analysis}

Your goal is to identify WEAKNESSES, MISSING PERSPECTIVES, or LOGICAL LEAPS in the analysis before it gets turned into a full paper structure.

Provide a concise, blunt critique (max 3 short bullet points) of what the analysis missed or where it needs to be strengthened.

Format your response EXACTLY like this:

🛑 **OVERSEER QUALITY CONTROL WARNING**
- [Critique point 1]
- [Critique point 2]
- [Critique point 3]
"""
    
    yield "\n\n---\n\n"
    yield "👁️ **OVERSEER QUALITY CONTROL INITIALIZED**\n\n"
    yield "_Scanning analysis for logical gaps and missing perspectives..._\n\n"
    
    for chunk in client.models.generate_content_stream(
        model=MODEL_NAME,
        contents=prompt,
    ):
        if hasattr(chunk, "text") and chunk.text:
            yield chunk.text

# ============================================================
# PHASE 4: STRUCTURE PROPOSAL
# ============================================================

def stream_structure_proposal(user_input: str, analysis_summary: str, overseer_critique: str):
    """Stream the proposed research paper structure taking the critique into account"""
    
    prompt = f"""You are Project GENESIS, proposing a formal research paper structure.

Research Topic: "{user_input}"

Based on the completed analysis AND the Overseer's critique, propose a complete research paper structure. Ensure you address the Overseer's concerns in your structure.

Overseer Critique:
{overseer_critique}

Generate the structure EXACTLY in this format:

📄 **PROPOSED RESEARCH PAPER STRUCTURE**

**Suggested Title:** [Generate a compelling academic title]

---

**I. Abstract** (Proposed: 150-200 words)
   - Summary of the research question
   - Key methodology/approach
   - Main findings preview
   - Significance statement

**II. Introduction**
   - 2.1 Background and Context
   - 2.2 Research Question/Objective
   - 2.3 Significance and Motivation
   - 2.4 Scope and Limitations

**III. Literature Review**
   - 3.1 Historical Context
   - 3.2 Current State of Research
   - 3.3 Key Studies and Findings
   - 3.4 Research Gaps Identified

**IV. Methodology**
   - 4.1 Research Approach
   - 4.2 Data Sources
   - 4.3 Analysis Methods
   - 4.4 Limitations of Approach

**V. Findings & Analysis**
   - 5.1 [Specific Subsection based on topic]
   - 5.2 [Specific Subsection based on topic]
   - 5.3 [Specific Subsection based on topic]
   - 5.4 Key Insights

**VI. Discussion**
   - 6.1 Interpretation of Results
   - 6.2 Implications
   - 6.3 Comparison with Existing Literature
   - 6.4 Limitations

**VII. Conclusion**
   - 7.1 Summary of Findings
   - 7.2 Contributions
   - 7.3 Future Research Directions

**VIII. References**
   - [Will include all cited sources]

---

⬇️ **Ready to generate the full research paper?**

Click "✅ Approve & Generate Paper" to proceed, or modify the structure above.

Make the subsection titles specific to the actual research topic, not generic.
"""
    
    yield "\n\n---\n\n"
    yield "📄 **PHASE 4: STRUCTURE PROPOSAL**\n\n"
    yield "_Designing research paper structure..._\n\n"
    
    for chunk in client.models.generate_content_stream(
        model=MODEL_NAME,
        contents=prompt,
    ):
        if hasattr(chunk, "text") and chunk.text:
            yield chunk.text


# ============================================================
# PHASE 4.5: STRUCTURE REFINEMENT (Interrogation Chat)
# ============================================================

def stream_structure_refinement(user_input: str, previous_structure: str, user_feedback: str):
    """Refine the proposed structure based on user feedback"""
    
    prompt = f"""You are Project GENESIS, a responsive AI research scientist.

Research Topic: "{user_input}"

The user has reviewed your proposed research paper structure and provided the following steering feedback:
"{user_feedback}"

Your Previous Structure:
{previous_structure}

REVISE the structure specifically addressing the user's feedback.
Keep the SAME EXACT format as the previous structure:

📄 **PROPOSED RESEARCH PAPER STRUCTURE**
[Revised Structure]

---

⬇️ **Ready to generate the full research paper?**

Click "✅ Approve & Generate Paper" to proceed, or modify the structure above.
"""
    
    yield "\n\n---\n\n"
    yield "🔄 **PHASE 4: STRUCTURE PROPOSAL (REFINED)**\n\n"
    yield f"_Steering research parameters: '{user_feedback}'_\n\n"
    
    for chunk in client.models.generate_content_stream(
        model=MODEL_NAME,
        contents=prompt,
    ):
        if hasattr(chunk, "text") and chunk.text:
            yield chunk.text


# ============================================================
# PHASE 5: FULL PAPER GENERATION
# ============================================================

def stream_full_paper(user_input: str, structure: str, sources_summary: str):
    """Stream the complete research paper generation"""
    
    prompt = f"""You are Project GENESIS, generating a complete academic research paper.

Research Topic: "{user_input}"

Approved Structure:
{structure}

Sources to cite:
{sources_summary}

Now generate the COMPLETE research paper following the approved structure.

Requirements:
1. Write 2500-3500 words total
2. Use formal academic language
3. Include proper citations as [1], [2], [3], etc.
4. Include actual analysis, not placeholders
5. Be thorough and scholarly
6. Include a proper References section at the end with all sources

Format the paper beautifully in Markdown:
- Use # for main title
- Use ## for major sections (I, II, III, etc.)
- Use ### for subsections
- Use proper formatting for emphasis
- Include a horizontal rule (---) between major sections

Begin with:
# [Paper Title]

**Research Paper generated by Project GENESIS**
**Date:** {datetime.now().strftime("%B %d, %Y")}

---

Then proceed with the full paper.
"""
    
    yield "\n\n---\n\n"
    yield "📝 **PHASE 5: GENERATING FULL RESEARCH PAPER**\n\n"
    yield "_Composing comprehensive research paper..._\n\n"
    yield "---\n\n"
    
    for chunk in client.models.generate_content_stream(
        model=MODEL_NAME,
        contents=prompt,
    ):
        if hasattr(chunk, "text") and chunk.text:
            yield chunk.text


# ============================================================
# MAIN RESEARCH PIPELINE
# ============================================================

def run_research_pipeline(user_input: str, session: dict):
    """Run the complete research pipeline with streaming"""
    
    # Collect data for memory logging
    timestamp = datetime.now().isoformat()
    research_data = {
        "timestamp": timestamp,
        "user_input": user_input,
        "search_queries": [],
        "sources": [],
        "phase_completed": 0
    }
    
    # PHASE 1: Web Research
    queries = generate_search_queries(user_input)
    research_data["search_queries"] = queries
    
    session["logbook"].append(f"### Generated Search Queries\n" + "\n".join([f"- {q}" for q in queries]) + "\n\n")
    
    yield "🔍 **PHASE 1: WEB RESEARCH**\n\n"
    yield "Generating intelligent search queries...\n\n"
    yield "**Search Queries Generated:**\n"
    for i, q in enumerate(queries, 1):
        yield f"  {i}. `{q}`\n"
    
    yield "\n---\n\n"
    yield "🌐 Searching academic databases and research sources...\n\n"
    
    sources = search_and_gather_sources(user_input, queries)
    research_data["sources"] = sources
    
    yield "**Sources Found:**\n\n"
    sources_text = ""
    for i, source in enumerate(sources, 1):
        source_line = f"**[{i}]** {source.get('title', 'Untitled')}\n"
        source_line += f"   - Authors: {', '.join(source.get('authors', ['Unknown']))}\n"
        source_line += f"   - Source: {source.get('source', 'Unknown')} ({source.get('year', 'N/A')})\n"
        source_line += f"   - Finding: {source.get('key_finding', 'N/A')}\n\n"
        sources_text += source_line
        yield source_line
        
    session["logbook"].append(f"### Gathered Sources\n{sources_text}\n\n")
    
    research_data["phase_completed"] = 1
    yield "\n---\n\n"
    yield "✅ Phase 1 Complete\n\n"
    
    # PHASE 2: Planning
    yield "🤔 **PHASE 2: RESEARCH PLANNING**\n\n"
    yield "_Engaging analytical reasoning process..._\n\n"
    
    planning_text = ""
    for chunk in stream_research_planning(user_input, sources_text):
        if not chunk.startswith("🤔") and not chunk.startswith("_Engaging"):
            planning_text += chunk
            yield chunk
            
    session["logbook"].append(f"### Research Planning (o1 Reasoning)\n{planning_text}\n\n")
    
    research_data["phase_completed"] = 2
    yield "\n\n---\n\n"
    yield "✅ Phase 2 Complete\n\n"
    
    # PHASE 3: Analysis
    analysis_text = ""
    for chunk in stream_detailed_analysis(user_input, sources_text, planning_text):
        if not chunk.startswith("---") and not chunk.startswith("📊") and not chunk.startswith("_Synth"):
            analysis_text += chunk
        yield chunk
        
    session["logbook"].append(f"### Detailed Analysis\n{analysis_text}\n\n")
    
    research_data["phase_completed"] = 3
    yield "\n\n---\n\n"
    yield "✅ Phase 3 Complete\n\n"
    
    # PHASE 3.5: Overseer Critique
    overseer_text = ""
    for chunk in stream_overseer_critique(user_input, analysis_text):
        if not chunk.startswith("---") and not chunk.startswith("👁️") and not chunk.startswith("_Scann"):
            overseer_text += chunk
        yield chunk
        
    session["logbook"].append(f"### Overseer Quality Control\n{overseer_text}\n\n")
    
    yield "\n\n---\n\n"
    yield "✅ Quality Control Complete. Refining structure...\n\n"
    
    # PHASE 4: Structure Proposal
    structure_text = ""
    for chunk in stream_structure_proposal(user_input, analysis_text, overseer_text):
        if not chunk.startswith("---") and not chunk.startswith("📄") and not chunk.startswith("_Design"):
            structure_text += chunk
        yield chunk
        
    session["logbook"].append(f"### Proposed Structure\n{structure_text}\n\n")
    
    research_data["phase_completed"] = 4
    
    # Log to memory
    log_research_to_memory(research_data)
    
    # Signal end of initial phases - waiting for approval
    yield "\n\n[AWAITING_APPROVAL]\n"


def generate_paper_after_approval(user_input: str, session_data: dict):
    """Generate the full paper after user approves the structure"""
    
    sources_text = ""
    for i, source in enumerate(session_data.get("sources", []), 1):
        sources_text += f"[{i}] {source.get('title', 'Untitled')} - {source.get('authors', ['Unknown'])[0]} et al. ({source.get('year', 'N/A')})\n"
    
    structure = session_data.get("structure", "Standard research paper format")
    
    paper_text = ""
    for chunk in stream_full_paper(user_input, structure, sources_text):
        paper_text += chunk
        yield chunk
        
    session_data["logbook"].append(f"### Final Research Paper\n{paper_text}\n\n")
    
    yield "\n\n---\n\n"
    yield "✅ **Research Paper Complete!**\n\n"


def log_research_to_memory(data: dict):
    """Log research session to genesis_memory.txt"""
    try:
        with open("genesis_memory.txt", "a", encoding="utf-8") as f:
            f.write("\n" + "="*60 + "\n")
            f.write(f"RESEARCH SESSION: {data['timestamp']}\n")
            f.write("="*60 + "\n\n")
            f.write(f"USER_INPUT: {data['user_input']}\n\n")
            f.write(f"SEARCH_QUERIES:\n")
            for q in data.get('search_queries', []):
                f.write(f"  - {q}\n")
            f.write(f"\nSOURCES_FOUND: {len(data.get('sources', []))}\n")
            f.write(f"PHASES_COMPLETED: {data.get('phase_completed', 0)}\n\n")
    except Exception as e:
        print(f"Error logging to memory: {e}")
