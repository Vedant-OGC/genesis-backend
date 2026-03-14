"""
Project GENESIS - API Server
"""

import os
import json
import time
from typing import Optional, Dict, Any

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

load_dotenv()


from research_engine import (
    run_research_pipeline,
    generate_paper_after_approval,
    stream_structure_refinement,
)

app = FastAPI(title="Project GENESIS - API Research Scientist")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory session storage 
research_sessions: Dict[str, Any] = {}


class ResearchRequest(BaseModel):
    idea: str
    session_id: Optional[str] = None


class ApprovalRequest(BaseModel):
    session_id: str
    approved: bool
    modified_structure: Optional[str] = None
    feedback: Optional[str] = None


def encode_sse_data(text: str) -> str:
    """Encode text for SSE by replacing newlines with a special marker"""
    return text.replace('\n', '\\n').replace('\r', '')


@app.get("/")
def root():
    return {"status": "ok", "message": "Project GENESIS API is running"}


@app.post("/genesis/research/start")
def start_research(req: ResearchRequest):
    """Start a new research session and stream the initial phases"""
    
    session_id = f"session_{int(time.time() * 1000)}"
    
    # Initialize session
    research_sessions[session_id] = {
        "user_input": req.idea,
        "sources": [],
        "structure": "",
        "status": "in_progress",
        "phase": 0,
        "logbook": [f"## [Phase 0] Session Initialized: {req.idea}\n\n"]
    }
    
    def event_generator():
        structure_text = ""
        in_structure_phase = False
        current_phase = 0
        
        for chunk in run_research_pipeline(req.idea, research_sessions[session_id]):
            # Track phases
            if "PHASE 1:" in chunk or "WEB RESEARCH" in chunk:
                current_phase = 1
                yield f"data: [PHASE:1]\n\n"
            elif "PHASE 2:" in chunk or "RESEARCH PLANNING" in chunk:
                current_phase = 2
                yield f"data: [PHASE:2]\n\n"
            elif "PHASE 3:" in chunk or "DETAILED ANALYSIS" in chunk:
                current_phase = 3
                yield f"data: [PHASE:3]\n\n"
            elif "PHASE 4:" in chunk or "STRUCTURE PROPOSAL" in chunk:
                current_phase = 4
                in_structure_phase = True
                yield f"data: [PHASE:4]\n\n"
            
            if in_structure_phase:
                structure_text += chunk
            
            # Check for awaiting approval signal
            if "[AWAITING_APPROVAL]" in chunk:
                research_sessions[session_id]["structure"] = structure_text
                research_sessions[session_id]["status"] = "awaiting_approval"
                research_sessions[session_id]["phase"] = 4
                
                yield f"data: [SESSION_ID:{session_id}]\n\n"
                yield f"data: [SHOW_APPROVAL]\n\n"
            else:
                # Encode the chunk to preserve newlines
                encoded = encode_sse_data(chunk)
                yield f"data: {encoded}\n\n"
            
            time.sleep(0.01)
        
        yield "data: [DONE]\n\n"
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Session-ID": session_id
        }
    )


@app.post("/genesis/research/approve")
def approve_structure(req: ApprovalRequest):
    """Approve the structure and generate the full paper"""
    
    session = research_sessions.get(req.session_id)
    
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    if not req.approved:
        research_sessions[req.session_id]["status"] = "cancelled"
        return {"status": "cancelled"}
    
    # Update structure if modified
    if req.modified_structure:
        session["structure"] = req.modified_structure
    
    # If the user provided feedback, we refine the structure instead of generating the final paper
    if req.feedback:
        session["status"] = "awaiting_approval"
        
        def refinement_generator():
            yield f"data: [PHASE:4]\n\n"
            
            structure_text = ""
            for chunk in stream_structure_refinement(
                session["user_input"], 
                session["structure"], 
                req.feedback
            ):
                structure_text += chunk
                encoded = encode_sse_data(chunk)
                yield f"data: {encoded}\n\n"
                time.sleep(0.01)
                
            session["structure"] = structure_text
            session["logbook"].append(f"### Structure Refinement (User Steering)\n**User Feedback:** {req.feedback}\n**New Structure:**\n{structure_text}\n\n")
            
            yield f"data: [SESSION_ID:{req.session_id}]\n\n"
            yield f"data: [SHOW_APPROVAL]\n\n"
            yield "data: [DONE]\n\n"
            
        return StreamingResponse(
            refinement_generator(),
            media_type="text/event-stream"
        )
    
    # Otherwise, generate the full paper
    session["status"] = "generating_paper"
    
    def event_generator():
        yield f"data: [PHASE:5]\n\n"
        
        for chunk in generate_paper_after_approval(
            session["user_input"],
            session
        ):
            encoded = encode_sse_data(chunk)
            yield f"data: {encoded}\n\n"
            time.sleep(0.01)
        
        research_sessions[req.session_id]["status"] = "complete"
        yield "data: [DONE]\n\n"
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream"
    )


@app.get("/genesis/session/{session_id}")
def get_session(session_id: str):
    """Get session status and data"""
    session = research_sessions.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    return session


@app.get("/genesis/research/log/{session_id}")
def get_research_logbook(session_id: str):
    """Get the raw scientific logbook for the research session"""
    session = research_sessions.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    logbook_entries = session.get("logbook", [])
    if not logbook_entries:
        return {"logbook": "No logbook entries found for this session."}
        
    return {"logbook": "".join(str(entry) for entry in logbook_entries)}
