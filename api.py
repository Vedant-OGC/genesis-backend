"""
Project GENESIS - API Server
"""

import os
import json
import time
from typing import Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

load_dotenv()


from research_engine import (
    run_research_pipeline,
    generate_paper_after_approval,
)

app = FastAPI(title="Project GENESIS - API Research Scientist")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory session storage (in production, use Redis or database)
research_sessions = {}


class ResearchRequest(BaseModel):
    idea: str
    session_id: Optional[str] = None


class ApprovalRequest(BaseModel):
    session_id: str
    approved: bool
    modified_structure: Optional[str] = None


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
        "phase": 0
    }
    
    def event_generator():
        structure_text = ""
        in_structure_phase = False
        current_phase = 0
        
        for chunk in run_research_pipeline(req.idea):
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
