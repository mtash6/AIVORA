# ==============================================================================
# APPLICATION: main.py (FastAPI Edition)
# DESCRIPTION: Unified AIVORA Core Operating System Interface Framework REST API.
#              Orchestrates Workforce Routing, Voice/Text Chat, Multi-Agent
#              Meeting Analytics, Semantic Text Extraction, Fiscal Ledgers,
#              Executive Financial Cockpits, and Admin Project Telemetry.
# ==============================================================================

import os
import asyncio
import threading
import tempfile
from contextlib import asynccontextmanager
from typing import List, Optional, Dict, Any
from datetime import datetime
import pandas as pd

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Query, Header, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, EmailStr

import whisper

# ---------------------------------------------------------------------------
# CORE SERVICE LAYER IMPORTS
# ---------------------------------------------------------------------------
from services.route_tasks import CognitiveUnderstandingEngine, IntelligenceDrivenRouter
from services.summarizer import DocumentSummarizerService
from services.meeting_agent import MeetingIntelligenceService
from services.chat import ChatService
from services.finnance import FinanceService
from services.consensus_engine import MultiAgentDecisionEngine
from services.proposal_agent import CorporateCommunicationsAgent
from services.recovery_engine import AutonomousTaskRecoveryEngine
from services.exec_finance import ExecutiveFinanceService
from services.admin_analytics import AdminAnalyticsService
from services.project_manager import ProjectManagerService
from services.employee_manager import EmployeeManagerService, WorkLocation, LeaveType, RoleEnum
from services.scheduler import SchedulerService

# ---------------------------------------------------------------------------
# GLOBAL SERVICE INSTANCES & STATE REGISTERS
# ---------------------------------------------------------------------------
recovery_engine: Optional[AutonomousTaskRecoveryEngine] = None
scheduler_engine: Optional[SchedulerService] = None
whisper_model: Any = None

# Persistent Service Instances
employee_service = EmployeeManagerService()
project_service = ProjectManagerService()
finance_service = FinanceService()
exec_finance_service = ExecutiveFinanceService()
admin_analytics_service = AdminAnalyticsService()


# ---------------------------------------------------------------------------
# LIFECYCLE MANAGEMENT (Startup & Shutdown Tasks)
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    global recovery_engine, scheduler_engine, whisper_model
    
    # 1. Initialize Autonomous Task Recovery Engine in daemon thread
    recovery_engine = AutonomousTaskRecoveryEngine(check_interval_seconds=15)
    
    def run_async_recovery_loop():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(recovery_engine.start())
        
    bg_thread = threading.Thread(target=run_async_recovery_loop, daemon=True)
    bg_thread.start()

    # 2. Initialize and start automated job scheduler
    scheduler_engine = SchedulerService()
    scheduler_engine.start()

    # 3. Load Whisper acoustic model weights
    whisper_model = whisper.load_model("base")

    yield  # Application serves incoming requests

    # Shutdown logic
    if scheduler_engine and scheduler_engine.scheduler.running:
        scheduler_engine.scheduler.shutdown()


# Initialize FastAPI Application
app = FastAPI(
    title="AIVORA Core Operating System API",
    description="Unified Enterprise AI OS Backend API",
    version="2.0.0",
    lifespan=lifespan
)

# Enable CORS for external frontend applications (React, Angular, Mobile Apps)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# HELPER FUNCTIONS & DEPENDENCIES
# ---------------------------------------------------------------------------
def safe_remove(file_path: str):
    """Safely removes temporary backend data files without triggering WinError 32."""
    try:
        if file_path and os.path.exists(file_path):
            os.remove(file_path)
    except PermissionError:
        pass


def get_api_key(x_api_key: Optional[str] = Header(None)) -> str:
    """Retrieves Gemini API Key from header or fallback environment variable."""
    effective_key = x_api_key or os.getenv("GEMINI_API_KEY", "")
    if not effective_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Gemini API Key missing. Provide 'X-API-Key' header or configure GEMINI_API_KEY env var."
        )
    return effective_key


# ---------------------------------------------------------------------------
# PYDANTIC SCHEMAS
# ---------------------------------------------------------------------------
class ChatRequest(BaseModel):
    message: str
    history: List[Dict[str, str]] = []

class SummarizeTextRequest(BaseModel):
    text: str
    ratio: int = Field(30, ge=10, le=80)

class BoardConsultRequest(BaseModel):
    proposal: str

class ProposalGenerateRequest(BaseModel):
    raw_context: str
    doc_type: str = "B2B Project Proposal"
    communication_tone: str = "Professional/Formal"

class TaskStatusUpdateRequest(BaseModel):
    status: str

class TaskCreateRequest(BaseModel):
    title: str
    assigned_to: str
    estimated_hours: int = Field(..., gt=0)
    due_date: str

class EmployeeCreateRequest(BaseModel):
    full_name: str
    email: EmailStr
    role: RoleEnum
    department_id: str
    skills: List[str] = []

class ClockInRequest(BaseModel):
    employee_id: str
    location: WorkLocation = WorkLocation.OFFICE

class ClockOutRequest(BaseModel):
    employee_id: str

class LeaveSubmitRequest(BaseModel):
    employee_id: str
    leave_type: LeaveType
    start_date: str
    end_date: str
    reason: Optional[str] = None

class LeaveApproveRequest(BaseModel):
    approver_employee_id: str
    approve: bool

class SkillGapAuditRequest(BaseModel):
    department_id: str
    target_skills: List[str]

class AddCustomJobRequest(BaseModel):
    job_id: str
    interval_minutes: int = Field(..., gt=0)


# ==============================================================================
# ROUTE MODULE 1: ENTERPRISE HUB (Task Routing)
# ==============================================================================
@app.post("/api/v1/routing/optimize", tags=["Task Routing"])
async def route_task_brief(
    excel_dataset: UploadFile = File(...),
    brief_file: UploadFile = File(...)
):
    """Processes project text brief against uploaded enterprise Excel dataset to route assignments."""
    with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as tmp_excel:
        tmp_excel.write(await excel_dataset.read())
        excel_path = tmp_excel.name

    try:
        brief_bytes = await brief_file.read()
        task_brief_text = brief_bytes.decode("utf-8")

        router = IntelligenceDrivenRouter()
        router.ingest_local_file(excel_path)
        
        engine = CognitiveUnderstandingEngine()
        project_instance = engine.decompose_brief(task_brief_text)
        assignments = router.evaluate_and_route(project_instance)
        
        return {"status": "success", "assignments": assignments}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Routing optimization error: {str(e)}")
    finally:
        safe_remove(excel_path)


@app.post("/api/v1/routing/analyze-gaps", tags=["Task Routing"])
async def analyze_structural_gaps(
    excel_dataset: UploadFile = File(...),
    brief_file: UploadFile = File(...)
):
    """Analyzes organizational gap risks against a given project brief."""
    with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as tmp_excel:
        tmp_excel.write(await excel_dataset.read())
        excel_path = tmp_excel.name

    try:
        brief_bytes = await brief_file.read()
        task_brief_text = brief_bytes.decode("utf-8")

        gap_router = IntelligenceDrivenRouter()
        gap_router.ingest_local_file(excel_path)
        discovered_gaps = gap_router.analyze_structural_gaps(task_brief_text)
        
        return {"status": "success", "gaps": discovered_gaps}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gap analysis error: {str(e)}")
    finally:
        safe_remove(excel_path)


@app.get("/api/v1/routing/recovery-status", tags=["Task Routing"])
async def get_recovery_status():
    """Gets background recovery engine status metrics."""
    return {
        "is_running": recovery_engine.is_running if recovery_engine else False,
        "check_interval_seconds": recovery_engine.check_interval_seconds if recovery_engine else None
    }


# ==============================================================================
# ROUTE MODULE 2: AI CHAT ASSISTANT & AUDIO ENGINE
# ==============================================================================
@app.post("/api/v1/chat/transcribe", tags=["AI Chat Assistant"])
async def transcribe_audio(audio_file: UploadFile = File(...)):
    """Transcribes audio files using cached Whisper base model."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(await audio_file.read())
        audio_path = tmp.name

    try:
        result = whisper_model.transcribe(audio_path)
        return {"status": "success", "transcription": result.get("text", "").strip()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Transcription failure: {str(e)}")
    finally:
        safe_remove(audio_path)


@app.post("/api/v1/chat/message", tags=["AI Chat Assistant"])
async def send_chat_message(
    message: str = Form(...),
    history: str = Form("[]"),  # Expects JSON string array of past messages
    chat_asset: Optional[UploadFile] = File(None),
    api_key: str = Depends(get_api_key)
):
    """Processes chat queries along with optional document/image contexts."""
    f_bytes = await chat_asset.read() if chat_asset else None
    f_type = chat_asset.content_type if chat_asset else None

    import json
    try:
        parsed_history = json.loads(history)
    except Exception:
        parsed_history = []

    try:
        chat_client = ChatService(api_key=api_key)
        response = chat_client.generate_chat_response(
            message=message,
            history=parsed_history,
            system_instruction="You are AIVORA, a helpful, highly capable enterprise automation advisor.",
            file_bytes=f_bytes,
            mime_type=f_type
        )
        return {"status": "success", "response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chat processing failed: {str(e)}")


# ==============================================================================
# ROUTE MODULE 3: MEETING INTELLIGENCE
# ==============================================================================
@app.post("/api/v1/meeting/summarize", tags=["Meeting Intelligence"])
async def generate_meeting_minutes(
    meeting_file: UploadFile = File(...),
    api_key: str = Depends(get_api_key)
):
    """Generates structured executive meeting minutes from transcripts."""
    file_ext = os.path.splitext(meeting_file.filename)[1]
    with tempfile.NamedTemporaryFile(suffix=file_ext, delete=False) as tf:
        tf.write(await meeting_file.read())
        temp_path = tf.name

    try:
        doc_parser = DocumentSummarizerService()
        transcript_text = doc_parser.extract_text_from_file(temp_path)
        
        agent_service = MeetingIntelligenceService()
        structured_minutes = agent_service.run_agent_summarization(transcript_text, api_key=api_key)
        
        return {"status": "success", "minutes": structured_minutes}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Agent workflow error: {str(e)}")
    finally:
        safe_remove(temp_path)


# ==============================================================================
# ROUTE MODULE 4: DOCUMENT SUMMARIZER
# ==============================================================================
@app.post("/api/v1/summarize/file", tags=["Document Summarizer"])
async def summarize_document_file(
    ratio: int = Form(30),
    file: UploadFile = File(...)
):
    """Extracts text from uploaded document and generates structured executive summary."""
    file_ext = os.path.splitext(file.filename)[1]
    with tempfile.NamedTemporaryFile(suffix=file_ext, delete=False) as tf:
        tf.write(await file.read())
        temp_path = tf.name

    try:
        summarizer_client = DocumentSummarizerService()
        doc_text = summarizer_client.extract_text_from_file(temp_path)
        results = summarizer_client.analyze_document_text(doc_text, ratio=ratio)
        return {"status": "success", "analysis": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Document parsing failure: {str(e)}")
    finally:
        safe_remove(temp_path)


@app.post("/api/v1/summarize/text", tags=["Document Summarizer"])
async def summarize_plain_text(payload: SummarizeTextRequest):
    """Generates structured summary from plain text string input."""
    if len(payload.text.strip()) < 15:
        raise HTTPException(status_code=400, detail="Text length insufficient for processing.")
    
    try:
        summarizer_client = DocumentSummarizerService()
        results = summarizer_client.analyze_document_text(payload.text, ratio=payload.ratio)
        return {"status": "success", "analysis": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Summarization processing error: {str(e)}")


# ==============================================================================
# ROUTE MODULE 5: FINANCIAL INTELLIGENCE
# ==============================================================================
@app.post("/api/v1/finance/audit", tags=["Financial Intelligence"])
async def audit_financial_ledger(ledger_file: UploadFile = File(...)):
    """Audits uploaded ledger dataset and returns metrics along with isolated anomalies."""
    file_ext = os.path.splitext(ledger_file.filename)[1]
    with tempfile.NamedTemporaryFile(suffix=file_ext, delete=False) as tf:
        tf.write(await ledger_file.read())
        temp_path = tf.name

    try:
        metrics, _ = finance_service.process_ledger(temp_path)
        
        anomalies = []
        if not finance_service.anomalies_df.empty:
            anomalies = finance_service.anomalies_df.to_dict(orient="records")
            
        return {
            "status": "success",
            "metrics": metrics,
            "anomalies": anomalies
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Financial processing error: {str(e)}")
    finally:
        safe_remove(temp_path)


# ==============================================================================
# ROUTE MODULE 6: EXECUTIVE FINANCIAL DASHBOARD
# ==============================================================================
@app.get("/api/v1/executive/finance-overview", tags=["Executive Finance"])
async def get_executive_finance_overview():
    """Returns real-time executive dashboard financial KPIs and health indicators."""
    try:
        audit = exec_finance_service.analyze_executive_finance()
        return {"status": "success", "kpis": audit["kpis"]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Executive telemetry calculation error: {str(e)}")


# ==============================================================================
# ROUTE MODULE 7: ADMIN PROJECT SUCCESS CONTROL
# ==============================================================================
@app.get("/api/v1/admin/telemetry", tags=["Admin Control"])
async def get_admin_telemetry():
    """Returns admin operational diagnostics and project SLA performance telemetry."""
    try:
        telemetry = admin_analytics_service.calculate_admin_telemetry()
        return {"status": "success", "metrics": telemetry["metrics"]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Admin telemetry compile error: {str(e)}")


# ==============================================================================
# ROUTE MODULE 8: AI EXECUTIVE BOARD (Multi-Agent Consensus Platform)
# ==============================================================================
@app.post("/api/v1/board/consult", tags=["AI Executive Board"])
async def consult_executive_board(
    payload: BoardConsultRequest,
    api_key: str = Depends(get_api_key)
):
    """Submits corporate proposal for virtual C-Suite multi-agent evaluation."""
    if not payload.proposal.strip():
        raise HTTPException(status_code=400, detail="Proposal prompt cannot be empty.")
        
    try:
        board_engine = MultiAgentDecisionEngine(api_key=api_key)
        deliberations = board_engine.consult_board(payload.proposal)
        return {"status": "success", "deliberations": deliberations}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Boardroom session error: {str(e)}")


# ==============================================================================
# ROUTE MODULE 9: SMART REPLIES & PROPOSALS
# ==============================================================================
@app.post("/api/v1/proposals/generate", tags=["Smart Replies & Proposals"])
async def generate_corporate_proposal(
    payload: ProposalGenerateRequest,
    api_key: str = Depends(get_api_key)
):
    """Generates structured B2B documents, SOWs, and proposal responses."""
    if not payload.raw_context.strip():
        raise HTTPException(status_code=400, detail="Baseline context notes are required.")

    try:
        comms_agent = CorporateCommunicationsAgent(api_key=api_key)
        generated_doc = comms_agent.generate_document(
            raw_context=payload.raw_context,
            doc_type=payload.doc_type,
            communication_tone=payload.communication_tone
        )
        return {"status": "success", "document": generated_doc}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Proposal generation error: {str(e)}")


# ==============================================================================
# ROUTE MODULE 10: PROJECT PORTFOLIO MANAGER
# ==============================================================================
@app.get("/api/v1/projects/summary", tags=["Project Portfolio"])
async def get_portfolio_summary():
    """Returns global project portfolio summary numbers."""
    return {"status": "success", "summary": project_service.get_portfolio_summary()}


@app.get("/api/v1/projects/{project_id}", tags=["Project Portfolio"])
async def get_project_details(project_id: str):
    """Fetches details, tasks, and calculated health score for a specific project."""
    if project_id not in project_service.projects:
        raise HTTPException(status_code=404, detail="Project ID not found.")
        
    proj_data = project_service.projects[project_id]
    health = project_service.calculate_project_health(project_id)
    return {"status": "success", "project": proj_data, "health": health}


@app.patch("/api/v1/projects/{project_id}/tasks/{task_id}", tags=["Project Portfolio"])
async def update_task_status(project_id: str, task_id: str, payload: TaskStatusUpdateRequest):
    """Updates status of a specific task within a project."""
    success = project_service.update_task_status(project_id, task_id, payload.status)
    if not success:
        raise HTTPException(status_code=400, detail="Failed to update task status. Verify IDs.")
    return {"status": "success", "message": f"Task {task_id} updated to {payload.status}"}


@app.post("/api/v1/projects/{project_id}/tasks", tags=["Project Portfolio"])
async def add_project_task(project_id: str, payload: TaskCreateRequest):
    """Registers a new deliverable task under a specified project."""
    if project_id not in project_service.projects:
        raise HTTPException(status_code=404, detail="Project ID not found.")

    new_task_id = project_service.add_task_to_project(
        project_id=project_id,
        task_title=payload.title,
        assigned_to=payload.assigned_to,
        estimated_hours=payload.estimated_hours,
        due_date=payload.due_date
    )
    return {"status": "success", "task_id": new_task_id}


# ==============================================================================
# ROUTE MODULE 11: WORKFORCE & EMPLOYEE MANAGER
# ==============================================================================
@app.get("/api/v1/employees", tags=["Workforce Management"])
async def list_employees():
    """Fetches all employee profiles and current performance rankings."""
    df_rankings = employee_service.get_employee_rankings_dataframe()
    return {
        "status": "success", 
        "employees": employee_service.employees,
        "rankings": df_rankings.to_dict(orient="records") if not df_rankings.empty else []
    }


@app.post("/api/v1/employees", tags=["Workforce Management"])
async def register_employee(payload: EmployeeCreateRequest):
    """Registers a new employee profile into system memory."""
    emp_id = employee_service.create_employee(
        full_name=payload.full_name,
        email=payload.email,
        role=payload.role.value,
        department_id=payload.department_id,
        skills=payload.skills
    )
    return {"status": "success", "employee_id": emp_id}


@app.get("/api/v1/employees/{employee_id}", tags=["Workforce Management"])
async def get_employee_profile(employee_id: str):
    """Retrieves deep-dive profile details for a given employee."""
    if employee_id not in employee_service.employees:
        raise HTTPException(status_code=404, detail="Employee not found.")
    return {"status": "success", "profile": employee_service.get_employee_profile(employee_id)}


@app.post("/api/v1/employees/attendance/clock-in", tags=["Workforce Management"])
async def attendance_clock_in(payload: ClockInRequest):
    """Logs shift clock-in event for an employee."""
    attendance_id = employee_service.clock_in(payload.employee_id, location=payload.location.value)
    return {"status": "success", "attendance_id": attendance_id}


@app.post("/api/v1/employees/attendance/clock-out", tags=["Workforce Management"])
async def attendance_clock_out(payload: ClockOutRequest):
    """Logs shift clock-out event for an active shift."""
    success = employee_service.clock_out(payload.employee_id)
    if not success:
        raise HTTPException(status_code=400, detail="No active open shift found for this employee.")
    return {"status": "success", "message": "Clocked out successfully."}


@app.post("/api/v1/employees/leave-requests", tags=["Workforce Management"])
async def submit_leave_request(payload: LeaveSubmitRequest):
    """Submits formal employee leave request for approval."""
    req_id = employee_service.request_leave(
        emp_id=payload.employee_id,
        leave_type=payload.leave_type.value,
        start_date=payload.start_date,
        end_date=payload.end_date,
        reason=payload.reason
    )
    if not req_id:
        raise HTTPException(status_code=400, detail="Leave request failed. Check balance or dates.")
    return {"status": "success", "leave_request_id": req_id}


@app.post("/api/v1/employees/leave-requests/{leave_id}/approve", tags=["Workforce Management"])
async def approve_leave_request(leave_id: str, payload: LeaveApproveRequest):
    """Approves or rejects a pending leave request."""
    success = employee_service.approve_leave(
        leave_id=leave_id, 
        approver_emp_id=payload.approver_employee_id, 
        approve=payload.approve
    )
    if not success:
        raise HTTPException(status_code=400, detail="Unable to process leave approval decision.")
    return {"status": "success", "message": f"Leave {leave_id} set to {'APPROVED' if payload.approve else 'REJECTED'}"}


@app.get("/api/v1/employees/{employee_id}/predictive-analytics", tags=["Workforce Management"])
async def get_employee_predictive_analytics(employee_id: str):
    """Runs AI predictive models for burnout risk and promotion readiness."""
    if employee_id not in employee_service.employees:
        raise HTTPException(status_code=404, detail="Employee not found.")

    burnout = employee_service.predict_burnout(employee_id)
    promotion = employee_service.predict_promotion(employee_id)
    return {
        "status": "success",
        "burnout_risk": burnout,
        "promotion_readiness": promotion
    }


@app.post("/api/v1/employees/skill-gap-audit", tags=["Workforce Management"])
async def audit_department_skill_gaps(payload: SkillGapAuditRequest):
    """Audits skills gap across a specified department against a list of required target skills."""
    analysis = employee_service.analyze_skill_gaps(payload.department_id, payload.target_skills)
    return {"status": "success", "analysis": analysis}


# ==============================================================================
# ROUTE MODULE 12: AUTOMATED JOB SCHEDULER
# ==============================================================================
@app.get("/api/v1/scheduler/jobs", tags=["Automated Job Scheduler"])
async def list_scheduled_jobs():
    """Lists all active cron & interval jobs registered with the scheduler."""
    return {
        "status": "success",
        "is_running": scheduler_engine.scheduler.running,
        "jobs": scheduler_engine.list_jobs()
    }


@app.get("/api/v1/scheduler/history", tags=["Automated Job Scheduler"])
async def get_scheduler_history():
    """Retrieves recent job execution telemetry logs."""
    return {"status": "success", "history": scheduler_engine.get_execution_history()}


@app.post("/api/v1/scheduler/resume", tags=["Automated Job Scheduler"])
async def resume_job_scheduler():
    """Resumes the background job scheduler."""
    scheduler_engine.resume()
    return {"status": "success", "message": "Scheduler resumed."}


@app.post("/api/v1/scheduler/pause", tags=["Automated Job Scheduler"])
async def pause_job_scheduler():
    """Pauses the background job scheduler."""
    scheduler_engine.pause()
    return {"status": "success", "message": "Scheduler paused."}


@app.post("/api/v1/scheduler/trigger/backup", tags=["Automated Job Scheduler"])
async def trigger_manual_backup():
    """Executes an immediate manual backup of databases and files."""
    res = scheduler_engine.job_automatic_backups()
    return {"status": "success", "result": res}


@app.post("/api/v1/scheduler/trigger/cleanup", tags=["Automated Job Scheduler"])
async def trigger_manual_cleanup():
    """Executes immediate cleanup of temporary system files."""
    res = scheduler_engine.job_system_cleanup()
    return {"status": "success", "result": res}


@app.post("/api/v1/scheduler/jobs", tags=["Automated Job Scheduler"])
async def register_custom_job(payload: AddCustomJobRequest):
    """Registers a new custom background interval job."""
    success = scheduler_engine.add_job(
        scheduler_engine.job_health_monitoring, 
        trigger_type="interval", 
        job_id=payload.job_id, 
        minutes=payload.interval_minutes
    )
    if not success:
        raise HTTPException(status_code=400, detail="Failed to register job identifier.")
    return {"status": "success", "message": f"Job {payload.job_id} successfully scheduled."}


# ---------------------------------------------------------------------------
# APPLICATION ENTRYPOINT
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
