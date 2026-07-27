# ==============================================================================
# APPLICATION: main.py
# DESCRIPTION: Unified AIVORA Core Operating System Interface Framework.
#              Orchestrates Workforce Routing, Voice/Text Chat, Multi-Agent
#              Meeting Analytics, Semantic Text Extraction, Fiscal Ledgers,
#              Executive Financial Cockpits, and Admin Project Telemetry.
# ==============================================================================

import os
import asyncio
import threading
import pandas as pd
import streamlit as st
import tempfile
import whisper
from streamlit_mic_recorder import mic_recorder

# ---------------------------------------------------------------------------
# CORE SERVICE LAYER IMPORTS
# ---------------------------------------------------------------------------
from datetime import datetime
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

# Configure global application presentation properties
st.set_page_config(page_title="AIVORA Operating System", layout="wide", page_icon="⚡")

# --- ASYNCHRONOUS BACKGROUND TASK RECOVERY ENGINE LIFECYCLE ---
@st.cache_resource
def initialize_global_recovery_engine():
    """
    Spins up the Autonomous Task Recovery Engine exactly once inside a dedicated 
    background daemon thread to avoid blocking the main Streamlit UI rendering cycles.
    """
    # In production, set check_interval_seconds to 60 or 300
    engine = AutonomousTaskRecoveryEngine(check_interval_seconds=15)
    
    def run_async_loop():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(engine.start())
        
    bg_thread = threading.Thread(target=run_async_loop, daemon=True)
    bg_thread.start()
    return engine

# Activate background workflow monitoring engine
recovery_engine = initialize_global_recovery_engine()

# --- OPTIMIZED AUDIO TRANSCRIPTION CACHING ---
@st.cache_resource
def load_whisper():
    """Loads and caches the model weights across session frames to eliminate startup latency."""
    return whisper.load_model("base")

whisper_model = load_whisper()

# --- SAFE CLEANUP HELPER FOR WINDOWS ---
def safe_remove(file_path: str):
    """Safely removes temporary backend data files without triggering WinError 32."""
    try:
        if file_path and os.path.exists(file_path):
            os.remove(file_path)
    except PermissionError:
        pass  # Gracefully swallow OS file locks
    # --- BACKGROUND AUTOMATION SCHEDULER LIFECYCLE ---
@st.cache_resource
def initialize_global_scheduler_service():
    """
    Spins up the APScheduler background daemon exactly once across session 
    frames to execute automated backups, health checks, and task recovery.
    """
    scheduler_service = SchedulerService()
    scheduler_service.start()
    return scheduler_service

# Activate platform background automation scheduler
scheduler_engine = initialize_global_scheduler_service()

# ---------------------------------------------------------------------------
# SIDEBAR CONTROL WORKSPACE
# ---------------------------------------------------------------------------
st.sidebar.title("⚡ AIVORA OS")
st.sidebar.markdown("### Core Modules")

app_mode = st.sidebar.radio("Select Active Framework Workspace:", [
    "🏢 Enterprise Hub (Task Routing)",
    "💬 AI Chat Assistant",
    "🤝 Meeting Intelligence",
    "📄 Document Summarizer",
    "💰 Financial Intelligence",
    "📊 Executive Financial Dashboard",
    "⚙️ Admin Project Success Control",
    "📋 Project Portfolio Manager",
    "👥 Workforce & Employee Manager",
    "⏱️ Automated Job Scheduler",  # <-- ADD THIS LINE
    "🧠 AI Executive Board",
    "📩 Smart Replies & Proposals"
])
st.sidebar.markdown("---")
st.sidebar.markdown("### Telemetry Matrix")
# Persistent visual indicators for the automated recovery background stack
if recovery_engine.is_running:
    st.sidebar.success("⚙️ Recovery Engine: ACTIVE")
else:
    st.sidebar.error("⚙️ Recovery Engine: OFFLINE")

st.sidebar.markdown("---")
st.sidebar.markdown("### Security Configuration")

# Secure unified entry point for external LLM models
env_key = os.getenv("GEMINI_API_KEY", "")
api_key_input = st.sidebar.text_input(
    "Gemini API Key:", 
    value=env_key, 
    type="password",
    help="Falls back to local machine environment values if left empty."
)
effective_api_key = api_key_input or env_key

st.sidebar.markdown("---")
if effective_api_key:
    st.sidebar.success("🔒 API Authentication Connected")
else:
    st.sidebar.warning("⚠️ LLM features will remain locked without an API key.")

# ---------------------------------------------------------------------------
# MODULE 1: ENTERPRISE HUB (Task Routing Engine)
# ---------------------------------------------------------------------------
if app_mode == "🏢 Enterprise Hub (Task Routing)":
    st.title("🏢 Enterprise Hub & Task Routing Matrix")
    st.markdown("Analyze workforce configurations, cross-check operational capabilities, and distribute task allocations.")

    uploaded_excel = st.file_uploader(
        "Ingest Core Enterprise Dataset (xlsx format):", 
        type=["xlsx"]
    )

    st.markdown("---")

    if uploaded_excel:
        with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as temp_file:
            temp_file.write(uploaded_excel.getbuffer())
            temp_excel_path = temp_file.name

        try:
            router = IntelligenceDrivenRouter()
            router.ingest_local_file(temp_excel_path)
            
            with pd.ExcelFile(temp_excel_path) as xls:
                df_employees_raw = pd.read_excel(xls, sheet_name="Employees")
                df_kpis_raw = pd.read_excel(xls, sheet_name="KPIs")
            
        except Exception as e:
            st.error(f"Failed to ingest asset map: {str(e)}")
            st.stop()
        finally:
            safe_remove(temp_excel_path)

        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "🧠 Task Routing Optimization", 
            "👥 Workforce Hierarchy", 
            "🎯 Role KPI Models", 
            "⚠️ Operational Gap Analyzer",
            "⚙️ Autonomous Task Recovery"
        ])

        with tab1:
            st.subheader("Pipeline Project Assignment Routing Engine")
            uploaded_brief = st.file_uploader("Upload Project Narrative Text Brief (.txt)", type=["txt"], key="routing_brief")
            
            if uploaded_brief:
                task_brief_text = uploaded_brief.read().decode("utf-8")
                
                if st.button("⚡ Execute AI Optimization Match", type="primary"):
                    with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as tf:
                        tf.write(uploaded_excel.getbuffer())
                        tp = tf.name
                    try:
                        active_router = IntelligenceDrivenRouter()
                        active_router.ingest_local_file(tp)
                        engine = CognitiveUnderstandingEngine()
                        
                        project_instance = engine.decompose_brief(task_brief_text)
                        assignments = active_router.evaluate_and_route(project_instance)
                        
                        st.success("✅ Operational distribution optimized across model parameters.")
                        
                        df_output = pd.DataFrame(assignments)
                        display_columns = ["task_id", "task_title", "category", "estimated_hours", "assigned_employee", "match_confidence", "routing_status"]
                        st.dataframe(df_output[display_columns], use_container_width=True)
                        
                    finally:
                        safe_remove(tp)

        with tab2:
            st.subheader("Corporate Structural Hierarchies")
            search_query = st.selectbox("Quick Filter Employee Data Profiles", ["All"] + list(df_employees_raw["Employee"].unique()))
            if search_query != "All":
                st.table(df_employees_raw[df_employees_raw["Employee"] == search_query])
            else:
                st.dataframe(df_employees_raw, use_container_width=True)

        with tab3:
            st.subheader("Target Objective Matrix")
            col_kpi1, col_kpi2 = st.columns([1, 2])
            with col_kpi1:
                selected_role = st.radio("Select Corporate Title Scope", list(df_kpis_raw["Role"].unique()))
            with col_kpi2:
                st.dataframe(df_kpis_raw[df_kpis_raw["Role"] == selected_role][["KPI"]], use_container_width=True)

        with tab4:
            st.subheader("Strategic Workflow Risk Analyzer")
            if uploaded_brief:
                with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as tf:
                    tf.write(uploaded_excel.getbuffer())
                    tp = tf.name
                try:
                    gap_router = IntelligenceDrivenRouter()
                    gap_router.ingest_local_file(tp)
                    discovered_gaps = gap_router.analyze_structural_gaps(task_brief_text)
                    
                    if discovered_gaps:
                        st.warning(f"⚠️ Found {len(discovered_gaps)} operational resource risk fields.")
                        df_gaps = pd.DataFrame(discovered_gaps)
                        df_gaps.columns = ["Missing Role", "Risk Reason", "Matched Indicators", "Suggested Responsibilities"]
                        st.dataframe(df_gaps, use_container_width=True)
                    else:
                        st.success("🎉 Structural Integrity Secure! No missing capability footprints detected.")
                finally:
                    safe_remove(tp)
            else:
                st.info("Upload a text brief in Tab 1 to run the structure analyzer.")

        with tab5:
            st.subheader("🤖 Autonomous Task State Machine Controller")
            st.markdown("Monitor real-time system capability actions when human deliverables breach core operation thresholds.")
            
            col_e1, col_e2 = st.columns(2)
            with col_e1:
                st.metric("State Controller Status", "RUNNING" if recovery_engine.is_running else "STOPPED")
                st.info("ℹ️ Engine auto-polls transaction database schemas to verify workflow health matrices.")
            with col_e2:
                st.metric("Polled Grace Boundaries", f"{recovery_engine.check_interval_seconds}s Window")
            
            st.markdown("#### Expected Mitigation Progression Loop")
            st.code(
                "[ASSIGNED Stage Overdue]     -> Trigger Automated Alerts & Notification Logs\n"
                "[WARNING Stage Overdue]      -> Escalate Severity Flag & Update System Urgency\n"
                "[ESCALATION Stage Overdue]   -> Spin Up Secondary Fallback Autonomous AI Agents",
                language="text"
            )
    else:
        st.info("Please upload the Core Dataset file to activate the operational runtime workspace.")

# ---------------------------------------------------------------------------
# MODULE 2: AI CHAT ASSISTANT (Enriched with Voice and Document Ingestion)
# ---------------------------------------------------------------------------
elif app_mode == "💬 AI Chat Assistant":
    st.title("💬 AIVORA Chat Assistant")
    st.markdown("Query systemic operational patterns, upload media files, or talk via audio command frames.")

    if not effective_api_key:
        st.info("🔑 Please enter a valid Gemini API Key in the left sidebar menu to activate the Chat interface.")
        st.stop()

    # Initialize persistent state registers
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "last_audio_bytes" not in st.session_state:
        st.session_state.last_audio_bytes = None
    if "voice_active_prompt" not in st.session_state:
        st.session_state.voice_active_prompt = None

    # Render Historical Conversational Messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Asset Control Console Drawer Layout
    with st.expander("🎙️ & 📄 Operational Media Ingestion Drawer", expanded=True):
        col_v, col_d = st.columns(2)
        with col_v:
            st.markdown("**Voice Control Engine**")
            audio = mic_recorder(
                start_prompt="🎤 Record Command",
                stop_prompt="⏹️ Process Audio",
                key="voice_recorder",
                use_container_width=True,
            )
        with col_d:
            st.markdown("**Document Reader Engine**")
            chat_asset = st.file_uploader(
                "Upload asset context (Images/PDFs):", 
                type=["png", "jpg", "jpeg", "pdf"],
                key="chat_asset_uploader"
            )

    # Process acoustic frames only when a fresh sequence signature is captured
    if audio and audio["bytes"] != st.session_state.last_audio_bytes:
        st.session_state.last_audio_bytes = audio["bytes"]
        audio_path = None
        
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                tmp.write(audio["bytes"])
                audio_path = tmp.name

            with st.spinner("🎙️ Transcribing structural acoustic waveforms via Whisper..."):
                result = whisper_model.transcribe(audio_path)
                transcribed_text = result.get("text", "").strip()
                
                if transcribed_text:
                    st.session_state.voice_active_prompt = transcribed_text
                    st.success(f"🗣️ Vocal Frame Captured: \"{transcribed_text}\"")
                    
        except Exception as audio_err:
            st.error(f"Acoustic Ingestion Error: {str(audio_err)}")
        finally:
            if audio_path and os.path.exists(audio_path):
                safe_remove(audio_path)

    # Standardize textual dialogue captures
    text_prompt = st.chat_input("Query enterprise knowledge graphs...")

    # Interface Ingestion Router Coordinator
    prompt = None
    if text_prompt:
        prompt = text_prompt
        st.session_state.voice_active_prompt = None  # Flush older voice cache queues
    elif st.session_state.voice_active_prompt:
        prompt = st.session_state.voice_active_prompt
        st.session_state.voice_active_prompt = None  # Consume frame instant immediately

    if prompt:
        # Display user input immediately 
        with st.chat_message("user"):
            st.markdown(prompt)
            if chat_asset:
                st.caption(f"📁 Attached Data Vector: `{chat_asset.name}` ({chat_asset.type})")

        st.session_state.messages.append({
            "role": "user",
            "content": prompt + (f" (File attached: {chat_asset.name})" if chat_asset else "")
        })

        with st.spinner("Thinking..."):
            try:
                # Extract file parameters if an active upload exists
                f_bytes = chat_asset.getvalue() if chat_asset else None
                f_type = chat_asset.type if chat_asset else None

                chat_client = ChatService(api_key=effective_api_key)
                response = chat_client.generate_chat_response(
                    message=prompt,
                    history=st.session_state.messages[:-1],
                    system_instruction="You are AIVORA, a helpful, highly capable enterprise automation advisor.",
                    file_bytes=f_bytes,
                    mime_type=f_type
                )
            except Exception as e:
                response = f"⚠️ Chat processing failed: {str(e)}"

        with st.chat_message("assistant"):
            st.markdown(response)

        st.session_state.messages.append({
            "role": "assistant",
            "content": response
        })
        
        st.rerun()

# ---------------------------------------------------------------------------
# MODULE 3: MEETING INTELLIGENCE
# ---------------------------------------------------------------------------
elif app_mode == "🤝 Meeting Intelligence":
    st.title("🤝 Meeting Intelligence Agent")
    st.markdown("Extract structured tasks, clear action ownerships, and event context details from raw notes.")

    if not effective_api_key:
        st.info("🔑 Please enter a valid Gemini API Key in the left sidebar menu to activate CrewAI agent orchestration.")
        st.stop()

    meeting_file = st.file_uploader("Upload Meeting Documentation Transcript (.txt or .docx)", type=["txt", "docx"])
    
    if meeting_file:
        with tempfile.NamedTemporaryFile(suffix=os.path.splitext(meeting_file.name)[1], delete=False) as tf:
            tf.write(meeting_file.getbuffer())
            temp_path = tf.name

        if st.button("Generate Structured Executive Minutes", type="primary"):
            with st.spinner("Orchestrating agent workflows over text context..."):
                try:
                    doc_parser = DocumentSummarizerService()
                    transcript_text = doc_parser.extract_text_from_file(temp_path)
                    
                    agent_service = MeetingIntelligenceService()
                    structured_minutes = agent_service.run_agent_summarization(transcript_text, api_key=effective_api_key)
                    
                    st.success("✅ Minutes generated via multi-agent validation loops.")
                    st.markdown("---")
                    st.markdown(structured_minutes)
                except Exception as e:
                    st.error(f"Agent Framework Error encountered: {str(e)}")
                finally:
                    safe_remove(temp_path)
    else:
        st.info("Upload a meeting transcript document to run the generation agent.")

# ---------------------------------------------------------------------------
# MODULE 4: DOCUMENT SUMMARIZER
# ---------------------------------------------------------------------------
elif app_mode == "📄 Document Summarizer":
    st.title("📄 Intelligent Document Summarizer")
    st.markdown("Deconstruct heavy documents via structural cosine-centrality similarity matrices.")

    uploaded_doc = st.file_uploader("Upload Target Text Document (.txt, .docx, .pdf)", type=["txt", "docx", "pdf"])
    
    st.markdown("### Extraction Parameters")
    summary_ratio = st.slider("Target Summary Core Density Ratio (%)", min_value=10, max_value=80, value=30, step=5)

    doc_text = ""
    if uploaded_doc is not None:
        with tempfile.NamedTemporaryFile(suffix=os.path.splitext(uploaded_doc.name)[1], delete=False) as tf:
            tf.write(uploaded_doc.getbuffer())
            temp_doc_path = tf.name
            
        try:
            summarizer_client = DocumentSummarizerService()
            doc_text = summarizer_client.extract_text_from_file(temp_doc_path)
            st.success(f"📂 Read complete contents from: {uploaded_doc.name}")
        except Exception as e:
            st.error(f"Parsing engine read failure: {str(e)}")
        finally:
            safe_remove(temp_doc_path)
    else:
        doc_text = st.text_area("Or manually input plain block context text for scanning parameters:", height=150)

    if st.button("Execute Document Analysis Pipeline", type="primary"):
        if doc_text and len(doc_text.strip()) > 15:
            with st.spinner("Extracting structural semantic embeddings..."):
                try:
                    summarizer_client = DocumentSummarizerService()
                    results = summarizer_client.analyze_document_text(doc_text, ratio=summary_ratio)
                    
                    st.markdown("---")
                    
                    metrics = results["metrics"]
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Total Words", f"{metrics['word_count']:,}")
                    col2.metric("Characters", f"{metrics['character_count']:,}")
                    col3.metric("Processed Blocks", f"{metrics['segments']}")
                    col4.metric("Est. Read Duration", f"{metrics['estimated_reading_time_mins']} min")
                    
                    st.markdown("---")
                    
                    col_left, col_right = st.columns(2)
                    with col_left:
                        st.subheader("🎯 Primary Semantic Bigrams")
                        if results["keywords"]:
                            st.markdown(" ".join([f"`{kw}`" for kw in results["keywords"]]))
                        else:
                            st.info("No repetitive complex bi-grams detected.")
                    with col_right:
                        st.subheader("🎭 Document Core Tone")
                        sentiment = results["sentiment"]
                        st.markdown(f"**Classification Label:** {sentiment['label']} (Confidence Indicator Score: `{sentiment['score']}%`)")
                    
                    st.markdown("---")
                    
                    st.subheader("✨ Extractive Summary Output")
                    if Math_summary := results["summary"]:
                        st.info(Math_summary)
                    else:
                        st.warning("Could not assemble condensed vectors from structural context layers.")
                        
                except Exception as e:
                    st.error(f"Mathematical processing failure inside pipeline calculations: {str(e)}")
        else:
            st.warning("⚠️ Processing aborted: Your text input area buffer context length is insufficient.")

# ---------------------------------------------------------------------------
# MODULE 5: FINANCIAL INTELLIGENCE (ML Forecasting & Auditing Platform)
# ---------------------------------------------------------------------------
elif app_mode == "💰 Financial Intelligence":
    st.title("💰 AIVORA Financial Intelligence Hub")
    st.markdown("Upload active corporate spreadsheets to calculate real-time runway horizons, predict cash flow velocities, and isolate statistical transaction anomalies.")

    # Initialize the core financial engine service
    fin_service = FinanceService()

    # File Ingestion Engine Pipeline
    uploaded_ledger = st.file_uploader(
        "Upload Corporate Ledger Spreadsheet (CSV or Excel formats):", 
        type=["csv", "xlsx", "xls"]
    )

    if uploaded_ledger is not None:
        # Secure file path resolution for file-system safety
        temp_dir = "outputs"
        os.makedirs(temp_dir, exist_ok=True)
        temp_file_path = os.path.join(temp_dir, f"temp_{uploaded_ledger.name}")
        
        with open(temp_file_path, "wb") as f:
            f.write(uploaded_ledger.getbuffer())

        st.success(f"📁 Source file `{uploaded_ledger.name}` loaded into active engine environment memory.")
        
        # Trigger the advanced processing engine pipeline
        if st.button("⚡ Run Advanced ML Financial Audit", type="primary"):
            with st.spinner("Executing predictive metrics calculations and parsing transaction risk..."):
                metrics, chart = fin_service.process_ledger(temp_file_path)
                
                # Check for parsing anomalies or exceptions
                if "error" in metrics:
                    st.error(f"Financial Processing Exception: {metrics['error']}")
                else:
                    st.markdown("### 📊 Core Operational Financial Indicators")
                    
                    # Row 1 Indicators Display Layout Dashboard Panels
                    col_rev, col_burn, col_margin = st.columns(3)
                    with col_rev:
                        st.metric("Gross Logged Revenue", f"${metrics['revenue']:,.2f}")
                    with col_burn:
                        st.metric("Avg Monthly Operating Expenses", f"${metrics['burn_rate']:,.2f}")
                    with col_margin:
                        st.metric("Calculated Operational Margin", f"{metrics['margin']:.2f}%")
                        
                    # Row 2 Advanced Intelligence Features Display Panels
                    st.markdown("---")
                    col_runway, col_anom = st.columns(2)
                    with col_runway:
                        runway_val = metrics['estimated_runway_months']
                        runway_text = "Stable Flow (>999 Mo)" if runway_val >= 999 else f"{runway_val:.1f} Months Available"
                        st.metric("🚀 Predicted Cash Runway Horizon", runway_text)
                    with col_anom:
                        anomaly_color = "🟢" if metrics['anomaly_count'] == 0 else "🚨"
                        st.metric(f"{anomaly_color} Isolated Anomaly Risk Items", f"{metrics['anomaly_count']} Flagged Rows")

                    # Cross-Module Synergy: Cache results so the AI Executive Board can read them
                    st.session_state.active_financial_summary = {
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "summary_text": (
                            f"Active Financial Audit Metrics Log:\n"
                            f"- Total Revenue Profile: ${metrics['revenue']:,.2f}\n"
                            f"- Monthly Operational Expense Baseline: ${metrics['burn_rate']:,.2f}\n"
                            f"- Net Margin Percent Vector: {metrics['margin']:.2f}%\n"
                            f"- ML Predicted Cash Runway Window: {runway_text}\n"
                            f"- Statistical Anomaly Flagged Items Count: {metrics['anomaly_count']} rows detected."
                        )
                    }
                    st.toast("📊 Telemetry data successfully shared globally with AIVORA core workspace modules.")

                    # Render the predictive analytical cash projection chart
                    if chart is not None:
                        st.markdown("---")
                        st.markdown("### 📈 Machine Learning Capital Projection Trajectory")
                        st.pyplot(chart)

                    # Render the Statistical Exception Report Panel if anomalies were isolated
                    if not fin_service.anomalies_df.empty:
                        st.markdown("---")
                        st.subheader("🚨 Categorical Variance & Expense Anomaly Audit Logs")
                        st.markdown("The following line items deviate from typical spending patterns in their categories by more than **2.2 standard deviations ($Z$-score)**:")
                        
                        # Display clean, scannable data grid view for operations teams
                        st.dataframe(
                            fin_service.anomalies_df,
                            use_container_width=True
                        )
                    else:
                        st.markdown("---")
                        st.success("✅ Financial Risk Assessment Complete: All operational items fall within safe categorical statistical limits.")
                        
        # Clean up the file system after processing is complete
        try:
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)
        except Exception:
            pass
    else:
        st.info("💡 Ingestion Queue Empty: Please upload a corporate transaction ledger file above to review financial intelligence data.")

# ---------------------------------------------------------------------------
# MODULE 6: EXECUTIVE FINANCIAL DASHBOARD
# ---------------------------------------------------------------------------
elif app_mode == "📊 Executive Financial Dashboard":
    st.title("📊 Executive Real-Time Financial Overview")
    st.markdown("Real-time executive cockpit tracking financial health scores, P&L velocity, and cash flow projections.")

    exec_service = ExecutiveFinanceService()
    
    if st.button("⚡ Refresh Real-Time Executive Ledger", type="primary"):
        with st.spinner("Calculating executive telemetry & running projection algorithms..."):
            audit = exec_service.analyze_executive_finance()
            kpis = audit["kpis"]

            # Row 1: Executive KPI Panel
            col_h, col_r, col_p, col_f = st.columns(4)
            col_h.metric("Financial Health Score", f"{kpis['health_score']} / 100")
            col_r.metric("Gross Revenue", f"${kpis['gross_revenue']:,.0f}")
            col_p.metric("Net Profit (P&L)", f"${kpis['net_profit']:,.0f}", f"{kpis['profit_margin_pct']:.1f}% Margin")
            col_f.metric("Forecasted Q Revenue", f"${kpis['forecasted_q_revenue']:,.0f}")

            # Row 2: Cash & Burn Indicators
            st.markdown("---")
            col_c, col_b, col_u, col_o = st.columns(4)
            col_c.metric("Cash Reserve", f"${kpis['cash_reserve']:,.0f}")
            col_b.metric("Monthly Burn Rate", f"${kpis['burn_rate']:,.0f}")
            col_u.metric("Budget Utilization", f"{kpis['budget_utilization_pct']:.1f}%")
            col_o.metric("Overdue Payments Risk", f"${kpis['overdue_payments']:,.0f}")

            # Visual Charts Section
            st.markdown("---")
            st.subheader("📈 Executive Visual Intelligence")
            charts = exec_service.generate_executive_charts(audit)

            c1, c2 = st.columns(2)
            with c1:
                st.pyplot(charts[0])  # P&L Trajectory
                st.pyplot(charts[2])  # Budget Allocation
            with c2:
                st.pyplot(charts[1])  # Cash Flow Projection
                st.pyplot(charts[3])  # Outstanding Invoices Donut

# ---------------------------------------------------------------------------
# MODULE 7: ADMIN PROJECT SUCCESS CONTROL
# ---------------------------------------------------------------------------
elif app_mode == "⚙️ Admin Project Success Control":
    st.title("⚙️ Admin Operations & Project Success Analytics")
    st.markdown("Admin-level telemetry cockpit tracking project success rates, task delivery velocities, and recovery engine intervention frequencies.")

    admin_service = AdminAnalyticsService()
    
    with st.spinner("Compiling operational logs & calculating SLA performance..."):
        telemetry = admin_service.calculate_admin_telemetry()
        m = telemetry["metrics"]

        # Row 1: Admin Telemetry Summary Metrics
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Project Success Rate", f"{m['project_success_rate']:.1f}%")
        col2.metric("On-Time SLA Delivery", f"{m['on_time_delivery_rate']:.1f}%")
        col3.metric("Autonomous Agent Split", f"{m['agent_execution_ratio']:.1f}%")
        col4.metric("Recovery Interventions", f"{m['recovery_interventions']} Events")

        st.markdown("---")
        st.subheader("📊 Operational Diagnostics & SLA Analytics")
        admin_charts = admin_service.generate_admin_charts(telemetry)

        ac1, ac2 = st.columns(2)
        with ac1:
            st.pyplot(admin_charts[0])  # Success Rate Trend
            st.pyplot(admin_charts[2])  # Recovery Stage Frequency
        with ac2:
            st.pyplot(admin_charts[1])  # Human vs AI Split
            st.pyplot(admin_charts[3])  # SLA Latency by Domain

# ---------------------------------------------------------------------------
# MODULE 8: AI EXECUTIVE BOARD (Multi-Agent Consensus Platform)
# ---------------------------------------------------------------------------
elif app_mode == "🧠 AI Executive Board":
    st.title("🧠 AI Executive Board Room")
    st.markdown("Submit business proposals, operational plans, or systemic challenges to get simultaneous critiques from your virtual C-Suite officers.")

    if not effective_api_key:
        st.info("🔑 Please enter a valid Gemini API Key in the left sidebar menu to activate the Board Room.")
        st.stop()

    proposal_input = st.text_area(
        "Input corporate proposal, project idea, or critical issue for evaluation:",
        placeholder="Example: We should transition our entire database infrastructure from local servers to a fully serverless cloud architecture next quarter...",
        height=150
    )

    if st.button("⚡ Summon Board Deliberation", type="primary"):
        if proposal_input.strip():
            with st.spinner("Convening parallel board meeting frames..."):
                try:
                    # Initialize the engine
                    board_engine = MultiAgentDecisionEngine(api_key=effective_api_key)
                    
                    # Run the multi-threaded deliberation loop
                    deliberations = board_engine.consult_board(proposal_input)
                    
                    st.success("✅ Executive feedback loops finalized successfully.")
                    st.markdown("---")
                    
                    # Render outputs side-by-side using Streamlit columns
                    cols = st.columns(3)
                    
                    for idx, (persona_title, feedback) in enumerate(deliberations.items()):
                        with cols[idx]:
                            st.subheader(persona_title)
                            st.info(feedback)
                            
                except Exception as e:
                    st.error(f"Boardroom session interrupted unexpectedly: {str(e)}")
        else:
            st.warning("⚠️ Submission aborted: Please input a valid textual scenario proposal to analyze.")

# ---------------------------------------------------------------------------
# MODULE 9: SMART REPLIES & PROPOSALS (Contextual Content Generator)
# ---------------------------------------------------------------------------
elif app_mode == "📩 Smart Replies & Proposals":
    st.title("📩 Smart Replies & Operational Proposal Engine")
    st.markdown("Draft structured corporate proposals, statements of work, or automated executive email updates using specific operational text.")

    if not effective_api_key:
        st.info("🔑 Please enter a valid Gemini API Key in the left sidebar menu to activate the communications generator.")
        st.stop()

    col_inputs, col_params = st.columns([2, 1])
    
    with col_inputs:
        raw_context_input = st.text_area(
            "Provide baseline notes, task parameters, or message history to compile:",
            placeholder="Example: We need to pitch a software migration project to Alpha Corp. Total cost $45k, timeline 3 months, includes automated QA setup, database migration, and system monitoring dashboard...",
            height=200
        )
        
    with col_params:
        selected_doc_type = st.selectbox(
            "Select Output Asset Type:",
            ["B2B Project Proposal", "Executive Briefing Update", "Formal Contract Description", "Client Support Escalation Response"]
        )
        
        selected_tone = st.select_slider(
            "Target Communication Tone Vector:",
            options=["Direct/Urgent", "Professional/Formal", "Persuasive/Strategic"]
        )

    st.markdown("---")
    
    if st.button("⚡ Generate Corporate Asset Content", type="primary"):
        if raw_context_input.strip():
            with st.spinner("Compiling contextual layers into document structure..."):
                try:
                    # Execute backend generator pipeline
                    comms_agent = CorporateCommunicationsAgent(api_key=effective_api_key)
                    generated_output = comms_agent.generate_document(
                        raw_context=raw_context_input,
                        doc_type=selected_doc_type,
                        communication_tone=selected_tone
                    )
                    
                    st.success(f"✅ Formal {selected_doc_type} drafted successfully.")
                    
                    # Present the output cleanly inside an isolated block
                    st.markdown("### 📋 Generated Asset Draft")
                    st.info(generated_output)
                    
                    # Provide an immediate download file action feature
                    file_friendly_name = selected_doc_type.lower().replace(" ", "_")
                    st.download_button(
                        label="💾 Download Document (.md)",
                        data=generated_output,
                        file_name=f"AIVORA_{file_friendly_name}.md",
                        mime="text/markdown"
                    )
                    
                except Exception as e:
                    st.error(f"Communications tracking pipeline calculation failed: {str(e)}")
        else:
            st.warning("⚠️ Action Aborted: Please input baseline context notes before triggering generation loops.")
            # ---------------------------------------------------------------------------
# MODULE: PROJECT PORTFOLIO MANAGER
# ---------------------------------------------------------------------------
elif app_mode == "📋 Project Portfolio Manager":
    st.title("📋 Project Portfolio & Task Manager")
    st.markdown("Track active project lifecycles, monitor task health metrics, and manage delivery bottlenecks.")

    pm_service = ProjectManagerService()
    portfolio_summary = pm_service.get_portfolio_summary()

    # Row 1: Global Portfolio Summary
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Active Projects", portfolio_summary["total_projects"])
    col2.metric("Total Portfolio Budget", f"${portfolio_summary['total_budget']:,.2f}")
    col3.metric("Global Completion Rate", f"{portfolio_summary['global_completion_pct']}%")
    col4.metric("Blocked Tasks Across Portfolio", portfolio_summary["blocked_tasks"])

    st.markdown("---")

    # Row 2: Selected Project Deep Dive
    project_ids = list(pm_service.projects.keys())
    if not project_ids:
        st.info("No projects active in the system storage.")
    else:
        selected_proj_id = st.selectbox(
            "Select Active Project to inspect:",
            options=project_ids,
            format_func=lambda pid: f"{pid} — {pm_service.projects[pid]['title']} ({pm_service.projects[pid]['client']})"
        )

        if selected_proj_id:
            proj_data = pm_service.projects[selected_proj_id]
            health = pm_service.calculate_project_health(selected_proj_id)

            st.subheader(f"📌 {proj_data['title']}")
            st.caption(f"**Client:** {proj_data['client']} | **Owner:** {proj_data['owner']} | **Target Completion:** {proj_data['target_completion']}")

            # Health Metrics Panel
            flag_icon = "🟢" if health["status_flag"] == "GREEN" else ("🟡" if health["status_flag"] == "AMBER" else "🔴")
            
            h_col1, h_col2, h_col3, h_col4 = st.columns(4)
            h_col1.metric("Health Index", f"{health['health_score']} / 100", f"{flag_icon} {health['status_flag']}")
            h_col2.metric("Completion Status", f"{health['completion_pct']}%")
            h_col3.metric("Blocked Items", health["blocked_tasks"])
            h_col4.metric("Overdue Items", health["overdue_tasks"])

            st.markdown("---")

            tab_tasks, tab_add_task = st.tabs(["📋 Task Roster", "➕ Add Task"])

            # Tab 1: Task List & Quick Status Update
            with tab_tasks:
                df_tasks = pm_service.get_project_tasks_dataframe(selected_proj_id)
                if not df_tasks.empty:
                    st.dataframe(df_tasks, use_container_width=True)

                    st.markdown("#### Quick Status Action")
                    c_tsk, c_stat, c_btn = st.columns([2, 2, 1])
                    with c_tsk:
                        target_task = st.selectbox(
                            "Select Task:",
                            [t["task_id"] for t in proj_data["tasks"]],
                            key="update_task_sel"
                        )
                    with c_stat:
                        target_status = st.selectbox(
                            "Target Status:",
                            ["BACKLOG", "IN_PROGRESS", "BLOCKED", "COMPLETED"],
                            key="update_status_sel"
                        )
                    with c_btn:
                        st.write("")
                        st.write("")
                        if st.button("Apply Status"):
                            if pm_service.update_task_status(selected_proj_id, target_task, target_status):
                                st.success(f"Task {target_task} updated to {target_status}!")
                                st.rerun()
                else:
                    st.info("No tasks logged for this project yet.")

            # Tab 2: Add New Task Form
            with tab_add_task:
                st.markdown("#### Register New Task Deliverable")
                with st.form(key="add_task_form"):
                    f_title = st.text_input("Task Title")
                    f_assignee = st.text_input("Assigned Personnel")
                    f_hours = st.number_input("Estimated Effort (Hours)", min_value=1, value=10)
                    f_due = st.date_input("Target Due Date")

                    if st.form_submit_button("➕ Create Task"):
                        if f_title and f_assignee:
                            new_id = pm_service.add_task_to_project(
                                project_id=selected_proj_id,
                                task_title=f_title,
                                assigned_to=f_assignee,
                                estimated_hours=int(f_hours),
                                due_date=f_due.strftime("%Y-%m-%d")
                            )
                            st.success(f"Task successfully registered: `{new_id}`")
                            st.rerun()
                        else:
                            st.warning("Please fill in both Task Title and Assignee before submitting.")
                            # ---------------------------------------------------------------------------
# MODULE: WORKFORCE & EMPLOYEE MANAGER
# ---------------------------------------------------------------------------
elif app_mode == "👥 Workforce & Employee Manager":
    st.title("👥 Workforce & Employee Management Workspace")
    st.markdown("Manage employee records, monitor attendance logs, process leave approvals, and review AI burnout & promotion predictions.")

    ems = EmployeeManagerService()

    # --- Top Metric Telemetry Panel ---
    df_rankings = ems.get_employee_rankings_dataframe()
    total_staff = len(ems.employees)
    active_staff = sum(1 for e in ems.employees.values() if e.is_active)
    pending_leaves = sum(1 for l in ems.leave_requests.values() if l.status == "PENDING")
    active_warnings = len(ems.warnings)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Headcount", f"{total_staff} Staff", f"{active_staff} Active")
    col2.metric("Departments", len(ems.departments))
    col3.metric("Pending Leave Requests", pending_leaves)
    col4.metric("Active System Warnings", active_warnings)

    st.markdown("---")

    tab_dir, tab_att, tab_leave, tab_ai = st.tabs([
        "👥 Staff Directory & Profiles",
        "⏱️ Attendance Clock Engine",
        "🌴 Leave Approval Queue",
        "🧠 AI Workforce Predictive Analytics"
    ])

    # -----------------------------------------------------------------------
    # TAB 1: STAFF DIRECTORY & PROFILES
    # -----------------------------------------------------------------------
    with tab_dir:
        st.subheader("Enterprise Employee Rankings & Roster")
        if not df_rankings.empty:
            st.dataframe(df_rankings, use_container_width=True)
        else:
            st.info("No employee records stored in system.")

        st.markdown("---")
        st.subheader("🔍 Deep-Dive Profile Inspector")
        
        emp_options = list(ems.employees.keys())
        if emp_options:
            selected_emp_id = st.selectbox(
                "Select Employee Profile:",
                options=emp_options,
                format_func=lambda eid: f"{eid} — {ems.employees[eid].full_name} ({ems.employees[eid].role})"
            )

            if selected_emp_id:
                profile_data = ems.get_employee_profile(selected_emp_id)
                emp = ems.employees[selected_emp_id]

                c_prof1, c_prof2 = st.columns(2)
                with c_prof1:
                    st.markdown(f"**Full Name:** {emp.full_name}")
                    st.markdown(f"**Email:** `{emp.email}`")
                    st.markdown(f"**Role:** `{emp.role}`")
                    st.markdown(f"**Department:** {profile_data['department_name']}")
                
                with c_prof2:
                    st.markdown(f"**Bonus Points:** `{emp.bonus_points} pts`")
                    st.markdown(f"**Active Warnings:** `{profile_data['active_warnings']}`")
                    st.markdown(f"**Skills:** {', '.join([f'`{s}`' for s in emp.skills]) if emp.skills else 'None logged'}")
                    st.markdown(f"**Certifications:** {', '.join(emp.certifications) if emp.certifications else 'None logged'}")

                # Quick Add Employee Form
                with st.expander("➕ Register New Employee", expanded=False):
                    with st.form("create_employee_form"):
                        f_name = st.text_input("Full Name")
                        f_email = st.text_input("Email Address")
                        f_role = st.selectbox("Role", [r.value for r in RoleEnum])
                        f_dept = st.selectbox("Department ID", list(ems.departments.keys()))
                        f_skills = st.text_input("Skills (comma-separated)", placeholder="Python, SQL, Financial Audit")

                        if st.form_submit_button("Register Employee"):
                            if f_name and f_email:
                                skill_list = [s.strip() for s in f_skills.split(",")] if f_skills else []
                                created_id = ems.create_employee(f_name, f_email, f_role, f_dept, skill_list)
                                st.success(f"Registered {f_name} with ID: `{created_id}`")
                                st.rerun()
                            else:
                                st.warning("Name and Email are required.")

    # -----------------------------------------------------------------------
    # TAB 2: ATTENDANCE CLOCK ENGINE
    # -----------------------------------------------------------------------
    with tab_att:
        st.subheader("Attendance Punch Clock")
        c_clock1, c_clock2 = st.columns(2)

        with c_clock1:
            st.markdown("#### 📥 Clock In")
            clock_in_emp = st.selectbox("Select Employee for Shift Start:", emp_options, key="clk_in_sel")
            clock_in_loc = st.radio("Work Location:", [WorkLocation.OFFICE.value, WorkLocation.REMOTE.value])

            if st.button("⏱️ Clock In Shift"):
                att_id = ems.clock_in(clock_in_emp, location=clock_in_loc)
                st.success(f"Clocked In recorded! Transaction ID: `{att_id}`")
                st.rerun()

        with c_clock2:
            st.markdown("#### 📤 Clock Out")
            clock_out_emp = st.selectbox("Select Employee for Shift End:", emp_options, key="clk_out_sel")

            if st.button("⏹️ Clock Out Shift"):
                if ems.clock_out(clock_out_emp):
                    st.success(f"Clocked out successfully for `{clock_out_emp}`!")
                    st.rerun()
                else:
                    st.warning("No active open shift found for this employee today.")

        st.markdown("---")
        st.subheader("Shift Telemetry Logs")
        if ems.attendance_records:
            df_att = pd.DataFrame([asdict(a) for a in ems.attendance_records])
            st.dataframe(df_att, use_container_width=True)
        else:
            st.info("No attendance records logged for current period.")

    # -----------------------------------------------------------------------
    # TAB 3: LEAVE APPROVAL QUEUE
    # -----------------------------------------------------------------------
    with tab_leave:
        col_req, col_queue = st.columns([1, 1])

        with col_req:
            st.markdown("#### 📝 Submit Leave Request")
            with st.form("request_leave_form"):
                req_emp = st.selectbox("Requesting Employee:", emp_options, key="lev_emp_sel")
                req_type = st.selectbox("Leave Type:", [t.value for t in LeaveType])
                req_start = st.date_input("Start Date")
                req_end = st.date_input("End Date")
                req_reason = st.text_area("Reason / Description", height=70)

                if st.form_submit_button("Submit Request"):
                    req_id = ems.request_leave(
                        req_emp, 
                        req_type, 
                        req_start.strftime("%Y-%m-%d"), 
                        req_end.strftime("%Y-%m-%d"), 
                        req_reason
                    )
                    if req_id:
                        st.success(f"Leave Request `{req_id}` submitted for approval!")
                        st.rerun()
                    else:
                        st.error("Submission failed: Check employee leave balance.")

        with col_queue:
            st.markdown("#### ⏳ Pending Approval Queue")
            pending_reqs = {lid: req for lid, req in ems.leave_requests.items() if req.status == "PENDING"}

            if pending_reqs:
                for lid, req in pending_reqs.items():
                    with st.expander(f"📌 {lid}: {ems.employees[req.emp_id].full_name} ({req.leave_type})"):
                        st.markdown(f"**Duration:** {req.start_date} to {req.end_date} ({req.total_days} Days)")
                        st.markdown(f"**Reason:** {req.reason or 'N/A'}")

                        col_app, col_rej = st.columns(2)
                        with col_app:
                            if st.button("✅ Approve", key=f"app_{lid}"):
                                ems.approve_leave(lid, approver_emp_id="EMP-101", approve=True)
                                st.success(f"Approved {lid}")
                                st.rerun()
                        with col_rej:
                            if st.button("❌ Reject", key=f"rej_{lid}"):
                                ems.approve_leave(lid, approver_emp_id="EMP-101", approve=False)
                                st.info(f"Rejected {lid}")
                                st.rerun()
            else:
                st.info("No pending leave requests in queue.")

    # -----------------------------------------------------------------------
    # TAB 4: AI WORKFORCE PREDICTIVE ANALYTICS
    # -----------------------------------------------------------------------
    with tab_ai:
        st.subheader("🤖 AI Workforce Predictive Intelligence")
        st.markdown("Simulate burnout risks, evaluate promotion readiness, and calculate department skill coverage.")

        target_ai_emp = st.selectbox("Select Employee for AI Telemetry Diagnostic:", emp_options, key="ai_emp_sel")

        if target_ai_emp:
            col_b, col_p = st.columns(2)

            # Burnout Analysis Card
            with col_b:
                burnout_res = ems.predict_burnout(target_ai_emp)
                st.markdown("#### 🔥 Burnout Risk Predictor")
                b_color = "🟢" if burnout_res["risk_level"] == "LOW" else ("🟡" if burnout_res["risk_level"] == "MEDIUM" else "🔴")
                
                st.metric("Burnout Index", f"{burnout_res.get('burnout_index', 0)} / 100", f"{b_color} {burnout_res.get('risk_level', 'UNKNOWN')} Risk")
                st.caption(f"**Avg Daily Hours:** {burnout_res.get('average_daily_hours', 0.0)} hrs/day")
                st.info(f"**AI Recommendation:** {burnout_res.get('recommendation', 'N/A')}")

            # Promotion Readiness Card
            with col_p:
                promo_res = ems.predict_promotion(target_ai_emp)
                st.markdown("#### 🚀 Promotion Readiness Score")
                
                st.metric("Readiness Rating", f"{promo_res['promotion_readiness_pct']}%")
                st.info(f"**AI Assessment:** {promo_res['recommendation']}")

        st.markdown("---")
        st.subheader("🎯 Department Skill Gap Analyzer")
        dept_id_sel = st.selectbox("Select Department:", list(ems.departments.keys()))
        target_skills_input = st.text_input("Target Skills Benchmark (comma-separated):", "Python, AWS, Financial Analytics, Docker")

        if st.button("Run Skill Gap Audit"):
            benchmark_list = [s.strip() for s in target_skills_input.split(",")]
            gap_analysis = ems.analyze_skill_gaps(dept_id_sel, benchmark_list)

            st.metric("Department Skill Coverage", f"{gap_analysis['coverage_pct']}%")
            if gap_analysis["missing_skills"]:
                st.warning(f"⚠️ Missing Department Capabilities: {', '.join([f'`{s}`' for s in gap_analysis['missing_skills']])}")
            else:
                st.success("🎉 Department possesses 100% of required skill benchmarks!")
                # ---------------------------------------------------------------------------
# MODULE: AUTOMATED JOB SCHEDULER
# ---------------------------------------------------------------------------
elif app_mode == "⏱️ Automated Job Scheduler":
    st.title("⏱️ Automated Job Scheduler & Task Control")
    st.markdown("Manage automated platform recurring jobs, view job execution telemetry, run system backups, and trigger manual maintenance tasks.")

    # Top Control Bar & Telemetry Status
    col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
    
    active_jobs = scheduler_engine.list_jobs()
    exec_history = scheduler_engine.get_execution_history()
    successful_runs = sum(1 for h in exec_history if h["status"] == "SUCCESS")
    failed_runs = sum(1 for h in exec_history if h["status"] == "FAILED")

    col_stat1.metric("Scheduler Status", "RUNNING" if scheduler_engine.scheduler.running else "STOPPED")
    col_stat2.metric("Active Scheduled Jobs", len(active_jobs))
    col_stat3.metric("Successful Executions", successful_runs)
    col_stat4.metric("Failed Executions", failed_runs, delta_color="inverse")

    st.markdown("---")

    # Scheduler Engine Controls
    c_ctrl1, c_ctrl2, c_ctrl3, c_ctrl4 = st.columns(4)
    with c_ctrl1:
        if st.button("▶️ Resume Scheduler", use_container_width=True):
            scheduler_engine.resume()
            st.success("Scheduler resumed.")
            st.rerun()
    with c_ctrl2:
        if st.button("⏸️ Pause Scheduler", use_container_width=True):
            scheduler_engine.pause()
            st.warning("Scheduler paused.")
            st.rerun()
    with c_ctrl3:
        if st.button("⚡ Run System Backup Now", type="primary", use_container_width=True):
            with st.spinner("Archiving database & files..."):
                res = scheduler_engine.job_automatic_backups()
                st.success(f"Backup Complete! File: `{res['backup_file']}` ({res['files_archived']} files)")
                st.rerun()
    with c_ctrl4:
        if st.button("🧹 Clean System Temp Files", use_container_width=True):
            res = scheduler_engine.job_system_cleanup()
            st.info(f"Cleaned {res['cleaned_files']} temporary files.")
            st.rerun()

    st.markdown("---")

    tab_jobs, tab_history, tab_manual, tab_add = st.tabs([
        "📋 Active Job Queue",
        "📊 Execution Telemetry Logs",
        "⚡ Instant Manual Triggers",
        "➕ Register Custom Job"
    ])

    # -----------------------------------------------------------------------
    # TAB 1: ACTIVE JOB QUEUE
    # -----------------------------------------------------------------------
    with tab_jobs:
        st.subheader("Registered System Cron & Interval Jobs")
        if active_jobs:
            df_jobs = pd.DataFrame(active_jobs)
            df_jobs.columns = ["Job ID", "Job Name", "Next Run Time", "Pending Status"]
            st.dataframe(df_jobs, use_container_width=True)
        else:
            st.info("No active jobs registered in scheduler.")

    # -----------------------------------------------------------------------
    # TAB 2: EXECUTION TELEMETRY LOGS
    # -----------------------------------------------------------------------
    with tab_history:
        st.subheader("Recent Execution Logs & Retry History")
        if exec_history:
            df_hist = pd.DataFrame(exec_history)
            st.dataframe(df_hist, use_container_width=True)
        else:
            st.info("No job execution telemetry logged yet.")

    # -----------------------------------------------------------------------
    # TAB 3: INSTANT MANUAL TRIGGERS
    # -----------------------------------------------------------------------
    with tab_manual:
        st.subheader("Run Background Jobs On-Demand")
        
        c_m1, c_m2 = st.columns(2)
        with c_m1:
            if st.button("🩺 Execute Hardware & System Health Audit", use_container_width=True):
                health_res = scheduler_engine.job_health_monitoring()
                st.json(health_res)

            if st.button("📊 Recalculate KPI Analytics & Refresh Dashboards", use_container_width=True):
                analytics_res = scheduler_engine.job_analytics_refresh()
                st.success("KPIs & Dashboards recalculated.")

        with c_m2:
            if st.button("🔒 Run Automated Security Audit", use_container_width=True):
                sec_res = scheduler_engine.job_security_audit()
                st.json(sec_res)

            if st.button("🤖 Run AI Predictive Systems Maintenance", use_container_width=True):
                ai_maint_res = scheduler_engine.job_ai_predictive_maintenance()
                st.json(ai_maint_res)

    # -----------------------------------------------------------------------
    # TAB 4: REGISTER CUSTOM JOB
    # -----------------------------------------------------------------------
    with tab_add:
        st.subheader("Register New Interval Job")
        with st.form("add_custom_job_form"):
            job_id_input = st.text_input("Job Identifier (ID)", placeholder="job_custom_sync")
            interval_mins = st.number_input("Interval Frequency (Minutes)", min_value=1, value=30)
            
            if st.form_submit_button("Register Job"):
                if job_id_input:
                    # Example binding to cleanup or health function
                    success = scheduler_engine.add_job(
                        scheduler_engine.job_health_monitoring, 
                        trigger_type="interval", 
                        job_id=job_id_input, 
                        minutes=int(interval_mins)
                    )
                    if success:
                        st.success(f"Job `{job_id_input}` successfully registered!")
                        st.rerun()
                    else:
                        st.error("Failed to register job.")
                else:
                    st.warning("Please specify a Job ID.")
