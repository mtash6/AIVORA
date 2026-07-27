import os
from typing import Optional
import whisper
from crewai import Agent, Task, Crew, Process, LLM

class MeetingIntelligenceService:
    def __init__(self, model_size: str = "base"):
        self.model_size = model_size
        self._whisper_model: Optional[whisper.Whisper] = None
        
    @property
    def whisper_model(self) -> whisper.Whisper:
        """Lazy-loads the heavy whisper architecture into memory only when execution demands it."""
        if self._whisper_model is None:
            print(f"Initializing Local Whisper Engine ({self.model_size})...")
            self._whisper_model = whisper.load_model(self.model_size)
        return self._whisper_model
        
    def transcribe_audio(self, file_path: str, language: str = "en") -> dict:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Audio record file missing at: {file_path}")
            
        # Invoking property handles memory allocations safely
        result = self.whisper_model.transcribe(file_path, language=language, fp16=False)
        return {
            "text": result.get("text", "").strip(),
            "language": result.get("language", language)
        }

    def run_agent_summarization(self, transcript: str, api_key: str) -> str:
        """Executes multi-agent summary workflows over transcripts using Gemini."""
        gemini_llm = LLM(
            model="gemini/gemini-2.0-flash", 
            api_key=api_key,
            temperature=0.3
        )
        
        minutes_specialist = Agent(
            role='Meeting Minutes Specialist',
            goal='Extract exact actions, critical items, and complete event contexts from conversation histories.',
            backstory='Expert analytics workforce coordinator specialized in clean summaries of messy, unstructured voice tracks.',
            llm=gemini_llm,
            verbose=True,
            allow_delegation=False
        )
        
        summary_task = Task(
            description=f"""Analyze the provided transcription data carefully.
            
            TRANSCRIPT DATA:
            {transcript}
            
            Format your final response in clear markdown sections:
            1. **Meeting Overview**: Core purpose.
            2. **Key Discussion Points**: Major statements grouped logically.
            3. **Decisions Made**: Central conclusions reached.
            4. **Action Items**: Explicitly assigned tasks with owners.
            5. **Open Questions**: Retained questions requiring follow-up.
            """,
            expected_output="Comprehensive, beautifully formatted markdown breakdown capturing all structural details.",
            agent=minutes_specialist
        )
        
        crew = Crew(
            agents=[minutes_specialist],
            tasks=[summary_task],
            process=Process.sequential,
            verbose=True
        )
        
        return str(crew.kickoff())