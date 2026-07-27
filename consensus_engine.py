# ==============================================================================
# MODULE: consensus_engine.py
# PATH: services/consensus_engine.py
# DESCRIPTION: Multi-agent persona simulator that runs executive corporate
#              deliberations using your existing ChatService module.
# ==============================================================================

import concurrent.futures
from services.chat import ChatService

class MultiAgentDecisionEngine:
    def __init__(self, api_key: str):
        self.api_key = api_key
        # Define the executive board personas and their core evaluation focus
        self.personas = {
            "📊 Chief Financial Officer (CFO)": (
                "You are the CFO. Analyze the user's business idea or problem "
                "strictly from a financial perspective: focus on estimated costs, "
                "potential ROI, capital efficiency, and revenue viability. Keep it concise."
            ),
            "🛠️ Chief Technology Officer (CTO)": (
                "You are the CTO. Analyze the user's business idea or problem "
                "strictly from a technical perspective: focus on architecture, scalability, "
                "potential tech stack challenges, implementation complexity, and development time. Keep it concise."
            ),
            "⚖️ Chief Risk Officer (CRO)": (
                "You are the CRO. Analyze the user's business idea or problem "
                "strictly from a risk perspective: focus on operational liabilities, compliance issues, "
                "security blindspots, market threats, and worst-case failure modes. Keep it concise."
            )
        }

    def consult_board(self, proposal: str) -> dict:
        """Runs the proposal across all 3 personas simultaneously using threads."""
        board_responses = {}

        def run_agent_analysis(persona_title, system_instruction):
            try:
                # Reuse your existing optimized ChatService module
                client = ChatService(api_key=self.api_key)
                response = client.generate_chat_response(
                    message=proposal,
                    system_instruction=system_instruction
                )
                return persona_title, response
            except Exception as e:
                return persona_title, f"Failed to acquire board assessment: {str(e)}"

        # Execute all three AI agents in parallel to save time
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(run_agent_analysis, title, instruction) 
                for title, instruction in self.personas.items()
            ]
            for future in concurrent.futures.as_completed(futures):
                title, result = future.result()
                board_responses[title] = result

        return board_responses