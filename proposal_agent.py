# ==============================================================================
# MODULE: proposal_agent.py
# PATH: services/proposal_agent.py
# DESCRIPTION: Automated business intelligence agent specializing in drafting
#              contracts, corporate emails, and customized B2B proposals.
# ==============================================================================

from services.chat import ChatService

class CorporateCommunicationsAgent:
    def __init__(self, api_key: str):
        self.api_key = api_key

    def generate_document(self, raw_context: str, doc_type: str, communication_tone: str) -> str:
        """
        Processes messy contextual operational fragments and formats them into
        highly structured, executive-ready asset documentation[cite: 1].
        """
        # Instantiate your optimized client
        chat_client = ChatService(api_key=self.api_key)
        
        # Rigorous formatting instructions
        system_guidelines = (
            f"You are the Enterprise Communication voice of the AIVORA operating system[cite: 1]. "
            f"Your objective is to generate a comprehensive, executive-grade corporate '{doc_type}' "
            f"written in a distinct '{communication_tone}' tone.\n\n"
            f"CRITICAL REQUIREMENTS:\n"
            f"1. Use clean markdown formatting (bold headers, bullet points, structured tables if necessary).\n"
            f"2. Never use generic placeholder text like '[Insert Name Here]' or 'Dear [Client]'. Interpolate logical or neutral entities if details are missing.\n"
            f"3. Do not include introductory conversational pleasantries (e.g., 'Sure, here is your proposal...'). Begin immediately with the document content.\n"
            f"4. Focus heavily on operational actionability, explicit timelines, and deliverables[cite: 1]."
        )
        
        user_message = (
            f"Draft a formalized corporate {doc_type} based strictly on the "
            f"following details and background context payload:\n\n"
            f"--- START BACKGROUND CONTEXT ---\n"
            f"{raw_context}\n"
            f"--- END BACKGROUND CONTEXT ---"
        )
        
        return chat_client.generate_chat_response(
            message=user_message,
            system_instruction=system_guidelines
        )