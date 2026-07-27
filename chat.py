# ==============================================================================
# MODULE: chat.py
# PATH: services/chat.py
# DESCRIPTION: Core LLM interaction service leveraging the modern Google GenAI SDK
#              with automatic role translation and multimodal file ingestion layers.
# ==============================================================================

import os
from typing import List, Dict, Optional
from google import genai
from google.genai import types

class ChatService:
    def __init__(self, api_key: Optional[str] = None):
        """
        Initializes the GenAI Client wrapper. Looks for an explicitly passed key,
        falling back to system environment variables.
        """
        target_key = api_key or os.getenv("GEMINI_API_KEY")
        if not target_key:
            raise ValueError(
                "Authentication Failed: GEMINI_API_KEY is completely missing "
                "from your local machine environment or Streamlit workspace."
            )
            
        self.client = genai.Client(api_key=target_key)

    def generate_chat_response(
        self, 
        message: str, 
        history: Optional[List[Dict]] = None, 
        system_instruction: Optional[str] = None,
        file_bytes: Optional[bytes] = None,
        mime_type: Optional[str] = None
    ) -> str:
        """
        Compiles structural history logs and handles multimodal inference delivery 
        streams via the production-grade gemini-2.0-flash model.
        """
        contents = []
        
        # Format existing conversation log arrays cleanly into structural SDK Parts
        if history:
            for turn in history:
                incoming_role = turn.get("role", "user")
                
                # TRANSLATION MATRIX MAP:
                # Streamlit uses 'assistant' for UI text blocks.
                # The Google GenAI endpoint strictly expects 'model' for historical responses.
                api_role = "model" if incoming_role == "assistant" else "user"
                
                contents.append(
                    types.Content(
                        role=api_role,
                        parts=[types.Part.from_text(text=turn.get("content", ""))]
                    )
                )
        
        # Build the current user interaction frame parts
        current_parts = [types.Part.from_text(text=message)]
        
        # [NEW]: If multimodal asset data is present, attach it inline to the prompt context
        if file_bytes and mime_type:
            current_parts.append(
                types.Part.from_bytes(
                    data=file_bytes,
                    mime_type=mime_type
                )
            )
        
        # Append the user content object to the terminal history position
        contents.append(types.Content(role="user", parts=current_parts))
        
        # Construct parameters matching AIVORA automation guardrails
        config = types.GenerateContentConfig(
            temperature=0.7,
            system_instruction=system_instruction
        )
        
        try:
            response = self.client.models.generate_content(
                model='gemini-2.0-flash',
                contents=contents,
                config=config,
            )
            
            if response.text:
                return response.text
            else:
                return "⚠️ System Notice: Received an empty payload response string from the provider."
                
        except Exception as e:
            return f"Operational Engine Exception encountered while calling GenAI provider: {str(e)}"