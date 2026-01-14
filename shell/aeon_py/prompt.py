from typing import List, Dict, Any

class PromptEngine:
    """
    Constructs the final text prompt for the LLM by combining:
    1. System Instructions
    2. Spatial Context (Atlas)
    3. Episodic History (Trace)
    4. Retrieved Knowledge (RAG)
    """
    
    def build(self, 
              trace_history: List[Dict[str, Any]], 
              active_context: Dict[str, Any], 
              retrieved_knowledge: List[Dict[str, Any]]) -> str:
        """
        Builds a structured prompt with explicit delimiters.
        
        Args:
            trace_history: List of recent conversation turns (User/System nodes).
            active_context: Metadata about the currently "active" room/concept in Atlas.
            retrieved_knowledge: List of top-k search results from Atlas (RAG).
            
        Returns:
            Formatted string for the LLM.
        """
        
        # 1. Format Knowledge (RAG)
        knowledge_str = self._format_knowledge(retrieved_knowledge)
        
        # 2. Format History
        history_str = self._format_history(trace_history)
        
        # 3. Get User Input (last item in history expected to be user, or passed separately?)
        # Convention: trace_history includes the LATEST user input as the last UserNode.
        latest_input = ""
        if trace_history and trace_history[-1]['type'] == 'UserNode':
            latest_input = trace_history[-1]['text']
        else:
            # Fallback or empty if system initiated - simplified for now
            latest_input = "(No input)"

        return f"""
### SYSTEM MEMORY
Current Location: {active_context.get('metadata', 'Unknown Location')}
Status: Use the provided knowledge to answer the user's question accurately.

### RELEVANT KNOWLEDGE
{knowledge_str}

### CONVERSATION HISTORY
{history_str}

### USER INPUT
{latest_input}
"""

    def _format_knowledge(self, knowledge: List[Dict[str, Any]]) -> str:
        if not knowledge:
            return "(No relevant knowledge found)"
            
        lines = []
        for idx, item in enumerate(knowledge, 1):
            # Atlas result structure varies, assuming 'preview' or metadata lookup needed
            # For Phase 5 simplified RAG:
            content = item.get('content', str(item.get('preview', '')))
            lines.append(f"[{idx}] {content}")
        return "\n".join(lines)

    def _format_history(self, history: List[Dict[str, Any]]) -> str:
        if not history:
            return "(No history)"
            
        lines = []
        for node in history:
            role = "User" if node['type'] == 'UserNode' else "Aeon"
            text = node['text']
            lines.append(f"{role}: {text}")
            
        # Don't include the very last one if it's the current user input?
        # The prompt template has a separate "USER INPUT" section.
        # So we should exclude the LAST item if it is the current user input.
        # Actually a cleaner way is: History is PAST. User Input is CURRENT.
        # But Trace.get_recent_history() likely returns everything including the just-added node.
        
        # Let's simple-slice: if last is User, pop it for the specific slot.
        # NOTE: logic in `loop.py` will dictate exactly what is passed. 
        # Here we just format what is given. 
        # But to avoid duplication, let's assume `trace_history` passed here is PURE HISTORY (excluding current turn)
        # OR we handle duplication prevention here.
        
        return "\n".join(lines[:-1]) if lines and history[-1]['type'] == 'UserNode' else "\n".join(lines)
