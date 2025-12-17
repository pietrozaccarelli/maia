import re

def extract_python_code(ai_response: str) -> str:
    """
    Extracts Python code from an AI-style response string.
    It handles cases with triple backticks, '```python', or inline code blocks.
    """
    # Look for fenced code blocks first (```python ... ```)
    matches = re.findall(r"```(?:python)?\s*([\s\S]*?)```", ai_response, re.IGNORECASE)
    if matches:
        # Join if there are multiple code blocks
        return "\n\n".join(m.strip() for m in matches)
    
    # If no fenced code blocks, fall back to inline backticks
    matches = re.findall(r"`([^`]+)`", ai_response)
    if matches:
        return "\n".join(m.strip() for m in matches)
    
    # If nothing found, return the whole string as-is (last resort)
    return ai_response.strip()

