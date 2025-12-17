import subprocess
import json
import time
from typing import Optional, Tuple, List
from langchain_community.chat_models import ChatOllama
from code_cleaner import clean_code
import os
import re

def run_ollama_model(model_name: str, prompt: str, temp: float) -> str:
    """
    Run an Ollama model and return the response.
    """
    try:
        client = ChatOllama(model=model_name, temperature=temp)
        result = ""
        for chunk in client.stream(prompt):
            content = chunk.content
            print(content, end="", flush=True)
            result += content
        return result.strip()
    except Exception as e:
        print(f"\nError running model {model_name}: {e}")
        return ""

def extract_code_from_response(response: str) -> str:
    """
    Extract Python code from model response.
    """

    # Correct regex pattern to match code blocks
    pattern = r"```python\s*(.*?)\s*```"

    matches = re.findall(pattern, response, re.DOTALL)

    if matches:
        return matches[-1].strip()

    lines = response.strip().split('\n')
    python_keywords = ['import', 'from', 'def', 'class', 'if', 'for', 'while', 'try', 'with']
    if lines and any(lines[0].strip().startswith(kw) for kw in python_keywords):
        return response.strip()

    if "DONE" in response:
        done_index = response.find("DONE")
        potential_code = response[done_index + 4:].strip()
        if potential_code:
            return extract_code_from_response(potential_code)

    return response.strip()



def check_if_done(response: str) -> Tuple[bool, str]:
    """
    Check if the response indicates completion.
    """
    if "DONE" in response.upper():
        code = extract_code_from_response(response)
        return True, code
    return False, response

def codeduo(
    user_request: str,
    model1: str = "qwen3-coder:latest",
    model2: str = "qwen3-coder:latest",
    temp1: float = 0.7,
    temp2: float = 0.4,
    max_iterations: int = 5
) -> str:
    """
    Simulate a collaborative AI duo using Ollama.
    """
    current_code = ""
    iteration = 0
    feedback_history: List[str] = []
    print(f"Starting collaborative AI duo for request: {user_request}\n")
    print(f"Models: {model1} (temp={temp1}) and {model2} (temp={temp2})")
    print(f"Max iterations: {max_iterations}\n")

    while iteration < max_iterations:
        iteration += 1
        print(f"\n{'='*50}")
        print(f"ITERATION {iteration}/{max_iterations}")
        print(f"{'='*50}")

        # AI 1: Generate or improve the code
        if iteration == 1:
            prompt1 = f"""You are an expert Python developer.
            User Request: {user_request}
            Your task: Write complete, working Python code that satisfies the user's request.
            - Include all necessary imports
            - Ensure all variables and functions are properly defined
            - Handle edge cases and errors
            - If async functions are needed, implement them correctly
            - Add helpful comments
            Provide the complete Python code."""
        else:
            prompt1 = f"""You are an expert Python developer improving existing code.
            User Request: {user_request}
            Current Code:
            ```python
            {current_code}
            ```
            Previous Feedback:
            {chr(10).join(f"- {fb}" for fb in feedback_history[-3:])}
            Your task:
            1. Address ALL feedback points
            2. Fix any errors or issues
            3. Ensure the code fully satisfies the user's request
            4. Verify all functions and variables are correctly implemented
            5. If async is needed, ensure proper implementation
            If the code is now perfect and addresses all concerns, respond with 'DONE' followed by the final code.
            Otherwise, provide the improved code."""
        print(f"\nAI 1:\n{model1} is generating code...")
        response1 = run_ollama_model(model1, prompt1, temp1)
        is_done, code_or_response = check_if_done(response1)
        current_code = extract_code_from_response(code_or_response)

        if is_done:
            print(f"\n\n✓ {model1} indicates the code is complete!")
            # AI 2 must also confirm
            prompt2 = f"""You are an expert Python code reviewer.
            User Request: {user_request}
            Code to Review:
            ```python
            {current_code}
            ```
            Your task:
            1. Thoroughly analyze the code for correctness, completeness, and best practices.
            2. If the code is perfect and ready, respond with 'DONE'.
            3. Otherwise, provide specific, actionable feedback for improvement."""
            print(f"\n\nAI 2:\n{model2} is reviewing the code for final approval...")
            response2 = run_ollama_model(model2, prompt2, temp2)
            is_done, feedback = check_if_done(response2)
            if is_done:
                print(f"\n\n✓ Both AIs agree the code is complete and correct!")
                break
            else:
                print(f"\n\n⚠ {model2} found issues. Continuing iteration...")
                feedback_history.append(feedback)
                continue

        # AI 2: Analyze and provide feedback
        prompt2 = f"""You are an expert Python code reviewer.
        User Request: {user_request}
        Code to Review:
        ```python
        {current_code}
        ```
        Previous Feedback History: {feedback_history[-2:] if feedback_history else "None"}
        Your task:
        1. Thoroughly analyze the code for correctness, completeness, and best practices.
        2. If the code is perfect and ready, respond with 'DONE'.
        3. Otherwise, provide specific, actionable feedback for improvement."""
        print(f"\n\nAI 2:\n{model2} is reviewing the code...")
        response2 = run_ollama_model(model2, prompt2, temp2)
        is_done, feedback = check_if_done(response2)
        if is_done:
            print(f"\n\n✓ Both AIs agree the code is complete and correct!")
            break
        feedback_history.append(feedback)
        print(f"\n\nFeedback added to history for next iteration.")

    if iteration >= max_iterations:
        print(f"\n⚠ Maximum iterations ({max_iterations}) reached.")

    # Clean and save the code
    print(f"\n\n{'='*50}")
    print("FINALIZING CODE")
    print(f"{'='*50}")
    try:
        filename = clean_code(current_code)
        print(f"\n✓ Cleaned code saved to: {filename}")
    except Exception as e:
        print(f"\n⚠ Code cleaning failed: {e}")
        print("  Using uncleaned code")
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"scripts/pythonscript_{timestamp}.py"
        os.makedirs("scripts", exist_ok=True)
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(current_code)
        print(f"\n✓ Code saved to: {filename}")

    print(f"\n{'='*50}")
    print("FINAL CODE:")
    print(f"{'='*50}")
    print(current_code)
    print(f"{'='*50}\n")

    return filename

if __name__ == "__main__":
    print("=" * 60)
    print("CODEDUO - Collaborative AI Code Generation")
    print("=" * 60)
    user_request = input("\nEnter your request for a Python script: ").strip()
    if not user_request:
        print("\n❌ No request provided. Exiting.")
        exit(1)
    use_custom = input("\nUse custom settings? (y/N): ").strip().lower() == 'y'
    if use_custom:
        model1 = input(f"First model name (default: qwen3-coder:latest): ").strip() or "qwen3-coder:latest"
        model2 = input(f"Second model name (default: qwen3-coder:latest): ").strip() or "qwen3-coder:latest"
        temp1 = float(input(f"Temperature for first model (default: 0.7): ").strip() or "0.7")
        temp2 = float(input(f"Temperature for second model (default: 0.4): ").strip() or "0.4")
        max_iter = int(input(f"Max iterations (default: 5): ").strip() or "5")
        code_filename = codeduo(user_request, model1, model2, temp1, temp2, max_iter)
    else:
        code_filename = codeduo(user_request)
    print(f"\n✅ Process complete!")
    print(f"📄 Final code saved to: {code_filename}")
    print(f"\nYou can run it with: python {code_filename}")
