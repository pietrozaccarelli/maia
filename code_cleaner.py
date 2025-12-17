from datetime import datetime
import os
import re

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SCRIPT_DIR = os.path.join(BASE_DIR, "scripts")

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")


def clean_code(code):
    if not os.path.exists(SCRIPT_DIR):
        os.makedirs(SCRIPT_DIR)

    pattern = r"```python\s*(.*?)\s*```"

    match = re.search(pattern, code, re.DOTALL)

    if match:
        cleaned_code = match.group(1).strip()
    else:
        cleaned_code = code.strip()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{SCRIPT_DIR}\\pythonscript_{timestamp}.py"

    with open(filename, "w", encoding="utf-8") as f:
        f.write(cleaned_code)

    print(f"Cleaned script saved as: {filename}")
    return filename
