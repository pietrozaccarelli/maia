import tkinter as tk
from tkinter import filedialog
from langchain_community.chat_models import ChatOllama
from langchain.schema import HumanMessage, SystemMessage
import re
import os
import threading
import queue
import logging
from difflib import Differ  # Import Differ for code diffing

# Configure logging
logging.basicConfig(level=logging.INFO)

# Initialize Langchain Chat Model
try:
    chat_model = ChatOllama(model="gemma3")
except Exception as e:
    logging.error(f"Error initializing ChatOllama: {e}")
    chat_model = None


class CodeEditorGUI:
    def __init__(self, master: tk.Tk):
        self.master = master
        master.title("Real-time Code Modifier")

        self.code_text = tk.Text(master, height=20, width=80)
        self.code_text.pack()

        self.entry_var = tk.StringVar()
        self.entry = tk.Entry(master, textvariable=self.entry_var, width=80)
        self.entry.pack()

        self.submit_button = tk.Button(master, text="Modify Code", command=self.submit_modification)
        self.submit_button.pack()

        self.status_var = tk.StringVar(value="Idle")
        self.status_label = tk.Label(master, textvariable=self.status_var)
        self.status_label.pack()

        self.load_button = tk.Button(master, text="Load File", command=self.load_file)
        self.load_button.pack()

        self.feedback_history: list[str] = []
        self.modification_queue: queue.Queue = queue.Queue()
        self.processing = False

    def load_file(self) -> None:
        file_path = filedialog.askopenfilename(filetypes=[("Python files", "*.py")])
        if file_path:
            try:
                with open(file_path, "r", encoding="utf-8") as file:
                    code = file.read()
                    self.code_text.delete("1.0", tk.END)
                    self.code_text.insert("1.0", code)
            except Exception as e:
                logging.error(f"Error loading file: {e}")

    def submit_modification(self) -> None:
        request = self.entry_var.get().strip()
        if not request:
            return

        # Prevent concurrent submissions
        if self.processing:
            logging.info("Still processing previous request; please wait.")
            return

        # Grab the current code from the editor and start a thread to call the LLM
        current_code = self.code_text.get("1.0", tk.END)
        # Update UI state
        self.processing = True
        self.submit_button.config(state=tk.DISABLED)
        self.status_var.set("Processing...")
        # Clear the input entry for the next prompt
        self.entry_var.set("")

        threading.Thread(target=self.call_llm, args=(request, current_code), daemon=True).start()

    def check_queue(self) -> None:
        """Poll the modification queue and update the GUI from the main thread."""
        try:
            while not self.modification_queue.empty():
                response = self.modification_queue.get_nowait()
                self.update_code(response)
        except queue.Empty:
            pass
        # Poll again
        self.master.after(200, self.check_queue)

    def call_llm(self, request: str, current_code: str) -> None:
        if chat_model is None:
            self.modification_queue.put(None)
            return

        try:
            # Build a simple system + human prompt that includes the current editor contents.
            system_msg = SystemMessage(
                content=(
                    "You are an assistant that modifies Python source code. "
                    "When asked to modify code, only return the full updated Python source. "
                    "Prefer returning the code inside a single triple-backtick Python block."
                )
            )
            human_msg = HumanMessage(
                content=(
                    f"Here is the current Python source:\n\n{current_code}\n\n"
                    f"Please modify it according to the following request and return only the updated Python source:\n{request}"
                )
            )

            messages = [system_msg, human_msg]
            # Call the chat model. Different LangChain versions may warn about
            # using __call__; that's benign for now.
            response = chat_model(messages)

            # Normalize the response into a plain string
            raw_text = None
            if hasattr(response, "content"):
                raw_text = response.content
            elif isinstance(response, str):
                raw_text = response
            elif hasattr(response, "generations"):
                try:
                    raw_text = response.generations[0][0].text
                except Exception:
                    raw_text = str(response)
            else:
                raw_text = str(response)

            # Extract code from the model output: prefer triple-backtick blocks
            cleaned_code = self.extract_code_from_response(raw_text)

            # Put a dict so the main thread can see both raw and cleaned
            self.modification_queue.put({"raw": raw_text, "code": cleaned_code})
        except Exception as e:
            logging.error(f"Error calling LLM: {e}")
            self.modification_queue.put(None)

    def update_code(self, response) -> None:
        """
        response is expected to be a dict with keys 'raw' and 'code'.
        If 'code' is present, replace the editor contents with it.
        Otherwise, append the raw response to the editor and feedback.
        """
        if response is None:
            # Reset UI state even on failure
            self.processing = False
            self.submit_button.config(state=tk.NORMAL)
            self.status_var.set("Idle")
            return

        if isinstance(response, dict):
            raw = response.get("raw")
            code = response.get("code")
        else:
            raw = str(response)
            code = None

        # Record raw response into feedback history
        self.feedback_history.append(f"AI Raw Response: {raw}")

        if code:
            # Replace editor contents with cleaned code
            self.code_text.delete("1.0", tk.END)
            self.code_text.insert("1.0", code)
        else:
            # No code found; append the raw response for visibility
            self.code_text.insert(tk.END, "\n# AI Response (no code block detected):\n")
            self.code_text.insert(tk.END, raw or "")

        # Schedule any additional GUI updates on the main thread
        self.master.after(0, self.update_gui)

        # Reset processing state and UI controls
        self.processing = False
        self.submit_button.config(state=tk.NORMAL)
        self.status_var.set("Idle")

    def extract_code_from_response(self, text: str) -> str | None:
        """
        Extract a Python code block from the model response. Returns the
        code string (without fences) if found, otherwise returns the
        stripped text if it looks like code, or None.
        """
        if not text:
            return None

        # Try to find triple-backtick blocks first (```python or ```)
        pattern = r"```(?:python)?\n([\s\S]*?)```"
        m = re.search(pattern, text, re.IGNORECASE)
        if m:
            return m.group(1).rstrip() + "\n"

        # If no fences, but looks like code (multiple lines and 'def' or 'import')
        lines = text.strip().splitlines()
        if len(lines) > 1 and any(re.search(r"^\s*(def |class |import |from )", ln) for ln in lines):
            return text.strip() + "\n"

        return None

    def update_gui(self) -> None:
        # This function will be called from the main thread for small UI updates.
        # For now it doesn't need to do anything special, but left as a hook.
        pass


def diff_code(old_code: str, new_code: str) -> str:
    """
    Compares two code strings and returns a diff.
    """
    d = Differ()
    diff_lines = list(d.compare(old_code.splitlines(keepends=True), new_code.splitlines(keepends=True)))
    return "".join(diff_lines)


# Main Application
if __name__ == "__main__":
    root = tk.Tk()
    gui = CodeEditorGUI(root)
    # Start queue polling so LLM responses are handled on the main thread
    gui.check_queue()
    root.mainloop()