import sys
import os
import subprocess
import threading
import tkinter as tk
from tkinter import ttk, messagebox
import importlib.util

# Configuration
MAIN_SCRIPT_NAME = "maia" # maia.py
SETUP_SCRIPT_NAME = "initialize_maia.py" 
VENV_DIR = "venv"

# Force UTF-8 for subprocess communication
os.environ["PYTHONUTF8"] = "1"

class RecoveryWindow:
    def __init__(self, root, error_message):
        self.root = root
        self.root.title("M.A.I.A. - Setup Required")
        self.root.geometry("600x450")
        self.root.configure(bg="#1e1e1e")

        # Header
        header_frame = tk.Frame(root, bg="#1e1e1e")
        header_frame.pack(pady=20)
        
        tk.Label(header_frame, text="[!] Initialization Failed", 
                 font=("Segoe UI", 16, "bold"), fg="#e74c3c", bg="#1e1e1e").pack()
        
        tk.Label(header_frame, text="Missing dependencies or environment configuration detected.", 
                 font=("Segoe UI", 10), fg="#aaaaaa", bg="#1e1e1e").pack(pady=5)

        # Console Output Area
        self.log_text = tk.Text(root, height=12, bg="#2d2d2d", fg="#edfde2", 
                                font=("Consolas", 9), state="disabled", relief="flat")
        self.log_text.pack(padx=20, pady=10, fill=tk.BOTH, expand=True)

        self.log_message(f"System Error: {error_message}\n")
        self.log_message("Please click 'Initialize Setup' to fix this automatically.\n")

        # Action Area
        btn_frame = tk.Frame(root, bg="#1e1e1e")
        btn_frame.pack(pady=20, fill=tk.X)

        self.setup_btn = tk.Button(btn_frame, text="Initialize Setup", 
                                   command=self.run_setup, 
                                   bg="#2ecc71", fg="white", 
                                   font=("Segoe UI", 11, "bold"), 
                                   padx=20, pady=10, relief="flat")
        self.setup_btn.pack()

        self.status_label = tk.Label(root, text="Waiting for user...", 
                                     bg="#1e1e1e", fg="#555555", font=("Segoe UI", 9, "italic"))
        self.status_label.pack(pady=(0, 10))

    def log_message(self, msg):
        self.log_text.config(state="normal")
        self.log_text.insert(tk.END, str(msg) + "\n")
        self.log_text.see(tk.END)
        self.log_text.config(state="disabled")

    def run_setup(self):
        """Executes the initialize_maia.py script in a separate thread."""
        if not os.path.exists(SETUP_SCRIPT_NAME):
            messagebox.showerror("Error", f"Could not find '{SETUP_SCRIPT_NAME}' in this directory.")
            return

        self.setup_btn.config(state="disabled", text="Installing...", bg="#7f8c8d")
        self.status_label.config(text="Running installation scripts...", fg="#f39c12")
        
        # Start background thread
        threading.Thread(target=self._execute_script, daemon=True).start()

    def _execute_script(self):
        try:
            cmd = [sys.executable, SETUP_SCRIPT_NAME]
            
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=False, 
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
            )

            for line in process.stdout:
                # Manually decode to prevent charmap crashes
                decoded_line = line.decode('utf-8', errors='replace').strip()
                self.root.after(0, lambda l=decoded_line: self.log_message(l))
            
            process.wait()

            if process.returncode == 0:
                self.root.after(0, self._on_success)
            else:
                self.root.after(0, self._on_failure)

        except Exception as e:
            self.root.after(0, lambda: self.log_message(f"CRITICAL ERROR: {e}"))
            self.root.after(0, self._on_failure)

    def _on_success(self):
        self.status_label.config(text="Setup Complete!", fg="#2ecc71")
        self.setup_btn.config(text="Relaunch M.A.I.A.", command=self.relaunch, bg="#3498db", state="normal")
        messagebox.showinfo("Success", "Dependencies installed successfully.\nPlease relaunch the application.")

    def _on_failure(self):
        self.status_label.config(text="Setup Failed.", fg="#e74c3c")
        self.setup_btn.config(state="normal", text="Retry Setup", bg="#e74c3c")

    def relaunch(self):
        self.root.destroy()
        venv_python = os.path.join(VENV_DIR, "Scripts", "python.exe")
        if os.path.exists(venv_python):
            subprocess.Popen([venv_python, "launcher.py"])
        else:
            subprocess.Popen([sys.executable, "launcher.py"])
        sys.exit()


def check_dependencies():
    """
    Tries to import the critical libraries used in maia.py.
    """
    critical_libs = [
        "langchain_community", 
        "sentence_transformers", 
        "chromadb", 
        "PyPDF2", 
        "pandas", 
        "reportlab",
        "openpyxl",
        "spacy",
        "win32com" # Added for pywin32 support
    ]
    
    missing = []
    
    for lib in critical_libs:
        if importlib.util.find_spec(lib) is None:
            missing.append(lib)
            
    if missing:
        return False, f"Missing modules: {', '.join(missing)}"
    
    return True, None


if __name__ == "__main__":
    is_ready, error_msg = check_dependencies()

    if is_ready:
        print("Environment healthy. Launching M.A.I.A...")
        try:
            import maia 
            maia.main()
        except ImportError as e:
            root = tk.Tk()
            gui = RecoveryWindow(root, str(e))
            root.mainloop()
        except Exception as e:
            print(f"Runtime Error: {e}")
            input("Press Enter to exit...")
    else:
        root = tk.Tk()
        gui = RecoveryWindow(root, error_msg)
        root.mainloop()