import os
import subprocess
import sys
import shutil
import time

def run_command(command, description, ignore_errors=False):
    print(f"\n--- {description} ---")
    try:
        subprocess.check_call(command, shell=True)
        print(f"[OK] {description} successful.")
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Error during: {description}")
        print(f"Error details: {e}")
        if not ignore_errors:
            return False
    return True

def get_venv_paths(venv_dir):
    return (
        os.path.join(venv_dir, "Scripts", "pip.exe"),
        os.path.join(venv_dir, "Scripts", "python.exe")
    )

def is_venv_compatible(venv_python):
    """Checks if the existing venv is using the same python version as this script."""
    try:
        # Get version string from venv python
        output = subprocess.check_output([venv_python, "--version"], text=True).strip()
        print(f"Existing Venv Version: {output}")
        
        # We specifically want to kill Python 3.14 environments or major mismatches
        if "3.14" in output or "3.13" in output:
            print("[!] Found Python 3.13/3.14 in venv. This is incompatible.")
            return False
            
        return True
    except Exception:
        return False

def install_package_with_retry(pip_exe, package):
    print(f"Installing {package}...")
    try:
        subprocess.check_call([pip_exe, "install", package])
        print(f"[OK] {package} installed.")
    except subprocess.CalledProcessError:
        print(f"[WARN] Failed to install {package}. Retrying with binary preference...")
        try:
            subprocess.check_call([pip_exe, "install", "--prefer-binary", "--no-cache-dir", package])
            print(f"[OK] {package} installed (Binary Mode).")
        except subprocess.CalledProcessError:
            return False
    return True

def main():
    sys.stdout.reconfigure(encoding='utf-8')
    
    # Check if THIS script is running on compatible python
    if sys.version_info >= (3, 13):
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print(f"CRITICAL ERROR: Running on Python {sys.version}.")
        print("This script must be launched with Python 3.10, 3.11, or 3.12.")
        print("Please run setup.bat, do not run this script directly.")
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        sys.exit(1)

    base_dir = os.path.dirname(os.path.abspath(__file__))
    venv_dir = os.path.join(base_dir, "venv")
    pip_exe, python_exe = get_venv_paths(venv_dir)

    # 1. VALIDATE AND PREPARE VENV
    venv_needs_creation = True

    if os.path.exists(venv_dir) and os.path.exists(python_exe):
        if is_venv_compatible(python_exe):
            print("[OK] Existing Virtual Environment is compatible.")
            venv_needs_creation = False
        else:
            print("[!] Existing venv is wrong version or broken. Deleting...")
            # Retry deletion a few times in case of file locks
            for _ in range(3):
                try:
                    shutil.rmtree(venv_dir)
                    break
                except Exception as e:
                    print(f"Waiting for file release... ({e})")
                    time.sleep(2)
            
            if os.path.exists(venv_dir):
                print("[CRITICAL] Could not delete 'venv' folder. Please delete it manually and restart.")
                return

    if venv_needs_creation:
        print(f"Creating new Virtual Environment using {sys.executable}...")
        run_command(f'"{sys.executable}" -m venv venv', "Creating venv")

    # 2. INSTALL DEPENDENCIES
    print("\n--- Initializing Dependencies ---")
    
    # 2.1 Upgrade Pip
    subprocess.call([python_exe, "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel"])

    # 2.2 Core Engine (Pinned for Stability)
    print("\n--- Installing Core Engine ---")
    core_deps = [
        "numpy<2.0.0", 
        "typing_extensions",
        "pydantic>=2.6.0", 
        "spacy>=3.7.5"
    ]
    for dep in core_deps:
        subprocess.call([pip_exe, "install", "--no-cache-dir", dep])

    # 2.3 Main Libraries
    dependencies = [
        "pywin32", "transformers", "torch", "sentence-transformers",
        "PyPDF2", "openpyxl", "python-docx", "reportlab", "pandas",
        "langchain", "langchain-community", "langchain-core", "langchain-huggingface",
        "trafilatura", "requests", "faiss-cpu", "chromadb", "beautifulsoup4", 
        "fake-useragent", "pyautogui", "snac", "soundfile"
    ]
    
    print("\n--- Installing Support Libraries ---")
    for dep in dependencies:
        install_package_with_retry(pip_exe, dep)

    # 4. Spacy Model
    print("\n--- Downloading Language Model ---")
    try:
        run_command(f'"{python_exe}" -m spacy download en_core_web_sm', "Downloading Spacy Model")
    except:
        print("[WARN] Model download failed.")

    # 5. Ollama Models
    print("\n--- Checking Ollama Service ---")
    try:
        subprocess.run(["ollama", "list"], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except:
        print("Starting Ollama Service...")
        subprocess.Popen(["ollama", "serve"], shell=True)
        time.sleep(10)

    for model in ["gemma2:2b", "nomic-embed-text"]:
        run_command(f"ollama pull {model}", f"Pulling AI Model: {model}")

    print("\n==========================================")
    print("       INITIALIZATION COMPLETE")
    print("==========================================")

if __name__ == "__main__":
    main()