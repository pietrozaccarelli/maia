import win32com.client as win32
import win32gui
import win32con
import pyautogui
import time
import re


# ------------------------- Utility Functions -------------------------
def bring_window_to_front(hwnd: int):
    """Bring the window with the given handle to the front."""
    try:
        win32gui.ShowWindow(hwnd, win32con.SW_RESTORE)
        win32gui.SetForegroundWindow(hwnd)
        time.sleep(0.2)
    except Exception:
        pyautogui.keyDown('alt')
        pyautogui.press('esc')
        pyautogui.keyUp('alt')
        time.sleep(0.2)

def get_word_windows():
    """Return a list of tuples (hwnd, title) for all visible Word windows."""
    hwnds = []
    def callback(hwnd, _):
        if win32gui.IsWindowVisible(hwnd):
            title = win32gui.GetWindowText(hwnd)
            if title and "Word" in title:
                hwnds.append((hwnd, title))
        return True
    win32gui.EnumWindows(callback, None)
    return hwnds

# ------------------------- Markdown Typing Functions -------------------------
def type_markdown_line(selection, line: str):
    """Type a single line of markdown into the Word selection object."""
    line = line.rstrip("\n")

    # HEADINGS
    if line.startswith("#"):
        level = len(line.split(" ")[0])
        text = line[level+1:] if len(line) > level else ""
        sizes = {1: 24, 2: 20, 3: 18}
        selection.Font.Size = sizes.get(level, 16)
        selection.Font.Bold = True
        selection.TypeText(text + "\n")
        selection.TypeParagraph()
        selection.Font.Bold = False
        return

    # UNORDERED LIST
    if line.startswith("- ") or line.startswith("* "):
        selection.TypeText("• ")
        text = line[2:]
    # ORDERED LIST
    elif re.match(r"\d+\.\s", line):
        number, text = line.split(". ", 1)
        selection.TypeText(f"{number}. ")
    else:
        text = line
        selection.Font.Size = 12

    # INLINE FORMATTING
    while True:
        bold_match = re.search(r"\*\*(.+?)\*\*", text)
        italic_match = re.search(r"\*(.+?)\*", text)

        if bold_match:
            pre = text[:bold_match.start()]
            selection.TypeText(pre)
            selection.Font.Bold = True
            selection.TypeText(bold_match.group(1))
            selection.Font.Bold = False
            text = text[bold_match.end():]
        elif italic_match:
            pre = text[:italic_match.start()]
            selection.TypeText(pre)
            selection.Font.Italic = True
            selection.TypeText(italic_match.group(1))
            selection.Font.Italic = False
            text = text[italic_match.end():]
        else:
            selection.TypeText(text)
            break

    selection.TypeText("\n")

def type_markdown_to_word(selection, markdown_string: str):
    """Type the full markdown string into the Word selection object."""
    for line in markdown_string.splitlines():
        if line.strip() == "":
            selection.TypeParagraph()
        else:
            type_markdown_line(selection, line)

# ------------------------- CrewAI Tool -------------------------

def writeword(markdown_input: str):
    """
    Open Microsoft Word and type the given markdown input into a new document.

    Args:
        markdown_input (str): Markdown text to type into Word.
    """
    # Start Word
    word = win32.gencache.EnsureDispatch("Word.Application")
    word.Visible = True

    doc = word.Documents.Add()
    time.sleep(1)

    # Bring Word to front
    word_windows = get_word_windows()
    target_hwnd = None
    for hwnd, title in word_windows:
        if doc.Name in title:
            target_hwnd = hwnd
            break

    if target_hwnd:
        bring_window_to_front(target_hwnd)
    else:
        print("Warning: Could not find the Word window.")

    selection = word.Selection
    type_markdown_to_word(selection, markdown_input)

    print("Markdown typed into Word successfully!")


