# M.A.I.A.

M.A.I.A. Is an ever-evolving infrastructure that allows the user to call locally-downloaded and cloud-provided AI models for daily tasks, with a focus on memory retention.

## Installation

*   Double-click on **setup**

### Once finished:

*   Double-click on **MAIA** to start the script.

    **NOTE:** You can move the *MAIA* file anywhere on your PC and it should work. **Do not move** the rest of the folders and files.

---

## M.A.I.A. Features

1.  Converse with a personal AI
2.  Memory persistence and automatic prompt reinforcement
3.  Complete customization, from the model used to the *system prompt*
4.  Ability to connect to paid providers via API key
5.  Advanced tools, such as Google search and executable Python code development

---

## Startup and Initial Interface

Upon startup, an interface with **4 buttons** is shown:

1.  **Load Existing Chat** – Loads a pre-existing chat from the chat folder
2.  **Start New Chat** – Creates a new chat
3.  **Memory Handler** – Opens the memory management menu
4.  **Custom LLM Configuration** – Opens the settings for AI parameters and models

---

## Creating Chats

Chats are structured as **JSON** files in the `/output` folder and contain:

*   Content of the conversation
*   Date and time messages were sent
*   Metadata

When creating a new chat:

*   You can set a **prefix** for the name
*   A **unique code** is added to the prefix to avoid conflicts between files

When reloading an existing chat:

*   The `/output` folder is shown to manually select the file

---

## Memory Handler

The **Memory Handler** menu contains **7 buttons**:

1.  **Elaborate Daily Episodes**
    Analyzes the day's chats and extracts experiences reusable in the future.

2.  **Elaborate All Episodes**
    Analyzes all archived chats and extracts reusable experiences.

3.  **Consolidate All Episodes**
    Merges conceptually similar experiences into more optimized instructions.
    *This command is executed automatically after step 1 or 2.*

4.  **Summarize All Chats**
    Summarizes all archived chats to facilitate future information retrieval.

5.  **Summarize Missing Chats**
    Summarizes only the chats not yet processed.

6.  **Search Summaries**
    Allows querying summaries to locate chats relevant to a topic.

7.  **Reset & Reindex All Embeddings**
    Sequentially executes commands **2, 3, and 4**.

The processed data is used to:

*   Boost the user prompt
*   Provide greater context to the AI

**WARNING:**
Commands **2, 4, and 7** overwrite previous processing.
To avoid formatting errors and obtain better results, the use of a **medium-large LLM (10–30 GB)** is recommended.

**NOTE:**
The **topics.txt** file allows listing macro-categories of topics to further optimize category subdivision. Simply insert one topic per line in the text file.

---

## Custom LLM Configuration

This interface allows setting a **different model for each agent**:

1.  **Main** – Main AI that answers questions
2.  **Router** – Manages the automatic execution of advanced commands
3.  **Refiner** – Optimizes Google searches
4.  **Coder** – Generates temporary code for dataframe analysis (*Beta*)
5.  **Summarizer** – Summarizes chats for the Memory Handler
6.  **Memory Analyzer** – Extracts relevant episodes for memory

The **"Other Settings (Retrieval & Search)"** section allows configuring advanced parameters to improve memory and precision.

### Configurable parameters for each model

Clicking on the model name allows you to set:

1.  **Provider** – Model provider
2.  **Model Name** – Model name according to the provider
3.  **API Key (Cloud)**
    *   ChatOllama and HuggingFace do not require an API Key
    *   Azure requires **endpoint + API Key**, separated by `|`
        *Example:* `https://my-endpoint.com/|MY_API_KEY`
4.  **Temperature** – Controls the model's creativity
5.  **Top P (0–0.99)** – Response sampling breadth
6.  **Max Tokens** – Maximum response length
7.  **System Prompt (Instructions)** – Basic model instructions
    *It is recommended not to modify these, except for the Main agent*

Click the button at the bottom of the window to **save the configuration**.

---

## Main Panel

The main screen includes:

*   Top button bar
*   Conversation display
*   Message input field
*   Two side buttons

### Writing a prompt

1.  Click in the text space at the bottom
2.  Write the message
3.  Press **Send** or **CTRL + Enter**

A notification will confirm the prompt was sent.

### Creating a new chat

*   Click on **Clear** to open a new chat with the prefix `autochat`

### Buttons on the top bar

1.  **Google** – Google search with automatic summary of results
2.  **Write Word** – Opens Microsoft Word with the last AI response
3.  **Advanced Tasks (ON/OFF)** – Activates the Router to automate tasks
4.  **Dataframe Reader (ON/OFF)** – Enables the Coder agent for Excel/CSV files
5.  **File** – File upload and management (.pdf, .xlsx, .csv, .docx) (*Beta*)
6.  **LLM Config** – Opens the model configuration
7.  **Memory Manager** – Opens the Memory Handler
8.  **Coding Mode** – Assisted Python code development with dual AI (*Beta*)
9.  **Pinned (ON/OFF)** – Keeps the window always on top

### FIle Upload Update:
The program now allows to upload multiple files and select them for Retrieval Augmented Generation. 
It currently supports pdf and docx files, although I'm working on expanding the file types, especially to handle dataframes by automatically formatting them in TOON.
