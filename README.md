# Plan-and-Execute Scaffolding Agent

A LangGraph-based AI agent that uses a **plan-and-execute** pattern to automatically scaffold projects based on natural language descriptions. The agent breaks down complex project setup tasks into discrete steps and executes them sequentially.

## 🎯 Overview

This agent demonstrates the **Plan-and-Execute** pattern, a powerful approach where:
1. A **Planner Agent** creates a detailed execution plan
2. An **Executor Agent** executes each step of the plan one at a time
3. The system loops until all steps are completed

## 🏗️ Architecture

### Components

```
┌─────────────┐
│   Input     │
│  Request    │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│   Planner   │ ◄─── Creates detailed step-by-step plan
│    Node     │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│  Executor   │ ◄─── Executes one step at a time
│    Node     │      using tools (execute_command)
└──────┬──────┘
       │
       ▼
┌─────────────┐
│  Decision   │ ◄─── Should continue or end?
│    Logic    │
└──────┬──────┘
       │
       ├─── Continue ──► Loop back to Executor
       │
       └─── End ──────► Final State
```

### State Management

The agent maintains state using `AgentState` TypedDict:
- **`input`**: The original user request
- **`plan`**: The generated execution plan
- **`past_steps`**: List of executed steps with their outputs
- **`response`**: The most recent response from the executor

## 🔧 How It Works

### 1. **Planning Phase** (`plan_node`)
- Takes the user's natural language request
- Uses GPT-4 to generate a detailed, step-by-step plan
- Breaks down the project into executable commands like:
  - `mkdir <directory>` - Create directories
  - `touch <file>` - Create files
  - `write file <filename> content: <content>` - Write file contents

### 2. **Execution Phase** (`execute_node`)
- Receives the plan and past executed steps
- Uses **OpenAI tool calling** to invoke the `execute_command` tool
- Executes one step at a time
- Tracks each execution in `past_steps`
- Returns the result of the execution

### 3. **Decision Phase** (`should_continue`)
- Checks if execution is finished (looks for "Finished" signal)
- Compares executed steps count vs. total plan steps
- Decides whether to:
  - **Continue**: Loop back to executor for next step
  - **End**: Terminate the workflow

### 4. **Tool: execute_command**
- **Real command executor** that actually creates files and directories
- Handles commands like:
  - `mkdir <directory>` - Creates actual directories
  - `touch <file>` - Creates actual files
  - `write file <filename> content: <content>` - Writes actual content to files
  - Any other shell command - Executes via subprocess
- ⚠️ **Warning**: This will make real changes to your file system!

## 🚀 Getting Started

### Prerequisites

```bash
Python 3.8+
OpenAI API Key
```

### Installation

1. Clone the repository:
```bash
git clone https://github.com/mathew-v-keyvalue/plan-and-execute-scaffolding-agent.git
cd plan-and-execute-scaffolding-agent
```

2. Create a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install langchain langchain-openai langgraph python-dotenv
```

4. Create a `.env` file:
```bash
OPENAI_API_KEY=your_openai_api_key_here
```

### Usage

Run the agent:
```bash
python main.py
```

**Important**: All generated projects are created in the `generated_projects/` directory to keep your workspace clean and prevent accidentally overwriting your agent code!

The default example creates a Python project with:
- A `src/` directory
- An `__init__.py` file
- A `main.py` with "Hello, World!"
- A `requirements.txt` file

All files will be created inside: `generated_projects/my_new_app/`

### Customizing the Input

Modify the `initial_input` in `main.py`:

```python
initial_input = "Set up a React project with TypeScript, ESLint, and a components folder"
```

## 📊 Key Technologies

- **LangChain**: Framework for building LLM applications
- **LangGraph**: State machine library for building agent workflows
- **OpenAI GPT-4**: LLM for planning and execution
- **Tool Calling**: OpenAI's function calling for structured tool invocation

## 🔑 Key Concepts

### Plan-and-Execute Pattern

This pattern is ideal for:
- ✅ Complex, multi-step tasks
- ✅ Tasks requiring sequential execution
- ✅ Scenarios where planning ahead improves outcomes
- ✅ Tasks that benefit from explicit step tracking

### Tool Calling vs ReAct

This implementation uses **OpenAI's native tool calling** instead of the ReAct pattern:
- **Tool Calling**: Direct function invocation via OpenAI API
  - More reliable and structured
  - Better error handling
  - Simpler implementation
  
- **ReAct**: Reasoning + Acting pattern
  - Requires specific prompt format
  - More verbose with thought/action/observation cycles

### State Graph (LangGraph)

The workflow is defined as a state graph:
```python
workflow = StateGraph(AgentState)
workflow.add_node("plan", plan_node)
workflow.add_node("execute", execute_node)
workflow.set_entry_point("plan")
workflow.add_edge("plan", "execute")
workflow.add_conditional_edges("execute", should_continue, {...})
```

## 🎓 Learning Points

1. **Separation of Concerns**: Planning and execution are separate, focused agents
2. **State Management**: LangGraph manages state transitions automatically
3. **Tool Integration**: Tools extend agent capabilities beyond text generation
4. **Iterative Execution**: The loop continues until all steps are complete
5. **Error Handling**: The system can handle partial failures and continue

## 🔄 Workflow Example

```
User Input: "Create a Flask API project"

PLANNING:
├─ Step 1: mkdir flask_api
├─ Step 2: touch flask_api/app.py
├─ Step 3: write file flask_api/app.py content: <Flask boilerplate>
└─ Step 4: touch flask_api/requirements.txt

EXECUTING:
├─ Execute Step 1 → "Directory created: flask_api"
├─ Execute Step 2 → "File created: flask_api/app.py"
├─ Execute Step 3 → "Content written to flask_api/app.py"
├─ Execute Step 4 → "File created: flask_api/requirements.txt"
└─ Signal: "Finished."

RESULT: Project scaffolded successfully!
```

## 🛠️ Extending the Agent

### Add More Tools

```python
@tool
def install_dependencies(package: str) -> str:
    """Install a Python package using pip"""
    os.system(f"pip install {package}")
    return f"Installed {package}"

tools = [execute_command, install_dependencies]
```

### Add Re-planning Capability

Modify the workflow to allow the executor to request re-planning if a step fails:

```python
workflow.add_conditional_edges(
    "execute",
    should_continue,
    {"continue": "execute", "replan": "plan", "end": END}
)
```

## 📝 Notes

- ⚠️ **Real Execution**: This agent makes REAL changes to your file system - use with caution!
- **GPT-4 Model**: Uses GPT-4 for better planning capabilities (can be changed to GPT-3.5-turbo)
- **Temperature**: Set to 0 for deterministic outputs
- **Error Handling**: Includes try-catch blocks and subprocess error handling
- **Safety**: Commands have a 30-second timeout to prevent hanging
- **Tip**: Test in a dedicated directory first before using in production environments

## 🤝 Contributing

Feel free to fork, improve, and submit pull requests!

## 📄 License

MIT License

## 🔗 Resources

- [LangChain Documentation](https://python.langchain.com/)
- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [OpenAI API Documentation](https://platform.openai.com/docs/)

---

**Built with ❤️ using LangChain and LangGraph**
