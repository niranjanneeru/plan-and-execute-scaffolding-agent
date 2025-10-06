import os
from typing import TypedDict, List
from langchain_core.messages import BaseMessage
from dotenv import load_dotenv
load_dotenv()

class AgentState(TypedDict):
    plan: str
    past_steps: List[str]
    response: str
    input: str  # Add this line
from langchain_openai import ChatOpenAI
from langchain.agents import tool

# Initialize the OpenAI LLM
llm = ChatOpenAI(
    model="gpt-4",
    temperature=0,
    api_key=os.getenv("OPENAI_API_KEY")
)  # You can adjust the model and temperature


# Global variable to store the working directory for generated projects
WORKSPACE_DIR = None

def set_workspace_directory(workspace_dir: str):
    """Set the workspace directory where projects will be generated"""
    global WORKSPACE_DIR
    WORKSPACE_DIR = workspace_dir
    os.makedirs(workspace_dir, exist_ok=True)
    print(f"📁 Workspace set to: {os.path.abspath(workspace_dir)}")

# Define a tool for executing commands (REAL execution)
@tool
def execute_command(command: str) -> str:
    """
    Executes a given command in the context of project scaffolding.
    This will actually create directories, files, and write content to the file system.
    All operations are performed in the workspace directory to avoid overwriting the agent code.
    """
    import subprocess
    import pathlib
    
    if WORKSPACE_DIR is None:
        return "✗ Error: Workspace directory not set. Call set_workspace_directory() first."
    
    print(f"Executing: {command}")
    
    try:
        if command.startswith("mkdir"):
            # Create directory
            dir_name = command.replace("mkdir", "").strip()
            target_path = os.path.join(WORKSPACE_DIR, dir_name)
            pathlib.Path(target_path).mkdir(parents=True, exist_ok=True)
            return f"✓ Directory created: {target_path}"
            
        elif command.startswith("touch"):
            # Create file
            filename = command.replace("touch", "").strip()
            target_path = os.path.join(WORKSPACE_DIR, filename)
            pathlib.Path(target_path).parent.mkdir(parents=True, exist_ok=True)
            pathlib.Path(target_path).touch()
            return f"✓ File created: {target_path}"
            
        elif command.startswith("write file"):
            # Write content to file
            parts = command.split("content:", 1)
            filename = parts[0].replace("write file", "").strip()
            content = parts[1].strip() if len(parts) > 1 else ""
            
            target_path = os.path.join(WORKSPACE_DIR, filename)
            
            # Create parent directories if needed
            pathlib.Path(target_path).parent.mkdir(parents=True, exist_ok=True)
            
            # Write content to file
            with open(target_path, 'w') as f:
                f.write(content)
            
            return f"✓ Content written to {target_path} ({len(content)} characters)"
            
        else:
            # For other commands, execute them in the workspace directory
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=30,
                cwd=WORKSPACE_DIR  # Execute in workspace directory
            )
            
            if result.returncode == 0:
                output = result.stdout.strip() if result.stdout else "Command executed successfully"
                return f"✓ {output}"
            else:
                error = result.stderr.strip() if result.stderr else "Unknown error"
                return f"✗ Error: {error}"
                
    except Exception as e:
        return f"✗ Error executing command: {str(e)}"


tools = [execute_command]
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langgraph.graph import StateGraph, END

# Prompt for the planning agent
planner_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an expert project scaffolder. Your task is to create a detailed plan to set up a new project based on the user's request. Break down the project setup into discrete, executable steps using commands like 'mkdir', 'touch', 'write file <filename> content: <content>'.",
        ),
        (
            "user",
            "{input}\n\nExisting steps: {past_steps}\n\nCreate the next set of detailed steps. Do not start with markdown code block for the plan. Respond with only the plan.",
        ),
    ]
)

# Planner agent
planner_agent = (
    planner_prompt | llm.bind(stop=["\nObservation"]) | RunnablePassthrough()
)


def plan_node(state: AgentState):
    print("---PLANNING---")
    # Get the current plan or initialize it
    current_plan = state.get("plan", "")
    past_steps = state.get("past_steps", [])

    # If no plan exists, generate one. Otherwise, continue with the existing plan.
    if not current_plan:
        response = planner_agent.invoke({"input": state["input"], "past_steps": ""})
        # The plan might come with a header, extract only the plan steps
        if "Plan:" in response.content:
            plan_content = response.content.split("Plan:", 1)[1].strip()
        else:
            plan_content = response.content.strip()

        return {"plan": plan_content, "past_steps": []}
    else:
        # If there's an existing plan, we're likely in a refinement step or just continuing execution
        return {"plan": current_plan, "past_steps": past_steps}


# Prompt for the executor agent - using tool calling instead of ReAct
executor_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an expert project scaffolder. Your task is to execute the given steps from the plan. Use the execute_command tool to run commands. Only execute one step at a time. After executing a step, report what was done. If all steps are completed, respond with 'Finished.'",
        ),
        (
            "user",
            "Plan:\n{plan}\n\nPast steps executed:\n{past_steps}\n\nExecute the next step from the plan that hasn't been executed yet.",
        ),
    ]
)

# Executor agent (using tool calling)
from langchain_core.runnables import RunnableLambda


def _get_executor_chain(llm, tools):
    # Bind tools to the LLM for tool calling
    llm_with_tools = llm.bind_tools(tools)
    return executor_prompt | llm_with_tools


# This wrapper allows the executor to be used in our graph
def execute_node(state: AgentState):
    print("---EXECUTING---")
    plan = state["plan"]
    past_steps = state.get("past_steps", [])
    
    # Get the executor chain
    executor_chain = _get_executor_chain(llm, tools)
    
    # Invoke the executor with the plan and past steps
    response = executor_chain.invoke({
        "plan": plan,
        "past_steps": "\n".join(past_steps) if past_steps else "None"
    })

    # Check if the response contains tool calls
    if hasattr(response, 'tool_calls') and response.tool_calls:
        tool_call = response.tool_calls[0]
        tool_name = tool_call['name']
        tool_input = tool_call['args']
        
        # Execute the tool
        if tool_name == 'execute_command':
            command = tool_input.get('command', '')
            tool_output = execute_command.invoke(command)
            past_steps.append(f"Executed: {command} -> {tool_output}")
            return {"past_steps": past_steps, "response": tool_output}
    
    # Check if execution is finished
    output = response.content if hasattr(response, 'content') else str(response)
    if "Finished" in output or "completed" in output.lower() or "all steps" in output.lower():
        return {"response": "Finished.", "past_steps": past_steps}
    
    # If no tool call and not finished, add the response to past steps
    past_steps.append(f"Response: {output}")
    return {"past_steps": past_steps, "response": output}


def should_continue(state: AgentState):
    print("---DECIDING---")
    if "Finished." in state.get("response", ""):
        print("---FINISHED EXECUTION---")
        return "end"

    # Check if there are still steps in the plan that haven't been executed
    plan_steps = [step.strip() for step in state["plan"].split("\n") if step.strip()]
    executed_count = len(state["past_steps"])

    # Simple heuristic: if we have executed fewer steps than the plan has, continue.
    # A more robust check would involve parsing the plan and executed steps.
    if executed_count < len(plan_steps) and "Observation:" in state.get("response", ""):
        # If an observation was just made, we likely executed a step and should re-evaluate.
        # This is a bit simplistic; a more advanced agent might re-plan based on observations.
        return "continue"
    elif "Action: execute_command" in state.get("response", ""):
        return "continue"  # If an action was just decided, continue to execute
    else:
        # If no clear signal to end or continue, it might mean more planning or a final response.
        # For simplicity, if the plan is exhausted and no "Finished" signal, we'll assume end.
        if executed_count >= len(plan_steps) and not state.get("response", "").strip():
            return "end"
        return "continue"  # Continue if not explicitly finished
workflow = StateGraph(AgentState)

# Add nodes
workflow.add_node("plan", plan_node)
workflow.add_node("execute", execute_node)

# Set the entry point
workflow.set_entry_point("plan")

# Define edges
# From plan, always go to execute
workflow.add_edge("plan", "execute")

# From execute, decide whether to continue or end
workflow.add_conditional_edges(
    "execute",
    should_continue,
    {"continue": "execute", "end": END},  # Loop back to execute more steps
)

# Compile the graph
app = workflow.compile()
# Example Usage
if __name__ == "__main__":
    # Set up a dedicated workspace directory for generated projects
    set_workspace_directory("./generated_projects")
    
    initial_input = "Set up a Python project named 'my_new_app'. It should have a 'src' directory, an '__init__.py' inside 'src', a 'main.py' inside 'src' with a simple 'Hello, World!' print statement, and a 'requirements.txt' file at the root with 'langchain' specified."

    print("🚀 Starting Plan-and-Execute Agent...")
    print(f"📝 Task: {initial_input}\n")
    
    # The plan and execute logic will manage its internal state
    final_state = app.invoke({"input": initial_input})
    
    print("\n" + "="*60)
    print("✅ SCAFFOLDING COMPLETED!")
    print("="*60)
    print("\n📊 Final State:")
    print(f"  - Plan: {final_state.get('plan', 'N/A')[:100]}...")
    print(f"  - Steps Executed: {len(final_state.get('past_steps', []))}")
    print(f"  - Status: {final_state.get('response', 'N/A')}")
    print(f"\n💡 Check the 'generated_projects' directory to see your new project!")
