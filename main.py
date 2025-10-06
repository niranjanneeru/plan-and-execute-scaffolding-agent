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


# Define a tool for executing commands (simulated for this example)
@tool
def execute_command(command: str) -> str:
    """
    Executes a given command in the context of project scaffolding.
    This is a simulated execution for demonstration.
    In a real application, this would interact with the file system (e.g., creating directories, files, writing content).
    """
    print(f"Executing: {command}")
    # Simulate file system operations
    if "mkdir" in command:
        return f"Directory created: {command.split()[-1]}"
    elif "touch" in command:
        return f"File created: {command.split()[-1]}"
    elif "write file" in command:
        parts = command.split("content:")
        filename = parts[0].replace("write file", "").strip()
        content = parts[1].strip() if len(parts) > 1 else "empty content"
        return (
            f"Content written to {filename}: '{content[:50]}...'"  # show first 50 chars
        )
    else:
        return f"Command '{command}' executed successfully (simulated)."


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
    initial_input = "Set up a Python project named 'my_new_app'. It should have a 'src' directory, an '__init__.py' inside 'src', a 'main.py' inside 'src' with a simple 'Hello, World!' print statement, and a 'requirements.txt' file at the root with 'langchain' specified."

    # The plan and execute logic will manage its internal state
    final_state = app.invoke({"input": initial_input})
    print("\n---FINAL STATE---")
    print(final_state)

    print("\n---SCAFFOLDING RESULT (SIMULATED)---")
    print("Project scaffolding process completed. Check the simulated commands above.")
