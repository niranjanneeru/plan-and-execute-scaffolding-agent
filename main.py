import os
import operator
from typing import Annotated, List, Tuple, Union
from typing_extensions import TypedDict
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage
from langgraph.prebuilt import create_react_agent
from langgraph.graph import StateGraph, END, START
from langchain.tools import tool

# Load environment variables
load_dotenv()


# --- Define Tools ---
# Set up workspace directory for generated projects
WORKSPACE_DIR = "generated_projects"
os.makedirs(WORKSPACE_DIR, exist_ok=True)


@tool
def create_directory(path: str) -> str:
    """Create a directory in the generated_projects folder"""
    if not path.startswith("generated_projects/"):
        path = f"generated_projects/{path}"
    
    try:
        os.makedirs(path, exist_ok=True)
        return f"✓ Created directory: {path}"
    except Exception as e:
        return f"✗ Error creating directory: {str(e)}"


@tool
def create_file(path: str, content: str = "") -> str:
    """Create a file with specified content in the generated_projects folder"""
    if not path.startswith("generated_projects/"):
        path = f"generated_projects/{path}"
    
    try:
        # Create parent directories if needed
        dir_path = os.path.dirname(path)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)
        
        # Create the file with content
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        
        return f"✓ Created file: {path} ({len(content)} characters)"
    except Exception as e:
        return f"✗ Error creating file: {str(e)}"


@tool
def write_to_file(path: str, content: str) -> str:
    """Write or append content to an existing file"""
    if not path.startswith("generated_projects/"):
        path = f"generated_projects/{path}"
    
    try:
        with open(path, "a", encoding="utf-8") as f:
            f.write(content)
        return f"✓ Wrote to file: {path} ({len(content)} characters)"
    except Exception as e:
        return f"✗ Error writing to file: {str(e)}"


@tool
def move_file(source: str, destination: str) -> str:
    """Move or rename a file or directory"""
    import shutil
    
    if not source.startswith("generated_projects/"):
        source = f"generated_projects/{source}"
    if not destination.startswith("generated_projects/"):
        destination = f"generated_projects/{destination}"
    
    try:
        # Create destination directory if needed
        dest_dir = os.path.dirname(destination)
        if dest_dir:
            os.makedirs(dest_dir, exist_ok=True)
        
        shutil.move(source, destination)
        return f"✓ Moved {source} to {destination}"
    except Exception as e:
        return f"✗ Error moving file: {str(e)}"


@tool
def delete_file(path: str) -> str:
    """Delete a file or directory"""
    import shutil
    
    if not path.startswith("generated_projects/"):
        path = f"generated_projects/{path}"
    
    try:
        if os.path.isfile(path):
            os.remove(path)
            return f"✓ Deleted file: {path}"
        elif os.path.isdir(path):
            shutil.rmtree(path)
            return f"✓ Deleted directory: {path}"
        else:
            return f"✗ Path does not exist: {path}"
    except Exception as e:
        return f"✗ Error deleting: {str(e)}"


@tool
def list_directory(path: str = "generated_projects") -> str:
    """List contents of a directory"""
    if not path.startswith("generated_projects/") and path != "generated_projects":
        path = f"generated_projects/{path}"
    
    try:
        items = os.listdir(path)
        if not items:
            return f"Directory {path} is empty"
        
        result = f"Contents of {path}:\n"
        for item in sorted(items):
            item_path = os.path.join(path, item)
            if os.path.isdir(item_path):
                result += f"  📁 {item}/\n"
            else:
                size = os.path.getsize(item_path)
                result += f"  📄 {item} ({size} bytes)\n"
        return result
    except Exception as e:
        return f"✗ Error listing directory: {str(e)}"


@tool
def generate_code(description: str, language: str) -> str:
    """Generate code based on description using LLM"""
    from pydantic import BaseModel
    from langchain_core.output_parsers import PydanticOutputParser
    
    class CodeResponse(BaseModel):
        code: str
    
    parser = PydanticOutputParser(pydantic_object=CodeResponse)
    
    prompt = f"""Generate {language} code for: {description}

{parser.get_format_instructions()}

Provide clean, well-commented code."""
    
    try:
        response = llm.invoke(prompt)
        parsed = parser.parse(response.content)
        return parsed.code
    except Exception as e:
        return f"✗ Error generating code: {str(e)}"


# Combine tools for the agent
tools = [
    create_directory,
    create_file,
    write_to_file,
    move_file,
    delete_file,
    list_directory,
    generate_code,
]

# --- LLM Initialization ---
llm = ChatOpenAI(model="gpt-4", temperature=0, api_key=os.getenv("OPENAI_API_KEY"))

# --- ReAct Agent for Execution ---
# The prompt for create_react_agent is implicit or can be passed via messages
# create_react_agent directly uses the LLM and tools to form a ReAct loop.
# It expects a list of messages as input.
agent_executor = create_react_agent(llm, tools)


# --- State Definition ---
class PlanExecute(TypedDict):
    input: str  # The original user query
    plan: List[str]
    past_steps: Annotated[
        List[Tuple[str, str]], operator.add
    ]  # List of (step_description, observation)
    response: str
    messages: Annotated[
        List[BaseMessage], operator.add
    ]  # Keep track of conversation for the agent


# --- Pydantic Models for Structured Output ---
class Plan(BaseModel):
    """Plan to follow in future"""

    steps: List[str] = Field(
        description="different steps to follow, should be in sorted order"
    )


class Response(BaseModel):
    """Response to user."""

    response: str


class Act(BaseModel):
    """Action to perform."""

    action: Union[Response, Plan] = Field(
        description="Action to perform. If you want to respond to user, use Response. "
        "If you need to further use tools to get the answer, use Plan."
    )


# --- Planner Prompt and Chain ---
planner_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """For the given objective, come up with a simple step by step plan for project scaffolding. \
This plan should involve individual tasks, that if executed correctly will yield the correct answer. Do not add any superfluous steps. \
The result of the final step should be the final answer. Make sure that each step has all the information needed - do not skip steps.

Available tools:
- create_directory: Create directories
- create_file: Create files with content
- write_to_file: Add content to existing files
- move_file: Move or rename files/directories
- delete_file: Delete files or directories
- list_directory: List directory contents
- generate_code: Generate code using AI

Break down the task into clear, actionable steps using these tools.""",
        ),
        (
            "placeholder",
            "{messages}",
        ),  # Planner expects messages, not just input directly
    ]
)
planner = planner_prompt | llm.with_structured_output(Plan)

# --- Replanner Prompt and Chain ---
replanner_prompt = ChatPromptTemplate.from_template(
    """You are a task replanner. Based on the progress so far, decide what to do next.

Original objective: {input}

Original plan: {plan}

Completed steps: {past_steps}

Analyze the completed steps and decide:
1. If ALL tasks are complete and the objective is fully achieved, return a Response with a summary.
2. If there are still steps remaining, return a Plan with ONLY the steps that still need to be done.

IMPORTANT: 
- Do NOT include already completed steps in the new plan.
- Do NOT return both a response and a plan - choose ONE.
- Use the available tools: create_directory, create_file, write_to_file, move_file, delete_file, list_directory, generate_code
- Be concise and specific.
"""
)
replanner = replanner_prompt | llm.with_structured_output(Act)


# --- Graph Nodes ---
async def execute_step(state: PlanExecute):
    plan = state["plan"]
    past_steps = state.get("past_steps", [])
    current_step_description = plan[0]  # Get the current step from the plan

    print(f"\n🔧 Executing step: {current_step_description}")

    # Construct the messages for the agent_executor
    agent_messages = [
        HumanMessage(content=f"Execute this task: {current_step_description}")
    ]

    # Invoke the ReAct agent
    agent_response = await agent_executor.ainvoke({"messages": agent_messages})

    # Extract the observation from the agent's response
    messages = agent_response["messages"]

    # Find the last tool message or AI message
    observation = ""
    for msg in reversed(messages):
        if isinstance(msg, ToolMessage):
            observation = msg.content
            break
        elif isinstance(msg, AIMessage) and msg.content:
            observation = msg.content
            break

    if not observation:
        observation = "Step completed"

    print(f"✓ Result: {observation[:100]}...")

    return {
        "past_steps": [(current_step_description, observation)],
        "plan": plan[1:],  # Remove the executed step from the plan
    }


async def plan_step(state: PlanExecute):
    print(f"\n📋 Planning for: {state['input']}")
    # The planner expects a list of messages, with the user's input as a HumanMessage
    plan_output = await planner.ainvoke(
        {"messages": [HumanMessage(content=state["input"])]}
    )
    print(f"\n📝 Plan created with {len(plan_output.steps)} steps:")
    for i, step in enumerate(plan_output.steps, 1):
        print(f"  {i}. {step}")

    return {
        "plan": plan_output.steps,
    }


async def replan_step(state: PlanExecute):
    print("\n🔄 Replanning...")

    # Format past_steps for better readability
    past_steps = state.get("past_steps", [])
    formatted_steps = "\n".join([f"- {step}: {obs}" for step, obs in past_steps])

    # Create a formatted state for the replanner
    replanner_input = {
        "input": state["input"],
        "plan": "\n".join(state.get("plan", [])),
        "past_steps": formatted_steps if formatted_steps else "None",
    }

    # Replanner needs the full state to make a decision
    output = await replanner.ainvoke(replanner_input)

    if isinstance(output.action, Response):
        print(f"\n✅ Final response: {output.action.response}")
        return {
            "response": output.action.response,
        }
    else:
        print(f"\n📝 Updated plan with {len(output.action.steps)} remaining steps")
        for i, step in enumerate(output.action.steps, 1):
            print(f"  {i}. {step}")
        return {"plan": output.action.steps}


def should_end(state: PlanExecute) -> str:
    # End if a final response is generated or if the plan is empty
    if "response" in state and state["response"]:
        print("\n🎉 Workflow complete!")
        return "__end__"
    elif not state.get("plan", []):
        print("\n🎉 All steps completed!")
        return "__end__"
    else:
        return "agent"  # Continue to the agent for execution


# --- Build the Workflow ---
workflow = StateGraph(PlanExecute)

workflow.add_node("planner", plan_step)
workflow.add_node("agent", execute_step)
workflow.add_node("replan", replan_step)

workflow.add_edge(START, "planner")
workflow.add_edge("planner", "agent")
workflow.add_edge("agent", "replan")

workflow.add_conditional_edges(
    "replan",
    should_end,
    {
        "agent": "agent",  # Continue executing if more steps
        "__end__": END,  # End if the replanner says it's done
    },
)

app = workflow.compile()
config = {"recursion_limit": 50}


# --- Example Usage ---
async def run_example(inputs):
    print(f"\n{'='*60}")
    print(f"🚀 Starting Plan-Execute Agent")
    print(f"{'='*60}")
    print(f"📝 Input: {inputs['input']}")
    print(f"{'='*60}\n")

    final_state = await app.ainvoke(inputs, config=config)

    print(f"\n{'='*60}")
    print(f"📊 Final State")
    print(f"{'='*60}")
    print(f"Response: {final_state.get('response', 'No response')}")
    print(f"Steps executed: {len(final_state.get('past_steps', []))}")
    print(f"{'='*60}\n")

    return final_state


if __name__ == "__main__":
    # Test with a general knowledge question
    import asyncio
    asyncio.run(
        run_example(
            {
                "input": "Set up a Python project 'my_new_project'. It needs a 'src' folder, an empty '__init__.py' in 'src', a 'main.py' in 'src' printing 'Hello Scaffolding!', and a 'README.md' at the root with title 'My New Project'."
            }
        )
    )
