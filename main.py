import os
import operator
from typing import Annotated, List, Tuple, Union
from typing_extensions import TypedDict
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from textwrap import dedent

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import BaseMessage, HumanMessage, ToolMessage, AIMessage
from langgraph.prebuilt import create_react_agent
from langgraph.graph import StateGraph, END, START
from langchain.tools import tool

load_dotenv()

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
        dir_path = os.path.dirname(path)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)

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
def code_generation(
    description: str, language: str, is_documented: bool = False
) -> str:
    """Generate code based on description.

    Args:
        description: What the code should do
        language: Programming language (python, javascript, etc.)
        is_documented: If True, generate comprehensive documentation and comments
    """
    from pydantic import BaseModel
    from langchain_core.output_parsers import PydanticOutputParser

    class CodeResponse(BaseModel):
        code: str

    parser = PydanticOutputParser(pydantic_object=CodeResponse)

    if is_documented:
        print("  📚 Generating DOCUMENTED code...")
        prompt = dedent(
            f"""
                Generate well-documented {language} code for: {description}

                Requirements:
                1. Include comprehensive docstrings/comments explaining what the code does
                2. Add inline comments for complex logic
                3. Follow best practices and proper error handling
                4. Use type hints (for Python) or appropriate type annotations
                5. Include usage examples in comments

                {parser.get_format_instructions()}

                Provide clean, well-documented, production-ready code.
            """)
    else:
        print("  📝 Generating basic code...")
        prompt = dedent(
            f"""
                Generate {language} code for: {description}

                {parser.get_format_instructions()}

                Provide clean, functional code.
            """)

    try:
        response = llm.invoke(prompt)
        parsed = parser.parse(response.content)
        doc_status = (
            "with comprehensive documentation"
            if is_documented
            else "without documentation"
        )
        return f"✓ Generated code {doc_status}\n\n{parsed.code}"
    except Exception as e:
        return f"✗ Error generating code: {str(e)}"


tools = [
    create_directory,
    create_file,
    write_to_file,
    move_file,
    delete_file,
    list_directory,
    code_generation,
]

llm = ChatOpenAI(model="gpt-4o", temperature=0, api_key=os.getenv("OPENAI_API_KEY"))


class PlanExecute(TypedDict):
    input: str  # Original user objective/request
    plan: List[str]  # Remaining steps to execute
    past_steps: Annotated[List[Tuple[str, str]], operator.add]  # Completed steps with results
    response: str  # Final response when all tasks complete
    messages: Annotated[List[BaseMessage], operator.add]  # Chat history for agent execution

class Plan(BaseModel):
    """Plan to follow in future"""

    steps: List[str] = Field(
        description="different steps to follow, should be in sorted order"
    )

planner_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            dedent("""
                For the given objective, come up with a simple step by step plan for project scaffolding.

                Available tools:
                - create_directory: Create directories
                - create_file: Create files with content
                - write_to_file: Add content to existing files
                - move_file: Move or rename files/directories
                - delete_file: Delete files or directories
                - list_directory: List directory contents
                - code_generation: Generate code

                Break down the task into clear, actionable steps."""
            ),
        ),
        ("placeholder", "{messages}"),
    ]
)
planner = planner_prompt | llm.with_structured_output(Plan)


async def plan_step(state: PlanExecute):
    print(f"\n{'='*60}")
    print("📋 PLANNING PHASE")
    print(f"{'='*60}")
    print(f"Input: {state['input']}\n")

    plan_output = await planner.ainvoke(
        {"messages": [HumanMessage(content=state["input"])]}
    )

    print("📝 Initial Plan (code_generation steps start with is_documented=False):")
    for i, step in enumerate(plan_output.steps, 1):
        print(f"  {i}. {step}")
    print()

    return {"plan": plan_output.steps}


agent_executor = create_react_agent(llm, tools)

async def execute_step(state: PlanExecute):
    plan = state["plan"]
    current_step_description = plan[0]

    print(f"\n🔧 Executing step: {current_step_description}")

    if (
        "code_generation" in current_step_description
        and "is_documented=True" in current_step_description
    ):
        print("  ⭐ REPLANNER ENHANCED THIS STEP: Added documentation!")

    agent_messages = [
        HumanMessage(content=f"Execute this task: {current_step_description}")
    ]
    agent_response = await agent_executor.ainvoke({"messages": agent_messages})

    messages = agent_response["messages"]
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

    print(f"✓ Result: {observation[:200]}...")

    return {
        "past_steps": [(current_step_description, observation)],
        "plan": plan[1:],
    }

class Response(BaseModel):
    """Response to user."""

    response: str


class Act(BaseModel):
    """Action to perform."""

    action: Union[Response, Plan] = Field(
        description="Action to perform. If you want to respond to user, use Response. "
        "If you need to further use tools to get the answer, use Plan."
    )    

replanner_prompt = ChatPromptTemplate.from_template(
    dedent(
    """
        You are a task replanner. Based on the progress so far, decide what to do next.

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
    """)
)
replanner = replanner_prompt | llm.with_structured_output(Act)


async def replan_step(state: PlanExecute):
    print(f"\n{'='*60}")
    print("🔄 REPLANNING PHASE")
    print(f"{'='*60}")

    past_steps = state.get("past_steps", [])
    formatted_steps = "\n".join(
        [f"✓ {step}: {obs[:80]}..." for step, obs in past_steps]
    )

    remaining_plan = state.get("plan", [])

    print(f"Completed: {len(past_steps)} steps")
    print(f"Remaining: {len(remaining_plan)} steps\n")

    if remaining_plan and "code_generation" in remaining_plan[0]:
        print("🎯 DETECTED: Next step is code_generation!")
        print("   Replanner will upgrade is_documented=False → is_documented=True\n")

    replanner_input = {
        "input": state["input"],
        "plan": "\n".join([f"{i+1}. {s}" for i, s in enumerate(remaining_plan)]),
        "past_steps": formatted_steps if formatted_steps else "None completed yet",
    }

    output = await replanner.ainvoke(replanner_input)

    if isinstance(output.action, Response):
        print("✅ All tasks complete!\n")
        print(f"Final response: {output.action.response}")
        return {"response": output.action.response}
    else:
        new_plan = output.action.steps
        print("📝 Updated Plan:")
        for i, step in enumerate(new_plan, 1):
            if "is_documented=True" in step:
                print(f"  ⭐ {i}. {step}")
                print("      └─ UPGRADED: is_documented changed to True!")
            else:
                print(f"     {i}. {step}")
        print()
        return {"plan": new_plan}

def should_end(state: PlanExecute) -> str:
    if "response" in state and state["response"]:
        print("\n🎉 Workflow complete!")
        return "__end__"
    elif not state.get("plan", []):
        print("\n🎉 All steps completed!")
        return "__end__"
    else:
        return "agent"


# --- Build Workflow ---
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
    {"agent": "agent", "__end__": END},
)

app = workflow.compile()
config = {"recursion_limit": 50}


# --- Example Usage ---
async def run_agent(inputs):
    print(f"\n{'='*70}")
    print("🚀 STARTING PLAN-EXECUTE AGENT WITH REPLANNING DEMO")
    print(f"{'='*70}")
    print(f"\nObjective: {inputs['input']}")
    print("\nThis demo shows how REPLANNING enhances code generation steps:")
    print("  ⭐ After Replan: is_documented=True (documented code)")
    print(f"{'='*70}\n")

    final_state = await app.ainvoke(inputs, config=config)

    print(f"\n{'='*70}")
    print("📊 FINAL SUMMARY")
    print(f"{'='*70}")
    print(f"Total steps executed: {len(final_state.get('past_steps', []))}")
    print(f"Final response: {final_state.get('response', 'Completed')}")
    print(f"{'='*70}\n")

    return final_state


if __name__ == "__main__":
    import asyncio

    # Demo: This will clearly show the replanning enhancement
    asyncio.run(
        run_agent(
            {
                "input": "Set up a Python Flask project 'my_api'. Create a 'src' folder, generate a main.py file with a simple Flask app and health endpoint, and create a README.md."
            }
        )
    )