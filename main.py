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
# Project scaffolding tool
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


# Combine tools for the agent
tools = [execute_command]

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
Ensure steps involve 'mkdir', 'touch', 'write file <filename> content: <content>' commands.""",
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
- For project scaffolding, use commands: 'mkdir', 'touch', 'write file <filename> content: <content>'
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
