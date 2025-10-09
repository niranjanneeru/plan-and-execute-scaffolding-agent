import os
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
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


## TODO: Code Generation Tool


tools = [
    create_directory,
    create_file,
    write_to_file,
    move_file,
    delete_file,
    list_directory,
]

llm = ChatOpenAI(model="gpt-4o", temperature=0, api_key=os.getenv("OPENAI_API_KEY"))


# --- Quick Tool Execution ---
def test_tools():
    """Simple test of all tools"""
    print("🛠️  Testing tools...")
    
    # Test basic operations
    print(create_directory.invoke({"path": "test_dir"}))
    print(create_file.invoke({"path": "test_dir/hello.txt", "content": "Hello World!"}))
    print(list_directory.invoke({"path": "test_dir"}))
    print(write_to_file.invoke({"path": "test_dir/hello.txt", "content": "\nGoodbye!"}))
    print(list_directory.invoke({}))


if __name__ == "__main__":
    test_tools()
