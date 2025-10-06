# Available Tools Documentation

## 📁 File & Directory Operations

### 1. `create_directory(path: str)`
Creates a directory in the `generated_projects/` folder.

**Parameters:**
- `path`: Directory path (automatically prefixed with `generated_projects/`)

**Example:**
```python
create_directory("my_project/src")
# Creates: generated_projects/my_project/src/
```

**Returns:** Success or error message

---

### 2. `create_file(path: str, content: str = "")`
Creates a file with specified content in the `generated_projects/` folder.

**Parameters:**
- `path`: File path (automatically prefixed with `generated_projects/`)
- `content`: File content (optional, defaults to empty string)

**Example:**
```python
create_file("my_project/main.py", "print('Hello, World!')")
# Creates: generated_projects/my_project/main.py with the content
```

**Returns:** Success message with character count or error

---

### 3. `write_to_file(path: str, content: str)`
Appends content to an existing file.

**Parameters:**
- `path`: File path
- `content`: Content to append

**Example:**
```python
write_to_file("my_project/main.py", "\n\nif __name__ == '__main__':\n    main()")
# Appends content to existing file
```

**Returns:** Success message with character count or error

---

### 4. `move_file(source: str, destination: str)`
Moves or renames a file or directory.

**Parameters:**
- `source`: Source path
- `destination`: Destination path

**Example:**
```python
move_file("my_project/old_name.py", "my_project/new_name.py")
# Renames the file

move_file("my_project/temp", "my_project/backup")
# Moves directory
```

**Returns:** Success or error message

---

### 5. `delete_file(path: str)`
Deletes a file or directory (including all contents if directory).

**Parameters:**
- `path`: Path to file or directory to delete

**Example:**
```python
delete_file("my_project/temp.txt")
# Deletes the file

delete_file("my_project/old_folder")
# Deletes directory and all contents
```

**Returns:** Success or error message

**⚠️ Warning:** This permanently deletes files/directories!

---

### 6. `list_directory(path: str = "generated_projects")`
Lists contents of a directory with file sizes.

**Parameters:**
- `path`: Directory path (defaults to root `generated_projects/`)

**Example:**
```python
list_directory("my_project")
# Lists all files and folders in generated_projects/my_project/
```

**Returns:** Formatted list of files (📄) and directories (📁) with sizes

---

## 🤖 AI-Powered Tools

### 7. `generate_code(description: str, language: str)`
Generates code based on a description using GPT-4.

**Parameters:**
- `description`: What the code should do
- `language`: Programming language (e.g., "Python", "JavaScript", "Java")

**Example:**
```python
generate_code("a function to calculate fibonacci numbers", "Python")
# Returns: Python code for fibonacci function
```

**Returns:** Generated code as a string

---

## 🎯 Usage Examples

### Example 1: Create a Python Project
```python
# Step 1: Create directory structure
create_directory("my_app")
create_directory("my_app/src")
create_directory("my_app/tests")

# Step 2: Create files
create_file("my_app/README.md", "# My App\n\nA Python application")
create_file("my_app/requirements.txt", "flask\nrequests")

# Step 3: Generate code
code = generate_code("a Flask API with a hello endpoint", "Python")
create_file("my_app/src/app.py", code)

# Step 4: Create test file
create_file("my_app/tests/__init__.py", "")
create_file("my_app/tests/test_app.py", "import pytest\n\n# Add tests here")

# Step 5: List to verify
list_directory("my_app")
```

### Example 2: Reorganize Project
```python
# Move files
move_file("my_app/old_module.py", "my_app/src/new_module.py")

# Delete temporary files
delete_file("my_app/temp.txt")
delete_file("my_app/cache")

# Verify changes
list_directory("my_app")
```

### Example 3: Build a React App
```python
# Create structure
create_directory("react_app/src/components")
create_directory("react_app/public")

# Generate component
component_code = generate_code("a React button component with click handler", "JavaScript")
create_file("react_app/src/components/Button.jsx", component_code)

# Create config files
create_file("react_app/package.json", '{\n  "name": "react-app",\n  "version": "1.0.0"\n}')
create_file("react_app/README.md", "# React App")
```

---

## 🔒 Safety Features

1. **Workspace Isolation**: All operations are confined to `generated_projects/` directory
2. **Auto-prefixing**: Paths are automatically prefixed to prevent accidental system file modification
3. **Error Handling**: All tools return clear error messages if operations fail
4. **Directory Creation**: Parent directories are created automatically when needed

---

## 💡 Best Practices

1. **Always use relative paths**: Don't include `generated_projects/` in your paths
   ```python
   # ✅ Good
   create_file("my_app/main.py", "...")
   
   # ❌ Unnecessary (but still works)
   create_file("generated_projects/my_app/main.py", "...")
   ```

2. **Create directories before files**: While `create_file` creates parent directories, it's clearer to explicitly create them
   ```python
   # ✅ Good
   create_directory("my_app/src")
   create_file("my_app/src/main.py", "...")
   ```

3. **Use generate_code for complex logic**: Let AI handle boilerplate and complex code
   ```python
   code = generate_code("a REST API with CRUD operations for users", "Python")
   create_file("api/routes.py", code)
   ```

4. **List directories to verify**: Use `list_directory` to confirm operations
   ```python
   create_directory("my_app")
   create_file("my_app/test.py", "")
   list_directory("my_app")  # Verify creation
   ```

---

## 🚀 Integration with Plan-Execute Agent

The agent automatically uses these tools based on your natural language request:

**Input:**
```
"Create a Python FastAPI project with user authentication"
```

**Agent will:**
1. Plan the project structure
2. Use `create_directory` for folders
3. Use `generate_code` for API code
4. Use `create_file` to save generated code
5. Use `list_directory` to verify
6. Return summary of created project

---

## 📊 Tool Selection Guide

| Task | Recommended Tool |
|------|------------------|
| Create folder | `create_directory` |
| Create empty file | `create_file(path, "")` |
| Create file with content | `create_file(path, content)` |
| Add to existing file | `write_to_file` |
| Rename file | `move_file` |
| Move file to folder | `move_file` |
| Remove file/folder | `delete_file` |
| Check what's in folder | `list_directory` |
| Generate complex code | `generate_code` |

---

**Last Updated:** 2025-10-06
