"""
run_tool.py
Subprocess harness for sandboxed tool execution.

Called as:
    python run_tool.py <func_name> <params_json>

Outputs a single JSON line to stdout:
    [status_code, result]

Any output that the tool itself prints goes to stderr so it does not
pollute the structured JSON result that the parent process reads.
"""
import json
import sys
import os
import importlib

# Ensure the MAV root is on sys.path so 'tools.*' imports resolve correctly
_MAV_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _MAV_ROOT not in sys.path:
    sys.path.insert(0, _MAV_ROOT)

# Redirect stdout → stderr so tool debug prints don't corrupt the JSON result.
# We hold a reference to the real stdout to write the final JSON response.
_real_stdout = sys.stdout
sys.stdout = sys.stderr


def _emit(status: int, result) -> None:
    """Write the JSON result to the real stdout and flush."""
    try:
        payload = json.dumps([status, result], default=str)
    except Exception:
        payload = json.dumps([status, str(result)])
    _real_stdout.write(payload + "\n")
    _real_stdout.flush()


def main() -> None:
    if len(sys.argv) < 3:
        _emit(-1, "Usage: run_tool.py <func_name> <params_json>")
        sys.exit(1)

    func_name = sys.argv[1]

    try:
        params_dict = json.loads(sys.argv[2])
    except json.JSONDecodeError as e:
        _emit(-1, f"Invalid params JSON: {e}")
        sys.exit(1)

    try:
        module = importlib.import_module(f"tools.{func_name}")
    except ImportError:
        _emit(-1, f"Tool module 'tools.{func_name}' not found.")
        sys.exit(1)

    try:
        func = getattr(module, func_name)
    except AttributeError:
        _emit(-1, f"Function '{func_name}' not found in 'tools.{func_name}'.")
        sys.exit(1)

    try:
        status, result = func(**params_dict)
        _emit(status, result)
    except TypeError as e:
        _emit(-1, f"Incorrect parameters for '{func_name}': {e}")
        sys.exit(1)
    except Exception as e:
        _emit(-1, str(e))
        sys.exit(1)


if __name__ == "__main__":
    main()
