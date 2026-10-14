from oni import call_shell

def run_shell_command(command: str) -> tuple[int, str]:
    """
    Executes a shell command directly in the system's terminal.

    Args:
        command (str): The shell command string to execute.

    Returns:
        tuple[int, str]: A tuple containing a status code (0 for success, -1 for failure)
                        and the command's standard output (on success) or an error message (on failure).
    """
    try:
        status, result = call_shell(command)

        if status == 0:
            return 0, result
        else:
            # call_shell returns -1 on failure, with the error message in result
            return -1, f"Shell command failed: {result}"
    except Exception as e:
        return -1, f"An unexpected error occurred while executing shell command: {str(e)}"
