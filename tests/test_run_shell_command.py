def test_run_shell_command():
    from tools.run_shell_command import run_shell_command

    # Test case 1: A simple, successful command
    command = "echo 'Hello from shell test'"
    print(f"Running test for run_shell_command with command: '{command}'")

    # Call the function
    status, output = run_shell_command(command)
    result = (status, output) # Store the full return value for type checking

    # Assert 1: Status code is 0 for success
    if status != 0:
        raise AssertionError(
            f"Test failed for command '{command}': "
            f"Expected status code 0, but got {status}. "
            f"Output: {output.strip()}"
        )

    # Assert 2: Return value is a tuple
    if not isinstance(result, tuple):
        raise AssertionError(
            f"Test failed for command '{command}': "
            f"Expected return value to be a tuple, but got type {type(result).__name__}."
        )

    # Assert 3: First element of tuple (status) is an int
    if not isinstance(status, int):
        raise AssertionError(
            f"Test failed for command '{command}': "
            f"Expected first element (status) to be an int, but got type {type(status).__name__}."
        )

    # Assert 4: Second element of tuple (output) is a str
    if not isinstance(output, str):
        raise AssertionError(
            f"Test failed for command '{command}': "
            f"Expected second element (output) to be a str, but got type {type(output).__name__}."
        )

    # Optional but good: Check content of output for 'echo' command
    expected_output_part = "Hello from shell test"
    if expected_output_part not in output:
        raise AssertionError(
            f"Test failed for command '{command}': "
            f"Expected output to contain '{expected_output_part}', but got '{output.strip()}'."
        )

    print("All tests passed for run_shell_command!")