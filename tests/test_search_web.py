def test_search_web():
    from tools.search_web import search_web

    print("Running test_search_web...")

    # Test case 1: A general, common query
    try:
        query1 = "latest Python version"
        status_code1, result1 = search_web(query1)

        assert status_code1 == 0, f"Test case 1 failed: Expected status code 0, got {status_code1} for query '{query1}'"
        assert isinstance(result1, str), f"Test case 1 failed: Expected result to be of type str, got {type(result1)} for query '{query1}'"
        print(f"Test case 1 passed: Query '{query1}' returned status code 0 and string result.")
    except Exception as e:
        raise AssertionError(f"Test case 1 failed due to unexpected exception: {e}")

    # Test case 2: A different, slightly more specific query
    try:
        query2 = "weather in London tomorrow"
        status_code2, result2 = search_web(query2)

        assert status_code2 == 0, f"Test case 2 failed: Expected status code 0, got {status_code2} for query '{query2}'"
        assert isinstance(result2, str), f"Test case 2 failed: Expected result to be of type str, got {type(result2)} for query '{query2}'"
        print(f"Test case 2 passed: Query '{query2}' returned status code 0 and string result.")
    except Exception as e:
        raise AssertionError(f"Test case 2 failed due to unexpected exception: {e}")

    # Test case 3: An empty query (might return a failure or an empty string, but we expect success for now)
    # The function signature implies success (status 0) is the common path, even if result is empty/minimal.
    try:
        query3 = ""
        status_code3, result3 = search_web(query3)

        assert status_code3 == 0, f"Test case 3 failed: Expected status code 0, got {status_code3} for empty query."
        assert isinstance(result3, str), f"Test case 3 failed: Expected result to be of type str, got {type(result3)} for empty query."
        print(f"Test case 3 passed: Empty query returned status code 0 and string result.")
    except Exception as e:
        raise AssertionError(f"Test case 3 failed due to unexpected exception: {e}")

    print("All assertions in test_search_web passed!")
