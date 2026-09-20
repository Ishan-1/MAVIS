def safe_divide(a: float, b: float) -> float:
    # BUG: fails on zero division instead of returning 0.0 or handling gracefully
    return a / b
