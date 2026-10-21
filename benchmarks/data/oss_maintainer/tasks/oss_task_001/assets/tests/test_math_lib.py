import pytest
from math_lib import safe_divide

def test_safe_divide_normal():
    assert safe_divide(10, 2) == 5.0

def test_safe_divide_by_zero():
    assert safe_divide(10, 0) == 0.0
