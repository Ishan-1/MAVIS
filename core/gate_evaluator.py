"""
core/gate_evaluator.py

Evaluates Gate Nodes in MAVIS DAG execution pipelines.
Supports:
  1. Deterministic evaluation (0 LLM tokens, sub-millisecond AST parser)
     evaluating expressions like:
       $node.status == 0
       len($node.output) > 0
       "FAILED" in $node.output
       "error" not in $node.output.lower()
  2. NLP evaluation via LLM-as-a-judge for semantic/fuzzy conditions.
"""
from __future__ import annotations

import ast
import json
import re
from typing import Any

from core.helpers import log_it
from core.llm import BaseLLMClient, get_llm_client
from core.metrics import MetricEmitter

_ENTITY = "gate_evaluator"
_EMITTER = MetricEmitter("gate_evaluator")

SAFE_BUILTINS = {
    "len": len,
    "str": str,
    "int": int,
    "float": float,
    "bool": bool,
    "list": list,
    "dict": dict,
    "True": True,
    "False": False,
    "None": None,
}

SAFE_METHODS = {
    "lower",
    "upper",
    "strip",
    "startswith",
    "endswith",
    "split",
    "get",
    "keys",
    "values",
    "items",
    "count",
    "find",
}


class NodeResultProxy:
    """Safe wrapper around node output for attribute and subscript lookups."""

    def __init__(self, data: Any):
        self._raw = data
        if isinstance(data, dict):
            self._dict = data
            self.status = data.get("status", 0)
            self.output = data.get("output", data)
            self.result = data.get("result", self.output)
        elif isinstance(data, (tuple, list)) and len(data) == 2 and isinstance(data[0], int):
            self.status = data[0]
            self.output = data[1]
            self.result = data[1]
            self._dict = {"status": self.status, "output": self.output, "result": self.result}
        elif isinstance(data, str) and (data.startswith("FAILED:") or "FAILED" in data[:20]):
            self.status = 1
            self.output = data
            self.result = data
            self._dict = {"status": 1, "output": data, "result": data}
        else:
            self.status = 0
            self.output = data
            self.result = data
            self._dict = {"status": 0, "output": data, "result": data}

    def __getattr__(self, name: str) -> Any:
        if name in self.__dict__:
            return self.__dict__[name]
        if name in self._dict:
            return self._dict[name]
        if hasattr(self.output, name):
            attr = getattr(self.output, name)
            if callable(attr) and name in SAFE_METHODS:
                return attr
        raise AttributeError(f"NodeResult has no attribute '{name}'")

    def __getitem__(self, key: Any) -> Any:
        if isinstance(self._dict, dict) and key in self._dict:
            return self._dict[key]
        if isinstance(self.output, (dict, list, tuple, str)):
            return self.output[key]
        raise KeyError(key)

    def __contains__(self, item: Any) -> bool:
        if isinstance(self.output, (dict, list, tuple, str, set)):
            return item in self.output
        return str(item) in str(self.output)

    def __len__(self) -> int:
        if hasattr(self.output, "__len__"):
            return len(self.output)
        return len(str(self.output))

    def __str__(self) -> str:
        return str(self.output)

    def __repr__(self) -> str:
        return f"NodeResultProxy(status={self.status}, output={self.output!r})"

    def get(self, key: Any, default: Any = None) -> Any:
        if isinstance(self._dict, dict) and key in self._dict:
            return self._dict[key]
        if isinstance(self.output, dict):
            return self.output.get(key, default)
        return default


def _safe_eval_node(node: ast.AST, env: dict[str, Any]) -> Any:
    """Recursively evaluate an AST expression using only safe operations."""
    if isinstance(node, ast.Expression):
        return _safe_eval_node(node.body, env)

    if isinstance(node, ast.Constant):
        return node.value

    if isinstance(node, ast.Name):
        if node.id in env:
            return env[node.id]
        if node.id in SAFE_BUILTINS:
            return SAFE_BUILTINS[node.id]
        raise ValueError(f"Disallowed or undefined variable: '{node.id}'")

    if isinstance(node, ast.UnaryOp):
        operand = _safe_eval_node(node.operand, env)
        if isinstance(node.op, ast.Not):
            return not operand
        if isinstance(node.op, ast.USub):
            return -operand
        if isinstance(node.op, ast.UAdd):
            return +operand
        raise ValueError(f"Disallowed unary operator: {type(node.op).__name__}")

    if isinstance(node, ast.BinOp):
        left = _safe_eval_node(node.left, env)
        right = _safe_eval_node(node.right, env)
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, ast.Div):
            return left / right
        if isinstance(node.op, ast.Mod):
            return left % right
        raise ValueError(f"Disallowed binary operator: {type(node.op).__name__}")

    if isinstance(node, ast.BoolOp):
        if isinstance(node.op, ast.And):
            for val in node.values:
                if not _safe_eval_node(val, env):
                    return False
            return True
        if isinstance(node.op, ast.Or):
            for val in node.values:
                if _safe_eval_node(val, env):
                    return True
            return False
        raise ValueError(f"Disallowed boolean operator: {type(node.op).__name__}")

    if isinstance(node, ast.Compare):
        left = _safe_eval_node(node.left, env)
        for op, comparator in zip(node.ops, node.comparators):
            right = _safe_eval_node(comparator, env)
            if isinstance(op, ast.Eq):
                res = (left == right)
            elif isinstance(op, ast.NotEq):
                res = (left != right)
            elif isinstance(op, ast.Lt):
                res = (left < right)
            elif isinstance(op, ast.LtE):
                res = (left <= right)
            elif isinstance(op, ast.Gt):
                res = (left > right)
            elif isinstance(op, ast.GtE):
                res = (left >= right)
            elif isinstance(op, ast.In):
                res = (left in right)
            elif isinstance(op, ast.NotIn):
                res = (left not in right)
            elif isinstance(op, ast.Is):
                res = (left is right)
            elif isinstance(op, ast.IsNot):
                res = (left is not right)
            else:
                raise ValueError(f"Disallowed comparison operator: {type(op).__name__}")
            if not res:
                return False
            left = right
        return True

    if isinstance(node, ast.Attribute):
        value = _safe_eval_node(node.value, env)
        attr = node.attr
        if isinstance(value, NodeResultProxy):
            return getattr(value, attr)
        if isinstance(value, dict) and attr in value:
            return value[attr]
        if hasattr(value, attr):
            target_attr = getattr(value, attr)
            if callable(target_attr) and attr in SAFE_METHODS:
                return target_attr
            elif not callable(target_attr):
                return target_attr
        raise AttributeError(f"Attribute '{attr}' not accessible on object of type {type(value).__name__}")

    if isinstance(node, ast.Subscript):
        value = _safe_eval_node(node.value, env)
        slice_val = _safe_eval_node(node.slice, env)
        return value[slice_val]

    if isinstance(node, ast.Call):
        func = _safe_eval_node(node.func, env)
        args = [_safe_eval_node(a, env) for a in node.args]
        kwargs = {kw.arg: _safe_eval_node(kw.value, env) for kw in node.keywords if kw.arg}

        if func in SAFE_BUILTINS.values():
            return func(*args, **kwargs)

        # Allow bound safe methods
        func_name = getattr(func, "__name__", None)
        if func_name in SAFE_METHODS:
            return func(*args, **kwargs)

        raise ValueError(f"Disallowed function call: {func}")

    raise ValueError(f"Disallowed AST expression type: {type(node).__name__}")


def evaluate_deterministic(condition_str: str, node_results: dict[str, Any]) -> bool:
    """
    Safely evaluate a deterministic condition string against node results.
    Replaces $node_id references with internal variables bound to NodeResultProxy objects.
    """
    if not condition_str or not condition_str.strip():
        return True

    clean_cond = condition_str.strip()

    # Find all $var references
    var_map: dict[str, str] = {}
    env: dict[str, Any] = {}

    def _replace_dep(match: re.Match) -> str:
        dep_id = match.group(1)
        safe_var = f"_node_{dep_id}"
        var_map[safe_var] = dep_id
        return safe_var

    # Replace $var with safe variable name
    sanitized_expr = re.sub(r"\$([A-Za-z0-9_]+)", _replace_dep, clean_cond)

    for safe_var, dep_id in var_map.items():
        raw_val = node_results.get(dep_id)
        env[safe_var] = NodeResultProxy(raw_val)

    try:
        parsed = ast.parse(sanitized_expr, mode="eval")
        result = _safe_eval_node(parsed, env)
        return bool(result)
    except Exception as e:
        log_it(f"Deterministic condition evaluation failed: '{condition_str}' -> {e}", _ENTITY)
        return False


def evaluate_nlp(
    condition_str: str,
    context_payload: Any,
    client: BaseLLMClient | None = None,
) -> bool:
    """
    Evaluate a fuzzy semantic condition via fast LLM-as-a-judge.
    Expects binary True/False judgment.
    """
    if not condition_str or not condition_str.strip():
        return True

    if client is None:
        client = get_llm_client()

    prompt = (
        "You are an impartial binary evaluation judge for an automated pipeline.\n"
        "Your task is to determine whether the CONDITION is satisfied based strictly on the provided CONTEXT.\n\n"
        f"CONDITION: {condition_str.strip()}\n\n"
        f"CONTEXT:\n{json.dumps(context_payload, indent=2, default=str)[:3000]}\n\n"
        "Respond ONLY with a JSON object in this exact format:\n"
        '{"verdict": true, "reason": "brief explanation"}\n'
        'or\n'
        '{"verdict": false, "reason": "brief explanation"}'
    )

    try:
        response_text = client.chat(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=100,
        )
        # Parse JSON
        m = re.search(r"\{.*\}", response_text, re.DOTALL)
        if m:
            data = json.loads(m.group(0))
            verdict = bool(data.get("verdict", False))
            log_it(f"NLP Judge verdict for '{condition_str}': {verdict} ({data.get('reason', '')})", _ENTITY)
            return verdict
    except Exception as e:
        log_it(f"NLP condition evaluation error: {e}", _ENTITY)

    return False


def evaluate_gate(
    gate_node: dict,
    node_results: dict[str, Any],
    client: BaseLLMClient | None = None,
) -> tuple[bool, list[str], list[str]]:
    """
    Execute a gate node and determine branching.

    Returns:
      (verdict, active_branch_node_ids, inactive_branch_node_ids)
    """
    condition_str = gate_node.get("condition") or gate_node.get("condition_str", "")
    mode = str(gate_node.get("mode", "deterministic")).lower()

    if_true = gate_node.get("if_true", [])
    if isinstance(if_true, str):
        if_true = [if_true]
    elif not isinstance(if_true, list):
        if_true = []

    if_false = gate_node.get("if_false", [])
    if isinstance(if_false, str):
        if_false = [if_false]
    elif not isinstance(if_false, list):
        if_false = []

    if mode == "nlp":
        # Extract referenced nodes from condition for context
        dep_matches = re.findall(r"\$([A-Za-z0-9_]+)", condition_str)
        if dep_matches:
            context = {d: node_results.get(d) for d in dep_matches if d in node_results}
        else:
            context = node_results

        verdict = evaluate_nlp(condition_str, context, client=client)
    else:
        verdict = evaluate_deterministic(condition_str, node_results)

    active_nodes = if_true if verdict else if_false
    inactive_nodes = if_false if verdict else if_true

    return verdict, active_nodes, inactive_nodes
