"""
core/dag.py

Directed Acyclic Graph (DAG) management module for MAVIS pipelines.
Provides:
  - Deep dependency extraction across scalar, list, and nested dict params.
  - Cycle detection and validation using graphlib.TopologicalSorter.
  - Dangling dependency detection (referencing non-existent steps).
  - Topological sorting to guarantee valid execution sequence.
  - Recursive parameter resolution against completed node outputs.
"""
from __future__ import annotations

import re
from graphlib import TopologicalSorter, CycleError
from typing import Any

# Pattern matching a parameter reference: $node_id or $node_id.field_name
DEP_REF_RE = re.compile(r"^\$([A-Za-z0-9_]+)(?:\.([A-Za-z0-9_]+))?$")
DEP_SEARCH_RE = re.compile(r"\$([A-Za-z0-9_]+)(?:\.([A-Za-z0-9_]+))?")


def extract_dependencies_from_value(value: Any) -> set[str]:
    """Recursively extract referenced node IDs from a parameter value."""
    deps: set[str] = set()

    if isinstance(value, str):
        # Direct reference like "$node1.output" or "$node1"
        m = DEP_REF_RE.match(value.strip())
        if m:
            deps.add(m.group(1))
        else:
            # Check for embedded references if present
            for em in DEP_SEARCH_RE.finditer(value):
                deps.add(em.group(1))
    elif isinstance(value, list):
        for item in value:
            deps.update(extract_dependencies_from_value(item))
    elif isinstance(value, dict):
        for sub_val in value.values():
            deps.update(extract_dependencies_from_value(sub_val))

    return deps


def extract_node_dependencies(node: dict) -> set[str]:
    """Extract all node ID dependencies referenced in a node's params."""
    params = node.get("params")
    if not isinstance(params, dict):
        return set()
    return extract_dependencies_from_value(params)


def validate_and_sort_dag(pipeline: list[dict]) -> tuple[list[dict] | None, str | None]:
    """
    Validate the pipeline DAG structure and return nodes in topological sorted order.

    Validates:
      1. Node schema (must have "id").
      2. Unique node IDs (no duplicates).
      3. No dangling references (all referenced $node_ids must exist in pipeline).
      4. Acyclicity (no circular dependencies).

    Returns:
      (sorted_pipeline, None) on success.
      (None, error_message) on validation or topological sort failure.
    """
    if not pipeline:
        return [], None

    nodes_by_id: dict[str, dict] = {}
    for idx, node in enumerate(pipeline):
        if not isinstance(node, dict):
            return None, f"Pipeline step at index {idx} is not an object: {node}"
        node_id = node.get("id")
        if not node_id or not isinstance(node_id, str):
            return None, f"Pipeline step at index {idx} is missing a valid 'id': {node}"
        if node_id in nodes_by_id:
            return None, f"Duplicate step ID '{node_id}' found in pipeline."
        nodes_by_id[node_id] = node

    # Extract dependencies for each node
    node_deps: dict[str, set[str]] = {}
    for node_id, node in nodes_by_id.items():
        deps = extract_node_dependencies(node)
        # Check for self-dependencies
        if node_id in deps:
            return None, f"Pipeline dependency cycle: step '{node_id}' depends on itself."
        # Check for dangling dependencies (referencing non-existent nodes)
        missing_deps = [d for d in deps if d not in nodes_by_id]
        if missing_deps:
            missing_str = ", ".join(f"'{d}'" for d in sorted(missing_deps))
            return None, f"Pipeline dependency error: step '{node_id}' references unknown step(s): {missing_str}."
        node_deps[node_id] = deps

    # Build graph and perform topological sort
    ts: TopologicalSorter = TopologicalSorter()
    for node_id, deps in node_deps.items():
        ts.add(node_id, *deps)

    try:
        execution_order = list(ts.static_order())
    except CycleError as ce:
        cycle_cycle = ce.args[1] if len(ce.args) > 1 else ()
        cycle_chain = " -> ".join(str(n) for n in cycle_cycle) if cycle_cycle else str(ce)
        return None, f"Pipeline dependency cycle detected: {cycle_chain}."
    except Exception as e:
        return None, f"Failed to compute topological order for pipeline: {e}."

    sorted_pipeline = [nodes_by_id[nid] for nid in execution_order if nid in nodes_by_id]
    return sorted_pipeline, None


def resolve_params(params: Any, node_results: dict[str, Any]) -> tuple[Any, str | None]:
    """
    Recursively resolve parameter references ($node_id, $node_id.output, $node_id.field)
    against completed node_results.

    Returns:
      (resolved_params, None) on success.
      (None, error_message) on missing dependency or resolution error.
    """
    if isinstance(params, str):
        trimmed = params.strip()
        m = DEP_REF_RE.match(trimmed)
        if m:
            dep_id = m.group(1)
            field = m.group(2)

            if dep_id not in node_results:
                return None, f"Dependency '{dep_id}' has not completed or produced a result."

            payload = node_results[dep_id]

            if not field or field == "output":
                return payload, None
            if field == "status":
                return 0, None
            if isinstance(payload, dict) and field in payload:
                return payload[field], None
            return payload, None

        # Return original string if not a direct $node_id reference
        return params, None

    if isinstance(params, list):
        resolved_list = []
        for item in params:
            resolved_item, err = resolve_params(item, node_results)
            if err:
                return None, err
            resolved_list.append(resolved_item)
        return resolved_list, None

    if isinstance(params, dict):
        resolved_dict = {}
        for key, val in params.items():
            resolved_val, err = resolve_params(val, node_results)
            if err:
                return None, err
            resolved_dict[key] = resolved_val
        return resolved_dict, None

    return params, None
