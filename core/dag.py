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

import json
from pathlib import Path
import re
from graphlib import TopologicalSorter, CycleError
from typing import Any

# Pattern matching a parameter reference: $node_id or $node_id.field_name
DEP_REF_RE = re.compile(r"^\$([A-Za-z0-9_]+)(?:\.([A-Za-z0-9_]+))?$")
DEP_SEARCH_RE = re.compile(r"\$([A-Za-z0-9_]+)(?:\.([A-Za-z0-9_]+))?")
SCRATCH_OFFLOAD_RE = re.compile(
    r"\n?\.\.\. \[(?:\d+ lines / )?\d+ bytes offloaded to (?:file: )?(data/scratch/[^\s\]]+)\] \.\.\.\n?"
)


def dereference_scratchpad_value(val: Any) -> Any:
    """If val is a string containing a scratchpad offload digest, restore full content from disk."""
    if not isinstance(val, str):
        return val
    m = SCRATCH_OFFLOAD_RE.search(val)
    if m:
        scratch_path = m.group(1)
        p = Path(scratch_path)
        if p.exists() and p.is_file():
            try:
                full_content = p.read_text(encoding="utf-8")
                # If the value is a head/notice/tail digest, the full content replaces the entire digest
                return full_content
            except Exception:
                pass
    return val



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
    """Extract all node ID dependencies referenced in a node's params, conditions, or explicit deps."""
    deps: set[str] = set()

    # 1. Explicit dependencies: "dependencies": ["n1"] or "deps": ["n1"]
    explicit_deps = node.get("dependencies") or node.get("deps")
    if isinstance(explicit_deps, list):
        deps.update(str(d) for d in explicit_deps if d)
    elif isinstance(explicit_deps, str) and explicit_deps:
        deps.add(explicit_deps)

    # 2. Gate and Control condition dependencies: "condition": "$n1.status == 0"
    if node.get("type") in ("gate", "control"):
        cond = node.get("condition") or node.get("condition_str", "") or node.get("expected_outcome", "")
        if isinstance(cond, str):
            for m in DEP_SEARCH_RE.finditer(cond):
                deps.add(m.group(1))

    # 3. Parameter references
    params = node.get("params")
    if isinstance(params, (dict, list, str)):
        deps.update(extract_dependencies_from_value(params))

    return deps


def get_downstream_nodes(
    start_nodes: set[str] | list[str],
    nodes_by_id: dict[str, dict],
    node_deps: dict[str, set[str]],
) -> set[str]:
    """Find all nodes that depend directly or transitively on start_nodes."""
    visited = set(start_nodes)
    queue = list(start_nodes)
    while queue:
        curr = queue.pop(0)
        for nid, deps in node_deps.items():
            if curr in deps and nid not in visited:
                visited.add(nid)
                queue.append(nid)
    return visited


def compute_pruned_nodes(
    gate_node: dict,
    verdict: bool,
    nodes_by_id: dict[str, dict],
    node_deps: dict[str, set[str]],
) -> set[str]:
    """
    Given an evaluated gate node and its verdict, return set of node IDs
    that should be skipped because their branch was not taken.
    """
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

    active_entry_nodes = set(if_true if verdict else if_false)
    inactive_entry_nodes = set(if_false if verdict else if_true)

    pruned: set[str] = set()

    # Prune inactive branch targets and their exclusive downstream nodes
    if inactive_entry_nodes:
        inactive_subtree = get_downstream_nodes(inactive_entry_nodes, nodes_by_id, node_deps)
        active_subtree = get_downstream_nodes(active_entry_nodes, nodes_by_id, node_deps) if active_entry_nodes else set()
        pruned.update(inactive_subtree - active_subtree)

    # Also prune nodes explicitly marked with the opposite branch tag
    gate_id = gate_node.get("id")
    for nid, node in nodes_by_id.items():
        node_branch = str(node.get("branch", "")).lower()
        if not node_branch:
            continue
        # If node belongs to this gate
        if gate_id in node_deps.get(nid, set()):
            if verdict and node_branch in ("false", "if_false", "no"):
                pruned.add(nid)
            elif (not verdict) and node_branch in ("true", "if_true", "yes"):
                pruned.add(nid)

    return pruned


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
        node_deps[node_id] = deps

    # Gate nodes implicitly precede their branch target nodes
    for node_id, node in nodes_by_id.items():
        if node.get("type") == "gate":
            targets = []
            for k in ("if_true", "if_false"):
                v = node.get(k)
                if isinstance(v, str):
                    targets.append(v)
                elif isinstance(v, list):
                    targets.extend(v)
            for t in targets:
                if t in nodes_by_id:
                    node_deps[t].add(node_id)

    # Control nodes implicitly succeed all non-control nodes if no explicit deps were defined
    for node_id, node in nodes_by_id.items():
        if node.get("type") == "control" and not node_deps[node_id]:
            all_other = {nid for nid, n in nodes_by_id.items() if nid != node_id and n.get("type") != "control"}
            node_deps[node_id].update(all_other)

    # Verify self-deps and dangling references
    for node_id, deps in node_deps.items():
        if node_id in deps:
            return None, f"Pipeline dependency cycle: step '{node_id}' depends on itself."
        missing_deps = [d for d in deps if d not in nodes_by_id]
        if missing_deps:
            missing_str = ", ".join(f"'{d}'" for d in sorted(missing_deps))
            return None, f"Pipeline dependency error: step '{node_id}' references unknown step(s): {missing_str}."

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
    against completed node_results. Supports both direct references (e.g. "$n1.output")
    and embedded references inside strings (e.g. 'git commit -m "$n2.output"').

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

            if isinstance(payload, dict) and field and field in payload:
                return dereference_scratchpad_value(payload[field]), None
            if field == "status":
                return 0, None
            if not field or field == "output":
                return dereference_scratchpad_value(payload), None
            return dereference_scratchpad_value(payload), None

        if "$" in params:
            err_holder: list[str] = []

            def _replace_dep(match: re.Match) -> str:
                dep_id = match.group(1)
                field = match.group(2)

                if dep_id not in node_results:
                    err_holder.append(f"Dependency '{dep_id}' has not completed or produced a result.")
                    return match.group(0)

                payload = node_results[dep_id]

                if isinstance(payload, dict) and field and field in payload:
                    val = payload[field]
                elif field == "status":
                    val = 0
                elif not field or field == "output":
                    val = payload
                else:
                    val = payload

                val = dereference_scratchpad_value(val)
                if isinstance(val, (dict, list)):
                    return json.dumps(val)
                return str(val)

            resolved_str = DEP_SEARCH_RE.sub(_replace_dep, params)
            if err_holder:
                return None, err_holder[0]
            return resolved_str, None

        # Return original string if not a reference
        return dereference_scratchpad_value(params), None


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
