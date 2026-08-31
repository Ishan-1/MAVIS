"""
core/agents/__init__.py
Agent loading and registration utilities for MAVIS.
"""
import importlib
import os
import sys
from typing import Optional
from core.agents.base import BaseAgent
from core.llm.base import BaseLLMClient
from core.helpers import log_it

_ENTITY = "agent_loader"


def load_agent(agent_name: str, client: BaseLLMClient) -> Optional[BaseAgent]:
    """
    Dynamically load an agent instance by name from the agents/ directory.
    Looks for a BaseAgent subclass in agents.<agent_name>.
    """
    clean_name = agent_name.strip().lower()
    module_path = f"agents.{clean_name}"

    try:
        if module_path in sys.modules:
            module = importlib.reload(sys.modules[module_path])
        else:
            module = importlib.import_module(module_path)

        # Look for a class subclassing BaseAgent
        for attr_name in dir(module):
            attr = getattr(module, attr_name)
            if isinstance(attr, type) and issubclass(attr, BaseAgent) and attr is not BaseAgent:
                return attr(client)

        log_it(f"No BaseAgent subclass found in {module_path}", _ENTITY)
        return None
    except Exception as e:
        log_it(f"Failed to load agent '{agent_name}': {e}", _ENTITY)
        return None


__all__ = ["BaseAgent", "load_agent"]
