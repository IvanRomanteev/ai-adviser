"""
A minimal implementation of a node-based state graph executor.

This stub is provided to support the orchestrated RAG pipeline defined in
`ai_adviser.agent.graph` without requiring the external `langgraph` library.
It exposes a `StateGraph` class that allows registering named nodes,
defining directed edges between them, specifying an entry point, and
compiling the graph into an executable function. The compiled function
progresses through the graph by following the first outgoing edge from each
node until it reaches the `END` sentinel or there are no more edges.

The `END` constant is used to indicate termination of the graph. Nodes
registered on the graph should accept a mutable state dictionary and return
a dictionary of updates. These updates are merged into the state before
proceeding to the next node.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

# Sentinel value marking the end of a graph execution.  When the current
# node is equal to END the compiled executor will terminate.
END: str = "__end__"


class StateGraph:
    """A lightweight directed graph for orchestrating stateful computations.

    Each node in the graph is a callable that accepts a mutable state
    dictionary and returns a dictionary of updates. The compiled graph
    executes nodes in sequence by following the first outgoing edge from
    each node until no more edges are available or the END sentinel is
    reached. This simple design is sufficient for linear flows typical in
    Retrieval‑Augmented Generation (RAG) pipelines.
    """

    def __init__(self, state_type: Any) -> None:
        # Mapping of node name to callable
        self._nodes: Dict[str, Callable[[Dict[str, Any]], Dict[str, Any]]] = {}
        # Adjacency list of edges
        self._edges: Dict[str, List[str]] = {}
        # Name of the entry point node
        self._entry_point: Optional[str] = None
        # Optional state type annotation; unused in this stub but kept for API
        self._state_type = state_type

    def add_node(self, name: str, func: Callable[[Dict[str, Any]], Dict[str, Any]]) -> None:
        """Register a new node on the graph.

        Args:
            name: Unique name for the node.
            func: A callable that accepts and returns state updates.
        """
        if name in self._nodes:
            raise ValueError(f"Node '{name}' is already defined on the graph")
        self._nodes[name] = func

    def set_entry_point(self, name: str) -> None:
        """Define the entry point node for the graph.

        Args:
            name: Name of the node to start execution from.
        """
        if name not in self._nodes:
            raise ValueError(f"Entry point '{name}' has not been added as a node")
        self._entry_point = name

    def add_edge(self, from_node: str, to_node: str) -> None:
        """Add a directed edge from one node to another.

        Args:
            from_node: Name of the source node.
            to_node: Name of the destination node. Use END to terminate.
        """
        if from_node not in self._nodes:
            raise ValueError(f"Edge source '{from_node}' is not a registered node")
        if to_node != END and to_node not in self._nodes:
            raise ValueError(f"Edge destination '{to_node}' is not a registered node")
        self._edges.setdefault(from_node, []).append(to_node)

    def compile(self) -> Callable[[Dict[str, Any]], Dict[str, Any]]:
        """Compile the graph into an executable function.

        Returns:
            A function that accepts an initial state dictionary and returns the
            final state after executing each node in sequence.
        """

        def executor(state: Dict[str, Any]) -> Dict[str, Any]:
            # Defensive copy to avoid mutating the caller's state
            current_state: Dict[str, Any] = dict(state or {})
            current_node = self._entry_point
            # Iterate until no node or end sentinel
            while current_node and current_node != END:
                node_func = self._nodes[current_node]
                try:
                    updates = node_func(current_state) or {}
                except Exception:
                    # propagate exceptions to caller
                    raise
                # merge updates into state
                if updates:
                    current_state.update(updates)
                # determine next node; take first outgoing edge if multiple
                next_nodes = self._edges.get(current_node, [])
                current_node = next_nodes[0] if next_nodes else None
            return current_state

        if self._entry_point is None:
            raise RuntimeError("Cannot compile graph without an entry point")
        return executor
