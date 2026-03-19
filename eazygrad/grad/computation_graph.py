from __future__ import annotations

from typing import Any

import eazygrad as ez
import numpy as np
import heapq
import graphviz
from ..utils import check
import sys

class Node:
	"""Single node in the dynamic computation graph."""
	def __init__(self, 
				 parents_id: list[int | None], 
				 operation: Any, 
				 result: Any, 
				 is_leaf: bool = False) -> None:

		self.parents_id = parents_id
		self.operation = operation
		self.result = result
		self.is_leaf = is_leaf

class ComputationGraph:
	"""
	Global dynamic computation graph used by EaZyGrad.

	The graph stores every differentiable operation as a node together with:

	- the parent node ids,
	- the operation object that knows how to backpropagate locally,
	- the result tensor produced at that node.

	Notes
	-----
	EaZyGrad currently uses a single global graph instance (`dag`) rather than
	per-tensor graph ownership. This keeps the implementation small and easy to
	trace, but it also means graph lifetime management is more manual than in
	PyTorch.

	See Also
	--------
	`PyTorch autograd overview <https://pytorch.org/docs/stable/autograd.html>`_
	"""

	def __init__(self) -> None:
		# the computation graph
		# map between node_id and list(parent_id)
		self.dag = {}
		# node id
		self.node_count = -1
		# map between node_id and node
		self.node_map = {}
		self.grad_enable = True

	def clear(self) -> None:
		# Clear all nodes
		self.dag = {}
		self.node_count = -1
		self.node_map = {}

	def clear_node(self, node_id: int) -> None:
		# clear a specific node 
		del self.dag[node_id]
		del self.node_map[node_id]

	def create_node(self, parents_id: list[int | None], operation: Any, result: Any, is_leaf: bool = False) -> int | None:
		if not self.grad_enable:
			return None
		# Increase node counter for id
		self.node_count += 1
		# Instantiate node
		node = Node(parents_id, operation, result, is_leaf)
		# Store node in a global map 
		self.node_map[self.node_count] = node
		# Register the node in the computation graph
		self._register_node(self.node_count, parents_id)
		return self.node_count

	def _is_still_required(self, node: int, to_delete: list[int]) -> bool:
		for result, parents in self.dag.items():
			if parents is None:
				continue
			if node in parents and result not in to_delete:
				return True
		return False

	def _register_node(self, node_id: int, parents_id: list[int | None]) -> None:
		# key : resulting node id
		# values : parents nodes
		self.dag[node_id] = parents_id

	def backward(self, root_node_id: int, debug: bool = False, retain_graph: bool = False) -> None:
		"""
		Backpropagate gradients through the computation graph.

		Parameters
		----------
		root_node_id : int
			Identifier of the root node from which backpropagation starts.
		debug : bool, default=False
			Reserved debug flag. Currently unused.
		retain_graph : bool, default=False
			If ``True``, keep traversed non-leaf nodes in the graph after the
			backward pass. If ``False``, traversed non-leaf nodes are removed as
			they are processed.

		Returns
		-------
		None

		Notes
		-----
		The method consumes the accumulated gradient stored in each visited
		node's ``result.acc_grad`` field and distributes gradients to parent
		nodes using the local backward rule of each recorded operation.
		Broadcasted gradients are reduced to match parent tensor shapes before
		being accumulated.

		See Also
		--------
		`torch.Tensor.backward <https://pytorch.org/docs/stable/generated/torch.Tensor.backward.html>`_
		"""
		pending_nodes = []
		heapq.heappush(pending_nodes, -root_node_id)
		while pending_nodes:
			current_node_id = -heapq.heappop(pending_nodes)
			current_node = self.node_map[current_node_id]
			if not current_node.is_leaf:
				grads_inputs = current_node.operation.backward(current_node.result.acc_grad)
				current_node.result.acc_grad = np.float32(0.0)
				parent_nodes_id = self.dag.get(current_node_id, [])
				# Remove non-leaf node from graph if not needed anymore
				if not retain_graph:
					self.clear_node(current_node_id)
				if parent_nodes_id:
					for parent_id, grad in zip(parent_nodes_id, grads_inputs):
						if parent_id is None:
							continue
						parent = self.node_map[parent_id]
						if parent.result.requires_grad:
							grad = ez.check.broadcasted_shape(grad, parent.result)
							if parent.result.grad is None:
								parent.result.grad = grad.copy()
							else:
								parent.result.grad += grad
							parent.result.acc_grad += grad
						if -parent_id not in pending_nodes:
							heapq.heappush(pending_nodes, -parent_id)
							

	def plot(self, root_node_id: int, full_graph: bool) -> None:
		# check if graphviz is installed
		check.graphviz()
		if full_graph:
			raise NotImplementedError
		pending_nodes = []
		heapq.heappush(pending_nodes, -root_node_id)
		G = graphviz.Digraph(comment='Computation graph', format='svg')
		while pending_nodes:
			node_id = -heapq.heappop(pending_nodes)
			if node_id is not None:
				node = self.node_map[node_id]
				label = (
					f"id: {node_id}\n"
					f"Operation: {node.operation}\n"
				)
				shape = "rectangle" if node.is_leaf else "circle"
				fillcolor = "lightskyblue" if node.result.requires_grad else "mediumpurple"
				G.node(str(node_id), label=label, shape=shape, style="filled", fillcolor=fillcolor, fontsize="20")
			else:
				label = (
					f"id: {None}\n"
					f"Operation: {None}\n"
				)
				G.node("None", label=label, shape="rectangle", style="filled", fillcolor="mediumpurple", fontsize="20")
				continue
			parents_node_id = self.dag.get(node_id, [])
			if parents_node_id:
				for parent_id in parents_node_id:
					G.edge(str(parent_id), str(node_id))
					if parent_id is None:
						continue
					if -parent_id not in pending_nodes:
						heapq.heappush(pending_nodes, -parent_id)
		
		# Render the graph
		G.render(f"dag@root{root_node_id}", view=True)

# Instantiate a global computation graph
dag = ComputationGraph()
