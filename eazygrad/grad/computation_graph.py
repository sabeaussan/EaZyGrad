import pytensor as pyt
import graphviz
import queue
import heapq

class Node:

	def __init__(self, 
				 parents_id, 
				 operation, 
				 result, 
				 is_leaf = False):

		self.parents_id = parents_id
		self.operation = operation
		self.result = result
		self.is_leaf = is_leaf

class ComputationGraph:

	def __init__(self):
		self.dag = {}
		self.node_count = -1
		self.node_map = {}

	def clear(self):
		self.dag = {}
		self.node_count = -1
		self.node_map = {}

	def create_node(self, parents_id, operation, result, is_leaf= False):
		# Increase node counter for id
		self.node_count += 1
		# Instantiate node
		node = Node(parents_id, operation, result, is_leaf)
		# Store node in a global map 
		self.node_map[self.node_count] = node
		# Register the node in the computation graph
		self._register_node(self.node_count, parents_id)
		return self.node_count

	def _is_still_required(self, node, to_delete):
		for result, parents in self.dag.items():
			if parents is None:
				continue
			if node in parents and result not in to_delete:
				return True
		return False

	def _register_node(self, node_id, parents_id):
		# key : resulting node id
		# values : parents nodes
		self.dag[node_id] = parents_id

	def backward(self, root_node_id, debug = False, retain_graph = False):
		pending_nodes = []
		heapq.heappush(pending_nodes, -root_node_id)
		while pending_nodes:
			current_node_id = -heapq.heappop(pending_nodes)
			current_node = self.node_map[current_node_id]
			if not current_node.is_leaf:
				grad_output = current_node.operation.backward(current_node.result.grad)
				parent_nodes_id = self.dag.get(current_node_id, [])
				if parent_nodes_id:
					for parent_id, grad in zip(parent_nodes_id, grad_output):
						parent = self.node_map[parent_id]
						if parent.result.requires_grad:
							grad = pyt.check_broadcasted_shape(grad, parent.result)
							if parent.result.grad is None:
								parent.result.grad = grad
							else:
								parent.result.grad += grad
						if -parent_id not in pending_nodes:
							heapq.heappush(pending_nodes, -parent_id)
							


	def plot(self, dump=False):
		# TODO :  add json dump
		# TODO : if node does not require grad, node is not in the graph ?
		G = graphviz.Digraph(comment='Computation graph', format='svg')
		nodes = set()
		for node_id in range(len(self.node_map)):
			if node_id in nodes:
				continue
			nodes.add(node_id)
			node = self.node_map[node_id]
			label = (
				f"id: {node_id}\n"
				f"Operation: {node.operation}\n"
			)
			shape = "rectangle" if node.is_leaf else "circle"
			fillcolor = "lightskyblue" if node.result.requires_grad else "mediumpurple"
			G.node(str(node_id), label=label, shape=shape, style="filled", fillcolor=fillcolor, fontsize="20")
			print(node_id, node.operation)
			for parent_id in self.dag[node_id]:
				if parent_id is not None:
					G.edge(str(parent_id), str(node_id))
		# Render the tree and display it.
		G.render("foo", view=True)



# Instantiate a global computation graph
dag = ComputationGraph()
