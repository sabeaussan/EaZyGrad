import eazygrad
import abc

class Module:

	def __init__(self):
		self._params = []

	def register_params(self, params):
		self._params.append(params)

	def forward(self, *args):
		"""
		Apply the forward pass of the module to the input x
		"""
		raise NotImplementedError(f"Forward pass not implemented for {self.__class__.__name__}")

	def __call__(self, *args):
		return self.forward(*args)

	def __repr__(self):
		return f"({self.__class__.__name__})"

	def parameters(self):
		# Recursively discover parameters of the module
		# and append them to the list of parameters
		params = [*self._params]
		for attr in self.__dict__.values():
			if issubclass(attr.__class__, Module):
				params.extend(attr.parameters())
		return params



class ModuleList(Module):

	def __init__(self):
		self.modules = []

	def __len__(self):
		return len(self.modules)

	def __iter__(self):
		self.index = 0
		return self

	def __getitem__(self, key):
		return self.modules[key]

	def __next__(self): 
		if self.index < len(self.modules):
			result = self.modules[self.index]
			self.index += 1
			return result
		else:
			raise StopIteration


	def __repr__(self):
		for module in self.modules:
			print(module)
		return ""

	def append(self, m):
		self.modules.append(m)

	def parameters(self):
		# overload parameters to iterate over the list of modules
		params = []
		for module in self.modules:
			params.extend(module.parameters())
		return params

	def forward(self, x):
		"""
		Apply the forward pass of the module to the input x
		"""
		raise NotImplementedError(f"Forward pass not implemented for {self.__class__.__name__}")

	def __call__(self, x):
		return self.forward(x)


