import eazygrad
import abc

class Module(abc.ABC):

	params = []

	def register_params(self, params):
		self.params.append(params)

	@abc.abstractmethod
	def forward(self, x):
		"""
		Apply the forward pass of the module to the input x
		"""
		raise NotImplementedError(f"Forward pass not implemented for {self.__class__.__name__}")

	def __call__(self, x):
		return self.forward(x)

	def __repr__(self):
		return f"({self.__class__.__name__})"

	def parameters(self):
		return self.params



class ModuleList:

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
		parameters = []
		for module in self.modules:
			for p in module.params:
				parameters.append(p)
		return parameters

	def forward(self, x):
		"""
		Apply the forward pass of the module to the input x
		"""
		raise NotImplementedError(f"Forward pass not implemented for {self.__class__.__name__}")

	def __call__(self, x):
		return self.forward(x)


