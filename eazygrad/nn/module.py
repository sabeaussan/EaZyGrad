from __future__ import annotations

from typing import Any, Iterator

import eazygrad
import abc

class Module:

	def __init__(self) -> None:
		self._params: list[Any] = []

	def register_params(self, params: Any) -> None:
		# Parameters are registered explicitly rather than discovered by type.
		self._params.append(params)

	def forward(self, *args: Any) -> Any:
		"""
		Apply the forward pass of the module to the input x
		"""
		raise NotImplementedError(f"Forward pass not implemented for {self.__class__.__name__}")

	def __call__(self, *args: Any) -> Any:
		return self.forward(*args)

	def __repr__(self) -> str:
		return f"({self.__class__.__name__})"

	def parameters(self) -> list[Any]:
		# Walk nested modules so optimizers can be built from a top-level model.
		params = [*self._params]
		for attr in self.__dict__.values():
			if issubclass(attr.__class__, Module):
				params.extend(attr.parameters())
		return params



class ModuleList(Module):
	"""
	Simple ordered container for submodules.

	Parameters
	----------
	None

	Notes
	-----
	`ModuleList` stores modules in insertion order and exposes them through
	indexing and iteration. It does not implement a forward pass by itself;
	it is intended to be used as a building block inside custom modules.
	"""

	def __init__(self) -> None:
		self.modules: list[Module] = []

	def __len__(self) -> int:
		return len(self.modules)

	def __iter__(self) -> Iterator[Module]:
		self.index = 0
		return self

	def __getitem__(self, key: int | slice) -> Module | list[Module]:
		return self.modules[key]

	def __next__(self) -> Module: 
		if self.index < len(self.modules):
			result = self.modules[self.index]
			self.index += 1
			return result
		else:
			raise StopIteration


	def __repr__(self) -> str:
		for module in self.modules:
			print(module)
		return ""

	def append(self, m: Module) -> None:
		self.modules.append(m)

	def parameters(self) -> list[Any]:
		# overload parameters to iterate over the list of modules
		params = []
		for module in self.modules:
			params.extend(module.parameters())
		return params

	def forward(self, x: Any) -> Any:
		"""
		Apply the forward pass of the module to the input x
		"""
		raise NotImplementedError(f"Forward pass not implemented for {self.__class__.__name__}")

	def __call__(self, x: Any) -> Any:
		return self.forward(x)
