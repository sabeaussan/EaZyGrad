import random
import numpy as np

class Dataloader:
	"""
		Very simple dataloader with no multiprocessing (mostly for MNIST which is already loaded in RAM)
	"""

	def __init__(self, dataset, batch_size, shuffle=True, drop_last=True):
		self.dataset = dataset
		self.batch_size = batch_size
		size = len(dataset.data)
		self.indices = list(range(size))
		self.num_batch = size // self.batch_size
		remainder = size % self.batch_size
		if remainder != 0 and not drop_last:
			self.num_batch += 1
		self.shuffle = shuffle
		self.drop_last = drop_last

	def __iter__(self):
		if self.shuffle:
			random.shuffle(self.indices)
		for i in range(self.num_batch):
			batch_idx = self.indices[i*self.batch_size:(i+1)*self.batch_size]
			d = self.dataset.data[batch_idx]
			t = self.dataset.targets[batch_idx]
			yield (d,t)
