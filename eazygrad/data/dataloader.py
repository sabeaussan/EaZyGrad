import random
import numpy as np

class Dataloader:
	"""
		Very simple dataloader with no multiprocessing (mostly for MNIST which already loaded in RAM)
	"""

	def __init__(self, dataset, batch_size, shuffle=True, drop_last=True):
		self.dataset = dataset
		self.batch_size = batch_size
		size = len(dataset.data)
		self.indices = list(range(size))
		self.num_batch = size // self.batch_size
		self.drop_last = drop_last
		if shuffle:
			random.shuffle(self.indices)

	def __iter__(self):
		for i in range(self.num_batch):
			d = self.dataset.data[i*self.batch_size:(i+1)*self.batch_size]
			t = self.dataset.targets[i*self.batch_size:(i+1)*self.batch_size]
			yield (d,t)

		if not self.drop_last:
			d = self.dataset.data[(i+1)*self.batch_size:]
			t = self.dataset.targets[(i+1)*self.batch_size:]
			yield (d,t)
