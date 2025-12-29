import pytensor as pyt
from pytensor.function import batch_norm2d
from module import Module
import numpy as np
import tinygrad.nn as tnn
from tinygrad.tensor import Tensor
from pytensor.nn import Conv2d 

class BatchNorm2d(Module):

	def __init__(self, num_features, eps=1e-5, momentum=0.1, requires_grad=True):
		self.num_features = num_features
		self.eps = eps
		self.momentum = momentum

		# Initialize gamma (scale) and beta (shift) parameters
		self.gamma = pyt.ones((1, num_features, 1, 1), requires_grad = requires_grad)
		self.beta = pyt.zeros((1, num_features, 1, 1), requires_grad = requires_grad)
		
		# Initialize running mean and variance for inference
		self.running_mean = pyt.zeros((1, num_features, 1, 1), requires_grad=False)
		self.running_var = pyt.ones((1, num_features, 1, 1), requires_grad=False)

		self.training=True

	def train(self):
		self.training = True

	def eval(self):
		self.training = False

	def forward(self, x):
		x_normalized, self.running_mean, self.running_var = batch_norm2d(x, self.running_mean, self.running_var, self.momentum, self.eps, self.training)
		# Scale and shift
		return self.gamma * x_normalized + self.beta


if __name__ == "__main__":
	import time
	import torch
	np.set_printoptions(threshold=10000, suppress=True, edgeitems=40, linewidth=5000)
	np.random.seed(42)
	stride = 1
	padding = "same"

	m_p = Conv2d(in_channels=16, out_channels=32, stride=stride, kernel_size=3, padding=padding, bias=False, requires_grad=True)
	ln_p = BatchNorm2d(num_features=32)
	x = pyt.randn((150,16,64,64), requires_grad=True) * 1
	

	# Check que le x10 est bien ajouté a array
	t_tot = time.time()
	for _ in range(1):
		t_forward = time.time()
		x_p = m_p(x) 
		y_p = ln_p(x_p)
		print("nano grad conv forward : ",time.time()-t_forward)
		t_mean = time.time()
		l_p = y_p.sum()
		print("nano grad mean : ",time.time()-t_mean)
		t_backward = time.time()
		l_p.backward()
		print("nano grad backward : ",time.time()-t_mean)
	print("nano grad : ",time.time()-t_tot)
	print("*"*80)

	x_torch = torch.tensor(x.numpy() , requires_grad=True)
	ln_t = torch.nn.BatchNorm2d(32)
	w_torch = torch.tensor(m_p.parameters[0].array, requires_grad=True)
	t_tot = time.time()
	for _ in range(1):
		t_forward = time.time()
		x_t = torch.conv2d(x_torch, w_torch, padding=padding, stride=stride)
		y_t = ln_t(x_t)
		print("torch conv forward : ",time.time()-t_forward)

		t_mean = time.time()
		l_t = y_t.sum()
		print("torch mean : ",time.time()-t_mean)
		x_t.retain_grad()
		t_backward = time.time()
		l_t.backward()
		print("torch backward : ",time.time()-t_mean)

	print("torch : ",time.time()-t_tot)
	print("error forward : ",np.mean(np.abs(y_t.detach().numpy()-y_p.numpy())))
	print(x_p.acc_grad[0][0][0])
	print("*"*80)
	print(x_t.grad[0][0][0])
	print("error backward inputs : ",np.mean(np.abs(x.acc_grad-x_torch.grad.numpy())))

	print(ln_p.gamma.acc_grad.flatten())
	#print(l.beta.acc_grad)
	print(ln_t.weight.grad.flatten())
	#print(m.bias.grad)
