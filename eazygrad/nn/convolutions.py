import pytensor as pyt
from module import Module
import numpy as np
from pytensor.function import conv2d, max_pool2d
import pyfftw
import multiprocessing
from pytensor.utils import compute_padding_value

class Conv2d(Module):

	def __init__(self, in_channels, out_channels, kernel_size, stride=1,  padding="same", mode="fft",  bias=True, requires_grad=True):
		self.in_channels = in_channels
		self.out_channels = out_channels
		self.kernel_size = kernel_size
		self.stride = stride
		self.padding = padding
		self.padding_value = compute_padding_value(kernel_size, padding)
		self.mode = mode
		gain = np.sqrt(1/self.out_channels)
		self.parameters = [pyt.uniform(low=-100, high=100, size=(out_channels, in_channels, kernel_size, kernel_size), requires_grad=requires_grad)]
		if bias:
			self.parameters.append(pyt.uniform(low=-gain, high=gain, size=(1, out_channels, 1, 1), requires_grad=requires_grad))

	def forward(self, x):
		# TODO: assert tensor size
		y = conv2d(x, self.parameters[0], self.padding_value, self.stride, mode=self.mode)
		if len(self.parameters) > 1:
			y += self.parameters[1]
		return y


	def __repr__(self):
		a = super().__repr__()
		return f"------> n_in : {self.n_in}  |  n_out : {self.n_out}"

if __name__ == "__main__":
	import time
	import torch
	#np.set_printoptions(threshold=10000, suppress=True, edgeitems=40, linewidth=5000)
	stride = 1
	padding = "valid"
	mode="fft"

	np.random.seed(42)
	l = Conv2d(in_channels=16, out_channels=32, stride=stride, kernel_size=3, padding=padding, mode="fft", bias=True, requires_grad=True)
	
	x = pyt.randn((150,16,64,64), requires_grad=True) * 10

	y_p = l(x)
	y_p = max_pool2d(input=y_p, kernel_size=3, out=None)
	l_p = y_p.mean()
	l_p.backward()
	
	x = pyt.randn((150,16,64,64), requires_grad=True) * 10
	t_tot = time.time()
	for _ in range(1):
		t_forward = time.time()
		y_p = l(x)
		print("nano grad conv forward : ",time.time()-t_forward)
		
		t_forward = time.time()
		y_p_p = max_pool2d(input=y_p, kernel_size=3, out=None)
		print("nano pool forward : ",time.time()-t_forward)
		t_mean = time.time()
		l_p = y_p_p.mean()
		print("nano grad mean : ",time.time()-t_mean)
		t_backward = time.time()
		l_p.backward()
		print("nano grad backward : ",time.time()-t_mean)
	print("nano grad : ",time.time()-t_tot)
	#print(y_p)
	print("*"*80)

	x_torch = torch.tensor(x.numpy() , requires_grad=True)
	w_torch = torch.tensor(l.parameters[0].array, requires_grad=True)
	b_torch = torch.tensor(l.parameters[1].array, requires_grad=True)
	t_tot = time.time()
	for _ in range(1):
		t_forward = time.time()
		y_t = torch.conv2d(x_torch, w_torch, padding=padding, stride=stride, bias=b_torch.flatten())
		print("torch conv forward : ",time.time()-t_forward)
		t_forward = time.time()
		y_t_p = torch.nn.functional.max_pool2d(y_t, kernel_size=3)
		print("torch pool forward : ",time.time()-t_forward)
		t_mean = time.time()
		l_t = y_t_p.mean()
		y_t.retain_grad()
		print("torch mean : ",time.time()-t_mean)
		t_backward = time.time()
		l_t.backward()
		print("torch backward : ",time.time()-t_mean)
	print("torch : ",time.time()-t_tot)
	#print(y_t)
	#print(y_p.numpy()/y_t.detach().numpy())
	print("error forward : ",np.mean(np.abs(y_t.detach().numpy()-y_p.numpy())))
	print(w_torch.grad.flatten())
	print("*"*80)
	print(l.parameters[0].grad.flatten())
	print("*"*80)
	print(b_torch.grad.flatten())
	print("*"*80)
	print(l.parameters[1].grad.flatten())
	print("*"*80)
	print(y_t.detach().numpy().flatten())
	print("*"*80)
	print(y_p.numpy().flatten())
	print("*"*80)
	print("error backward weights : ",np.mean(np.abs(w_torch.grad.detach().numpy()-l.parameters[0].grad)))
	print("error backward bias : ",np.mean(np.abs(b_torch.grad.detach().numpy()-l.parameters[1].grad)))
	print("error backward inputs : ",np.mean(np.abs(x_torch.grad.detach().numpy()-x.grad)))

	"""out = np.empty((150, 21, 21, 16, 3, 3), dtype=np.float32)
	y_p = max_pool2d(input=x, kernel_size=3, padding_value=0, stride=3, out=out)
	x = pyt.randn((150,16,64,64), requires_grad=True) * 10
	out = np.empty((150, 21, 21, 16, 3, 3), dtype=np.float32)
	for _ in range(1):
		t_forward = time.time()
		y_p = max_pool2d(input=x, kernel_size=3, padding_value=0, stride=3, out=out)
		#print(y_p[0][0])
		print("nano grad forward : ",time.time()-t_forward)

	for _ in range(1):
		t_forward = time.time()
		y_t = torch.nn.functional.max_pool2d(x_torch, kernel_size=3, stride=3)
		#print(y_t[0][0])
		print("torch forward : ",time.time()-t_forward)

	print("error forward : ",np.mean(np.abs(y_t.detach().numpy()-y_p.numpy())))"""

	

