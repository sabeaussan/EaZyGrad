from PIL import Image
from . import utils
import os
from urllib.error import URLError
import sys
import codecs
import numpy as np

def _flip_byte_order(arr):
	return (
		arr.contiguous().view(arr.uint8).view(*arr.shape, arr.element_size()).flip(-1).view(*arr.shape[:-1], -1).view(arr.dtype)
	)

def read_label_file(path):
		x = read_sn3_pascalvincent_array(path, strict=False)
		if x.dtype != np.uint8:
			raise TypeError(f"x should be of dtype np.uint8 instead of {x.dtype}")
		if x.ndim != 1:
			raise ValueError(f"x should have 1 dimension instead of {x.ndim}")
		return x.astype(np.int64)
	
def read_image_file(path):
	x = read_sn3_pascalvincent_array(path, strict=False)
	if x.dtype != np.uint8:
		raise TypeError(f"x should be of dtype np.uint8 instead of {x.dtype}")
	if x.ndim != 3:
		raise ValueError(f"x should have 3 dimension instead of {x.ndim}")
	return x

SN3_PASCALVINCENT_TYPEMAP = {
	8: np.uint8,
	9: np.int8,
	11: np.int16,
	12: np.int32,
	13: np.float32,
	14: np.float64,
}

def get_int(b):
	return int(codecs.encode(b, "hex"), 16)

def read_sn3_pascalvincent_array(path, strict= True):
	"""Read a SN3 file in "Pascal Vincent" format (Lush file 'libidx/idx-io.lsh').
	Argument may be a filename, compressed filename, or file object.
	"""
	# read
	with open(path, "rb") as f:
		data = f.read()

	# parse
	if sys.byteorder == "little" or sys.platform == "aix":
		magic = get_int(data[0:4])
		nd = magic % 256
		ty = magic // 256
	else:
		nd = get_int(data[0:1])
		ty = get_int(data[1:2]) + get_int(data[2:3]) * 256 + get_int(data[3:4]) * 256 * 256

	assert 1 <= nd <= 3
	assert 8 <= ty <= 14
	torch_type = SN3_PASCALVINCENT_TYPEMAP[ty]
	s = [get_int(data[4 * (i + 1) : 4 * (i + 2)]) for i in range(nd)]

	if sys.byteorder == "big" and not sys.platform == "aix":
		for i in range(len(s)):
			s[i] = int.from_bytes(s[i].to_bytes(4, byteorder="little"), byteorder="big", signed=False)

	parsed = np.frombuffer(bytearray(data), dtype=torch_type, offset=(4 * (nd + 1)))

	# The MNIST format uses the big endian byte order, while `torch.frombuffer` uses whatever the system uses. In case
	# that is little endian and the dtype has more than one byte, we need to flip them.
	if sys.byteorder == "little" and parsed.itemsize > 1:
		parsed = _flip_byte_order(parsed)

	assert parsed.shape[0] == np.prod(s) or not strict
	return parsed.reshape(*s)


class MNISTDataset:
	 
	mirrors = [
		"https://ossci-datasets.s3.amazonaws.com/mnist/",
		"http://yann.lecun.com/exdb/mnist/",
	]

	resources = [
		("train-images-idx3-ubyte.gz", "f68b3c2dcbeaaa9fbdd348bbdeb94873"),
		("train-labels-idx1-ubyte.gz", "d53e105ee54ea40749a09fcbcd1e9432"),
		("t10k-images-idx3-ubyte.gz", "9fb629c4189551a2d022fa330f9573f3"),
		("t10k-labels-idx1-ubyte.gz", "ec29112dd5afa0611ce80d1b7f02629c"),
	]

	def __init__(self, root="", train=True):
		self.root = root
		self.train = train  # training set or test set
		self.raw_folder = os.path.join(root, "MNIST")
		self.download()
		self.data, self.targets = self._load_data()


	def _check_exists(self):
		return all(
			utils.check_integrity(
				os.path.join(self.raw_folder, os.path.splitext(os.path.basename(url))[0])
			) for url, _ in self.resources
		)
	
	def download(self) -> None:
		"""Download the MNIST data if it doesn't exist already."""

		if self._check_exists():
			print("Files already exist, no need to download anything.")
			return

		os.makedirs(self.raw_folder, exist_ok=True)

		# download files
		for filename, md5 in self.resources:
			errors = []
			for mirror in self.mirrors:
				url = f"{mirror}{filename}"
				try:
					utils.download_and_extract_archive(url, download_root=self.raw_folder, filename=filename, md5=md5)
				except URLError as e:
					errors.append(e)
					continue
				break
			else:
				s = f"Error downloading {filename}:\n"
				for mirror, err in zip(self.mirrors, errors):
					s += f"Tried {mirror}, got:\n{str(err)}\n"
				raise RuntimeError(s)
	
	def _load_data(self):
		image_file = f"{'train' if self.train else 't10k'}-images-idx3-ubyte"
		data = read_image_file(os.path.join(self.raw_folder, image_file))

		label_file = f"{'train' if self.train else 't10k'}-labels-idx1-ubyte"
		targets = read_label_file(os.path.join(self.raw_folder, label_file))

		return data, targets
	
	def __getitem__(self, index):
		"""
		Args:
			index (int): Index

		Returns:
			tuple: (image, target) where target is index of the target class.
		"""
		img, target = self.data[index], self.targets[index]

		return img, target
	
	
				
if __name__ == "__main__":
	dataset = MNISTDataset("")
	print(dataset[10][1].dtype)
