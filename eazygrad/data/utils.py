import os
import pathlib
import urllib
import urllib.request
from tqdm import tqdm
import zipfile
import tarfile
import bz2
import gzip
import lzma

def _decompress(from_path, to_path = None, remove_finished = False):
	r"""Decompress a file.

	The compression is automatically detected from the file name.

	Args:
		from_path (str): Path to the file to be decompressed.
		to_path (str): Path to the decompressed file. If omitted, ``from_path`` without compression extension is used.
		remove_finished (bool): If ``True``, remove the file after the extraction.

	Returns:
		(str): Path to the decompressed file.
	"""
	suffix, archive_type, compression = _detect_file_type(from_path)
	if not compression:
		raise RuntimeError(f"Couldn't detect a compression from suffix {suffix}.")

	if to_path is None:
		to_path = pathlib.Path(os.fspath(from_path).replace(suffix, archive_type if archive_type is not None else ""))

	# We don't need to check for a missing key here, since this was already done in _detect_file_type()
	compressed_file_opener = _COMPRESSED_FILE_OPENERS[compression]

	with compressed_file_opener(from_path, "rb") as rfh, open(to_path, "wb") as wfh:
		wfh.write(rfh.read())

	if remove_finished:
		os.remove(from_path)

	return pathlib.Path(to_path)

def _extract_tar(from_path, to_path, compression):
	with tarfile.open(from_path, f"r:{compression[1:]}" if compression else "r") as tar:
		tar.extractall(to_path)
		
_ZIP_COMPRESSION_MAP = {
	".bz2": zipfile.ZIP_BZIP2,
	".xz": zipfile.ZIP_LZMA,
}

def _extract_zip(from_path, to_path, compression):
	with zipfile.ZipFile(
		from_path, "r", compression=_ZIP_COMPRESSION_MAP[compression] if compression else zipfile.ZIP_STORED
	) as zip:
		zip.extractall(to_path)

USER_AGENT = "eazygrad"

_ARCHIVE_EXTRACTORS = {
	".tar": _extract_tar,
	".zip": _extract_zip,
}
_COMPRESSED_FILE_OPENERS = {
	".bz2": bz2.open,
	".gz": gzip.open,
	".xz": lzma.open,
}
_FILE_TYPE_ALIASES = {
	".tbz": (".tar", ".bz2"),
	".tbz2": (".tar", ".bz2"),
	".tgz": (".tar", ".gz"),
}

def _urlretrieve(url, filename, chunk_size = 1024 * 32):
	with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": USER_AGENT})) as response:
		with open(filename, "wb") as fh, tqdm(total=response.length, unit="B", unit_scale=True) as pbar:
			while chunk := response.read(chunk_size):
				fh.write(chunk)
				pbar.update(len(chunk))

def _get_redirect_url(url: str, max_hops: int = 3) -> str:
	initial_url = url
	headers = {"Method": "HEAD", "User-Agent": USER_AGENT}

	for _ in range(max_hops + 1):
		with urllib.request.urlopen(urllib.request.Request(url, headers=headers)) as response:
			if response.url == url or response.url is None:
				return url

			url = response.url
	else:
		raise RecursionError(
			f"Request to {initial_url} exceeded {max_hops} redirects. The last redirect points to {url}."
		)

def _detect_file_type(file):
	"""Detect the archive type and/or compression of a file.

	Args:
		file (str): the filename

	Returns:
		(tuple): tuple of suffix, archive type, and compression

	Raises:
		RuntimeError: if file has no suffix or suffix is not supported
	"""
	suffixes = pathlib.Path(file).suffixes
	if not suffixes:
		raise RuntimeError(
			f"File '{file}' has no suffixes that could be used to detect the archive type and compression."
		)
	suffix = suffixes[-1]

	# check if the suffix is a known alias
	if suffix in _FILE_TYPE_ALIASES:
		return (suffix, *_FILE_TYPE_ALIASES[suffix])

	# check if the suffix is an archive type
	if suffix in _ARCHIVE_EXTRACTORS:
		return suffix, suffix, None

	# check if the suffix is a compression
	if suffix in _COMPRESSED_FILE_OPENERS:
		# check for suffix hierarchy
		if len(suffixes) > 1:
			suffix2 = suffixes[-2]

			# check if the suffix2 is an archive type
			if suffix2 in _ARCHIVE_EXTRACTORS:
				return suffix2 + suffix, suffix2, suffix

		return suffix, None, suffix

	valid_suffixes = sorted(set(_FILE_TYPE_ALIASES) | set(_ARCHIVE_EXTRACTORS) | set(_COMPRESSED_FILE_OPENERS))
	raise RuntimeError(f"Unknown compression or archive type: '{suffix}'.\nKnown suffixes are: '{valid_suffixes}'.")

def extract_archive(from_path, to_path = None, remove_finished = False):
	"""Extract an archive.

	The archive type and a possible compression is automatically detected from the file name. If the file is compressed
	but not an archive the call is dispatched to :func:`decompress`.

	Args:
		from_path (str): Path to the file to be extracted.
		to_path (str): Path to the directory the file will be extracted to. If omitted, the directory of the file is
			used.
		remove_finished (bool): If ``True``, remove the file after the extraction.

	Returns:
		(str): Path to the directory the file was extracted to.
	"""

	def path_or_str(ret_path):
		if isinstance(from_path, str):
			return os.fspath(ret_path)
		else:
			return ret_path

	if to_path is None:
		to_path = os.path.dirname(from_path)

	suffix, archive_type, compression = _detect_file_type(from_path)
	if not archive_type:
		ret_path = _decompress(
			from_path,
			os.path.join(to_path, os.path.basename(from_path).replace(suffix, "")),
			remove_finished=remove_finished,
		)
		return path_or_str(ret_path)

	# We don't need to check for a missing key here, since this was already done in _detect_file_type()
	extractor = _ARCHIVE_EXTRACTORS[archive_type]

	extractor(from_path, to_path, compression)
	if remove_finished:
		os.remove(from_path)

	return path_or_str(pathlib.Path(to_path))

def download_url(url, root, filename = None, md5 = None, max_redirect_hops = 3):
	"""Download a file from a url and place it in root.

	Args:
		url (str): URL to download file from
		root (str): Directory to place downloaded file in
		filename (str, optional): Name to save the file under. If None, use the basename of the URL
		md5 (str, optional): MD5 checksum of the download. If None, do not check
		max_redirect_hops (int, optional): Maximum number of redirect hops allowed
	"""
	root = os.path.expanduser(root)
	if not filename:
		filename = os.path.basename(url)
	fpath = os.fspath(os.path.join(root, filename))

	os.makedirs(root, exist_ok=True)

	# check if file is already present locally
	if check_integrity(fpath):
		return

	# expand redirect chain if needed
	url = _get_redirect_url(url, max_hops=max_redirect_hops)

	# check if file is located on Google Drive
	# file_id = _get_google_drive_file_id(url)
	# if file_id is not None:
	# 	return download_file_from_google_drive(file_id, root, filename, md5)

	# download the file
	try:
		_urlretrieve(url, fpath)
	except (urllib.error.URLError, OSError) as e:  # type: ignore[attr-defined]
		if url[:5] == "https":
			url = url.replace("https:", "http:")
			_urlretrieve(url, fpath)
		else:
			raise e

	print(fpath)
	# check integrity of downloaded file
	if not check_integrity(fpath):
		raise RuntimeError("File not found or corrupted.")

def check_integrity(fpath):
	# TODO : add checksum
	if not os.path.isfile(fpath):
		return False
	return True

def download_and_extract_archive(url, download_root, extract_root = None, filename = None, md5 = None, remove_finished = False):
	download_root = os.path.expanduser(download_root)
	if extract_root is None:
		extract_root = download_root
	if not filename:
		filename = os.path.basename(url)

	download_url(url, download_root, filename, md5)

	archive = os.path.join(download_root, filename)
	extract_archive(archive, extract_root, remove_finished)