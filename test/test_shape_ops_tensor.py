import numpy as np
import pytest
from unittest.mock import MagicMock
from hypothesis import given, strategies as st
import hypothesis.extra.numpy as hnp
import pytensor

# TODO : test reshape, squeeze, expand_dims, get_item, transpose, swap_axis...