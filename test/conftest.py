import eazygrad
import pytest

@pytest.fixture
def mock_dag(mocker):
    """
		Mock all dag local references
    """
    mocker.patch.object(eazygrad._tensor, "dag")
    mocker.patch.object(eazygrad.tensor_factories, "dag")
    mocker.patch.object(eazygrad.functions.activations, "dag")
    mocker.patch.object(eazygrad.functions.specials, "dag")
    mocker.patch.object(eazygrad.functions.math, "dag")