from methods.classification import Classification
import logging
import pytest

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

def test_column(data):
    with pytest.raises(KeyError):
        Classification(data,"class")