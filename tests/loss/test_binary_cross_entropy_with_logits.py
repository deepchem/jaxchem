import unittest

from jaxchem.loss import binary_cross_entropy_with_logits


class TestBCEWithLogits(unittest.TestCase):
    """Test BCEWithLogits class"""

    def test_update_with_positive_score(self):
        assert False