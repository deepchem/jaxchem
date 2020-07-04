import unittest

import pytest
import numpy as np
from jaxchem.loss import binary_cross_entropy_with_logits


class TestBCEWithLogits(unittest.TestCase):
    """Test BCEWithLogits class"""

    def test_bce_with_logits(self):
        inputs = np.random.normal(size=(6,))
        targets = np.array([0, 1, 1, 0, 0, 1])
        loss = binary_cross_entropy_with_logits(inputs, targets)
        assert loss > 0

        inputs = np.random.normal(size=(3, 6))
        targets = np.array([
            [0, 1, 1, 0, 0, 1],
            [1, 1, 0, 0, 0, 1],
            [1, 1, 1, 0, 0, 0],
        ])
        loss = binary_cross_entropy_with_logits(inputs, targets)
        assert loss > 0

        inputs = np.random.normal(size=(3, 6))
        targets = np.array([
            [0, 1, 1, 0, 0, 1],
            [1, 1, 0, 0, 0, 1],
            [1, 1, 1, 0, 0, 0],
        ])
        loss = binary_cross_entropy_with_logits(inputs, targets, average=False)
        assert loss > 0

    def test_bce_with_logits_with_invalid_shape(self):
        with pytest.raises(ValueError):
            inputs = np.random.normal(size=(1, 2))
            targets = np.array([0, 1, 1, 0, 0, 1])
            binary_cross_entropy_with_logits(inputs, targets)
