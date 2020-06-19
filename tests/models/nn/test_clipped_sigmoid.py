import unittest

import numpy as np
from jaxchem.models.nn import clipped_sigmoid


class TestClippedSigmoid(unittest.TestCase):
    """Test clipped_sigmoid function."""

    def test_clipping(self):
        "Test clipping."
        # large x
        x = np.array([100.0])
        out = clipped_sigmoid(x)
        np.testing.assert_allclose(out, np.array([1.0 - 1.0e-15]))

        # small x
        x = np.array([-100.0])
        out = clipped_sigmoid(x)
        np.testing.assert_allclose(out, np.array([1.0e-15]))
