import unittest

from jaxchem.utils import EarlyStopping


class TestEarlyStopping(unittest.TestCase):
    """Test EarlyStopping class"""

    def test_update_with_positive_score(self):
        """Test update function for positive score"""
        # is_greater_better is True
        early_stop = EarlyStopping(is_greater_better=True)
        early_stop.update(score=1.0, params=None)
        assert early_stop.counter == 0
        assert early_stop.best_score == 1.0
        assert early_stop.is_train_stop is False
        # this is not updated
        early_stop.update(score=0.5, params=None)
        assert early_stop.counter == 1
        assert early_stop.best_score == 1.0
        # update
        early_stop.update(score=1.5, params=None)
        assert early_stop.counter == 0
        assert early_stop.best_score == 1.5

        # is_greater_better is False
        early_stop = EarlyStopping(is_greater_better=False)
        early_stop.update(score=1.0, params=None)
        # update
        early_stop.update(score=0.5, params=None)
        assert early_stop.counter == 0
        assert early_stop.best_score == 0.5
        # this is not updated)
        early_stop.update(score=1.5, params=None)
        assert early_stop.counter == 1
        assert early_stop.best_score == 0.5

    def test_update_with_negative_score(self):
        """Test update function for negative score"""
        early_stop = EarlyStopping(is_greater_better=True)
        early_stop.update(score=-1.0, params=None)
        # first update
        assert early_stop.counter == 0
        assert early_stop.best_score == -1.0
        assert early_stop.is_train_stop is False
        # update
        early_stop.update(score=-0.5, params=None)
        assert early_stop.counter == 0
        assert early_stop.best_score == -0.5
        # this is not updated
        early_stop.update(score=-1.5, params=None)
        assert early_stop.counter == 1
        assert early_stop.best_score == -0.5

        early_stop = EarlyStopping(is_greater_better=False)
        early_stop.update(score=-1.0, params=None)
        # this is not updated)
        early_stop.update(score=-0.5, params=None)
        assert early_stop.counter == 1
        assert early_stop.best_score == -1.0
        # update
        early_stop.update(score=-1.5, params=None)
        assert early_stop.counter == 0
        assert early_stop.best_score == -1.5

    def test_is_train_stop(self):
        """Test is_train_stop variable"""
        # is_greater_better is True
        patience = 5
        init_score = 6
        early_stop = EarlyStopping(patience=patience, is_greater_better=True)
        early_stop.update(score=init_score, params=None)
        for i in range(5):
            assert early_stop.is_train_stop is False
            early_stop.update(score=i, params=None)
        assert early_stop.is_train_stop

        # is_greater_better is False
        patience = 5
        init_score = 0
        early_stop = EarlyStopping(patience=patience, is_greater_better=False)
        early_stop.update(score=init_score, params=None)
        for i in range(5):
            assert early_stop.is_train_stop is False
            early_stop.update(score=i + 1, params=None)
        assert early_stop.is_train_stop
