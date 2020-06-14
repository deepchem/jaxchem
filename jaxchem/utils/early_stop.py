"""
We referred the below implementation
https://github.com/Bjarten/early-stopping-pytorch/blob/master/pytorchtools.py
"""


class EarlyStopping:
    """Early stops the training if score doesn't improve after a given patience."""

    def __init__(self, patience=10, delta=0, is_greater_better=True):
        """
        Parameters
        ----------
        patience : int
            How long to wait after last time validation loss improved, default to be 10.
        delta : float
            Minimum change in the monitored quantity to qualify as an improvement, default to be 0.
        is_greater_better : bool
            Whether the greater score is better or not default to be True.
        """
        self.patience = patience
        self.delta = delta
        self.is_greater_better = is_greater_better
        self.counter = 0
        self.best_score = None
        self.best_params = None
        self.is_train_stop = False
        self.__tmp_best_score = None

    def update(self, score, params):
        """Update early stopping counter.

        Parameters
        ----------
        score : float
            validation score per epoch
        params : Any
            all parameters of each epoch
        """
        tmp_score = score if self.is_greater_better else -score
        if self.best_score is None:
            self.__tmp_best_score = tmp_score
            self.best_score = score
            self.best_params = params
        elif tmp_score < self.__tmp_best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.is_train_stop = True
        else:
            self.__tmp_best_score = tmp_score
            self.best_score = score
            self.best_params = params
            self.counter = 0
