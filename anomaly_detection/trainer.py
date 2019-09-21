import pickle
from sklearn.mixture import GaussianMixture


class InsufficientTrainingDataError(Exception):
    def __init__(self, data) -> None:
        message = (
            "Number of training data (%d examples) is insufficient to train parameters"
            % len(data)
        )
        super(Exception, self).__init__(message)

    def error_type(self):
        return "InsufficientTrainingDataError"


class Trainer(object):
    def __init__(self):
        self.model = None

    def train(self, data, n_components=1):
        if len(data) == 0:
            raise InsufficientTrainingDataError(data)

        self.model = GaussianMixture(n_components=n_components)
        self.model.fit(data)

    def save(self, filename):
        with open(filename, mode="wb") as f:
            pickle.dump(self.model, f)
