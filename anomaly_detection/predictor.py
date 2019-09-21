import pickle
import math


class PredictionResult(dict):
    def __init__(self, is_anomaly, score, is_error, message):
        dict.__init__(
            self, is_anomaly=is_anomaly, score=score, is_error=is_error, message=message
        )


class Predictor(object):
    def __init__(self, trainer=None) -> None:
        self.trainer = trainer
        self.is_anomaly = False
        self.is_error = False
        self.message = None
        self.score = None

    def init_result(self):
        self.is_anomaly = False
        self.is_error = False
        self.message = None
        self.score = None

    def load(self, filename):
        with open(filename, mode="rb") as f:
            self.trainer = pickle.load(f)

    def predict(self, features, threshold=0.5):
        # trainerを使って賢く分類するようにしましょう
        if len(features) != self.trainer.means_.shape[1]:
            self.is_error = True
            self.message = f'number of features ({len(features)}) ' \
                           f'does not match trained model ({self.trainer.means_.shape[1]})'
            results = PredictionResult(
                is_anomaly=self.is_anomaly, score=self.score, is_error=self.is_error, message=self.message
            )

        else:
            log_prob = self.trainer.score([features])
            prob = math.exp(log_prob)
            if prob <= threshold:
                self.is_anomaly = True
            results = PredictionResult(
                is_anomaly=self.is_anomaly, score=prob, is_error=self.is_error, message=self.message
            )
        return results
