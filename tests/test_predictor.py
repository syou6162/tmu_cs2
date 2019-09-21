import unittest
import pickle
from anomaly_detection.trainer import Trainer
from anomaly_detection.predictor import Predictor
from sklearn.metrics import classification_report


class TestPredict(unittest.TestCase):
    def setUp(self):
        self.model_filename = "my_model_file.bin"
        with open("data/train.pickle", mode="rb") as train_file, open("data/test.pickle", mode="rb") as test_file:
            self.train_data = pickle.load(train_file)["features"]
            self.test_data = pickle.load(test_file)
        trainer = Trainer()
        trainer.train(self.train_data, n_components=1)
        trainer.save(self.model_filename)

    def test_predict(self):
        predictor = Predictor()
        predictor.load(self.model_filename)
        result = predictor.predict([1, 2, 3])
        # assert result["is_anomaly"] is False
        assert result["is_error"] is False
        assert result["message"] is None

    # 様々な入力ケースに対してテストを書いていきましょう
    def test_predict_with_insufficient_data(self):
        predictor = Predictor()
        predictor.load(self.model_filename)
        result = predictor.predict([])
        assert result["is_anomaly"] is False
        assert result['is_error'] is True
        assert result["score"] is None
        assert result["message"] == f'number of features (0) ' \
                                    f'does not match trained model ({predictor.trainer.means_.shape[1]})'

    def train_and_predict(self, n_components=1, threshold=0.001):
        trainer = Trainer()
        trainer.train(self.train_data, n_components=n_components)
        predictor = Predictor(trainer=trainer.model)
        y_pred = list()
        for X in self.X_test:
            result = predictor.predict(X, threshold=threshold)
            y_pred.append(1 if result['is_anomaly'] else 0)
            predictor.init_result()
        return y_pred

    def test_score(self):
        self.y_test = self.test_data['labels']
        self.X_test = self.test_data['features']
        score_weak = classification_report(
            self.y_test,
            self.train_and_predict(n_components=1, threshold=0.001),
            output_dict=True
        )
        score_strong = classification_report(
            self.y_test,
            self.train_and_predict(n_components=2, threshold=0.01),
            output_dict=True
        )
        assert score_weak['0']['precision'] <= score_strong['0']['precision']
        assert score_weak['0']['recall'] <= score_strong['0']['recall']
        assert score_weak['0']['f1-score'] <= score_strong['0']['f1-score']
        assert score_weak['1']['precision'] <= score_strong['1']['precision']
        assert score_weak['1']['recall'] <= score_strong['1']['recall']
        assert score_weak['1']['f1-score'] <= score_strong['1']['f1-score']
