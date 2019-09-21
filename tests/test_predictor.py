import unittest
import pickle
from anomaly_detection.trainer import Trainer
from anomaly_detection.predictor import Predictor


class TestPredict(unittest.TestCase):
    def setUp(self):
        self.model_filename = "my_model_file.bin"
        with open("data/train.pickle", mode="rb") as f:
            self.train_data = pickle.load(f)["features"]
        trainer = Trainer()
        self.n_components = 3
        trainer.train(self.train_data, n_components=self.n_components)
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
        assert result["message"] == f'number of features (0) does not match trained model ({self.n_components})'
