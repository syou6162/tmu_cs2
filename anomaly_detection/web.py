import sys
import os
import json

from flask import Flask, request, Response

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/..")
from anomaly_detection.predictor import Predictor


def make_json_response(body, status: int = 200):
    return Response(json.dumps(body), mimetype="application/json", status=status)


app = Flask(__name__)


@app.route("/", methods=["GET"])
def index():
    return Response("Hello world!\n", mimetype="text/plain")


@app.route("/predict", methods=["POST"])
def predict():
    if not request.json:
        return make_json_response({"message": "Invalid content type"}, status=400)
    content = request.json

    predictor = Predictor()
    predictor.load("model_file.bin")
    result = predictor.predict(content["features"])
    return make_json_response(result)


if __name__ == "__main__":
    app.run(host="0.0.0.0")
