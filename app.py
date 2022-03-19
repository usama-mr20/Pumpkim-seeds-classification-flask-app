import json
import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
from flask_cors import CORS

# Create flask app
flask_app = Flask(__name__)
CORS(flask_app)

model = pickle.load(open("svc_model.pkl", "rb"))


@flask_app.route("/")
def Home():
    return render_template("index.html")


@flask_app.route("/predict", methods=["POST"])
def predict():
    record = json.loads(request.data)
    for i in range(len(record.get("data"))):
        if record.get("data")[i] == '':
            return "An Err in values"

    float_features = [float(x) for x in record.get("data")]
    features = [np.array(float_features)]
    prediction = model.predict(features)
    res = " ".join(str(x) for x in prediction)
    print(res)
    return res


if __name__ == "__main__":
    flask_app.run(debug=True)
