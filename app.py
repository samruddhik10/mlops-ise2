from flask import Flask, request, jsonify
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

app = Flask(__name__)

# Train a small ML model using the Iris dataset
iris = load_iris()
X, y = iris.data, iris.target

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)
model.fit(X, y)

request_count = 0


@app.route("/")
def home():
    return """
    <h1>ML DevOps Prediction App</h1>
    <p>Application is running successfully.</p>
    <p>Use /health to check application health.</p>
    <p>Use /metrics to view request metrics.</p>
    <p>Use /predict to predict a flower species.</p>
    """


@app.route("/health")
def health():
    return jsonify({"status": "healthy"}), 200


@app.route("/metrics")
def metrics():
    return jsonify({
        "total_prediction_requests": request_count
    }), 200


@app.route("/predict", methods=["POST"])
def predict():
    global request_count

    data = request.get_json(silent=True)

    if not isinstance(data, dict):
        return jsonify({"error": "Send a JSON object"}), 400

    features = [
        "sepal_length",
        "sepal_width",
        "petal_length",
        "petal_width"
    ]

    try:
        values = [float(data[key]) for key in features]
    except (KeyError, TypeError, ValueError):
        return jsonify({
            "error": "Provide all four measurements as numbers"
        }), 400

    if not all(0 < value <= 10 for value in values):
        return jsonify({
            "error": "Measurements must be greater than 0 and at most 10"
        }), 400

    prediction = int(model.predict([values])[0])
    request_count += 1

    return jsonify({
        "predicted_species": iris.target_names[prediction],
        "model": "RandomForestClassifier"
    }), 200


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=False)