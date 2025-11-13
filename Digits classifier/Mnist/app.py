from flask import Flask, request, jsonify, send_from_directory
import numpy as np

W = np.load("weights.npy")
b = np.load("bias.npy")

app = Flask(__name__, static_folder=".")

def softmax(z):
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

@app.route("/")
def home():
    return send_from_directory(".", "index.html")

@app.route("/predict", methods=["POST"])
def predict_api():
    data = request.json.get("image", None)
    if data is None:
        return jsonify({"error": "No image provided"}), 400
    x = np.array(data).flatten().reshape(1, 784) * 255.0

    z = np.dot(x, W) + b
    probs = softmax(z)
    pred = int(np.argmax(probs))
    return jsonify({"prediction": pred})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001)
