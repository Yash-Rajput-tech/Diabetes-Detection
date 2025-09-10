from flask import Flask, render_template, request
import numpy as np
import pickle
import tensorflow as tf

app = Flask(__name__)

# Load model and scaler
model = tf.keras.models.load_model("diabetes_model.h5")
scaler = pickle.load(open("scaler.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if request.method == "POST":
        # Collect input from form
        features = [float(x) for x in request.form.values()]
        features = np.array(features).reshape(1, -1)

        # Scale input
        features_scaled = scaler.transform(features)

        # Predict
        prediction = model.predict(features_scaled)
        probability = float(prediction[0][0])  # sigmoid output (0–1)
        percentage = round(probability * 100, 2)  # convert to %

        # Threshold for classification
        if probability > 0.5:
            result = f"Diabetic (Risk: {percentage}%)"
        else:
            result = f"Not Diabetic (Risk: {percentage}%)"

        return render_template("index.html", prediction_text=f"Prediction: {result}")

if __name__ == "__main__":
    app.run(debug=True)