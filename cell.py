from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load the trained model
model = joblib.load("CellPricePrediction.pkl")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Read form values
        ram = int(request.form["ram"])
        battery_power = int(request.form["battery_power"])
        px_height = int(request.form["px_height"])
        px_width = int(request.form["px_width"])
        three_g = int(request.form["three_g"])
        four_g = int(request.form["four_g"])

        # Feature vector (must match training order)
        features = np.array([[
            ram,
            battery_power,
            px_height,
            px_width,
            three_g,
            four_g
        ]])

        # Predict
        prediction = int(model.predict(features)[0])

        # Labels
        price_labels = {
            0: "📱 Low Cost Phone",
            1: "📲 Medium Cost Phone",
            2: "💎 High Cost Phone",
            3: "👑 Premium Phone"
        }

        result = price_labels[prediction]

        return render_template(
            "result.html",
            prediction=prediction,
            result=result,
            ram=ram,
            battery_power=battery_power,
            px_height=px_height,
            px_width=px_width,
            three_g=three_g,
            four_g=four_g
        )

    except Exception as e:
        return render_template(
            "index.html",
            prediction_text=f"Error: {e}"
        )


if __name__ == "__main__":
    app.run(debug=True)