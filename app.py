from flask import Flask, render_template, request, redirect, url_for
import joblib
import numpy as np
import os

app = Flask(__name__)

model = joblib.load("model.pkl")
results = joblib.load("results.pkl")

history = []

@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        temp = float(request.form["temperature"])
        humidity = float(request.form["humidity"])
        occupancy = float(request.form["occupancy"])
        hour = float(request.form["hour"])

        data = np.array([[temp, humidity, occupancy, hour]])
        prediction = round(model.predict(data)[0], 2)

        # Save prediction to history
        history.append(prediction)

        # Redirect to clear POST (PRG pattern)
        return redirect(url_for("home", pred=prediction))

    # GET request (fresh load)
    prediction = request.args.get("pred")
    suggestion = None
    efficiency = None

    if prediction:
        prediction = float(prediction)

        if prediction > 200:
            suggestion = "⚠️ High energy usage! Optimize HVAC."
            efficiency = 60
        elif prediction > 120:
            suggestion = "⚡ Moderate usage."
            efficiency = 75
        else:
            suggestion = "✅ Efficient usage."
            efficiency = 90

    return render_template(
        "index.html",
        prediction=prediction,
        suggestion=suggestion,
        history=history,
        results=results,
        efficiency=efficiency
    )





if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
