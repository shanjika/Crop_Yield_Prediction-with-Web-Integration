from flask import Flask, request, render_template
import numpy as np
import pickle

app = Flask(__name__)

# Load model and encoders using Pickle
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
le_area = pickle.load(open("le_area.pkl", "rb"))
le_item = pickle.load(open("le_item.pkl", "rb"))

@app.route("/", methods=["GET"])
def home():
    return render_template("index.html",
                           prediction=None,
                           message=None,
                           is_good=None,
                           form_data={})

@app.route("/predict", methods=["POST"])
def predict():
    try:
        form_data = request.form
        year = int(form_data["year"])
        rainfall = float(form_data["average_rain_fall_mm_per_year"])
        pesticides = float(form_data["pesticides_tonnes"])
        avg_temp = float(form_data["avg_temp"])
        area = form_data["area"]
        item = form_data["item"]

        # Input validation
        if not (0 <= avg_temp <= 50):
            return render_template("index.html", message="🌡️ Temperature out of realistic range.", form_data=form_data)
        if not (0 <= rainfall <= 3000):
            return render_template("index.html", message="🌧️ Rainfall out of realistic range.", form_data=form_data)
        if not (0 <= pesticides <= 500):
            return render_template("index.html", message="🧪 Pesticide usage out of realistic range.", form_data=form_data)

        area_encoded = le_area.transform([area])[0] if area in le_area.classes_ else -1
        item_encoded = le_item.transform([item])[0] if item in le_item.classes_ else -1

        features = np.array([[year, rainfall, pesticides, avg_temp, area_encoded, item_encoded]])
        features_scaled = scaler.transform(features)

        pred_yield = model.predict(features_scaled)[0]
        pred_yield_rounded = round(pred_yield, 2)

        GOOD_YIELD_THRESHOLD = 3000
        is_yield_high = pred_yield >= GOOD_YIELD_THRESHOLD

        reasons = []
        if avg_temp > 35:
            reasons.append("Temperature is too high")
        elif avg_temp < 15:
            reasons.append("Temperature is too low")
        if rainfall < 500:
            reasons.append("Rainfall is too low")
        elif rainfall > 2500:
            reasons.append("Rainfall is too high")
        if pesticides > 500:
            reasons.append("Excess pesticide usage")

        reason_text = ", ".join(reasons) if reasons else None

        alternatives = []
        for crop in le_item.classes_:
            if crop != item:
                crop_encoded = le_item.transform([crop])[0]
                alt_features = np.array([[year, rainfall, pesticides, avg_temp, area_encoded, crop_encoded]])
                alt_scaled = scaler.transform(alt_features)
                alt_yield = model.predict(alt_scaled)[0]
                if alt_yield > pred_yield:
                    alternatives.append((crop, round(alt_yield, 2)))

        alternatives.sort(key=lambda x: x[1], reverse=True)

        if is_yield_high and not reasons:
            message = f"✅ Predicted Yield: {pred_yield_rounded} → Great! This crop is a good choice."
            is_good = True
        elif alternatives:
            message = f"⚠️ Predicted Yield: {pred_yield_rounded} → Not ideal.<br>Suggested better crop varieties:<br>"
            for alt in alternatives[:3]:
                message += f"✔ {alt[0]} → Yield: {alt[1]}<br>"
            if reason_text:
                message += f"<br><strong>Reason for low yield:</strong> {reason_text}"
            is_good = False
        else:
            message = f"⚠️ Predicted Yield: {pred_yield_rounded} → Low yield and no better crop alternatives found."
            if reason_text:
                message += f"<br><strong>Reason:</strong> {reason_text}"
            is_good = False

        return render_template("index.html",
                               prediction=pred_yield_rounded,
                               message=message,
                               is_good=is_good,
                               form_data=form_data)

    except Exception as e:
        return f"❌ Error occurred: {str(e)}"

if __name__ == "__main__":
    app.run(debug=True)

