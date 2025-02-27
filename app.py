from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)

# Load the trained model and MinMaxScaler
model = pickle.load(open("new_rf_model.pkl", "rb"))
scaler = pickle.load(open("minmax_scaler.pkl", "rb"))  # Load the same scaler used in training

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get user inputs and convert to float
        features = [float(request.form[key]) for key in request.form]
        
        # Convert input to 2D array
        input_data = np.array(features).reshape(1, -1)
        
        # Normalize the input using MinMaxScaler
        normalized_input = scaler.transform(input_data)

        # Make prediction
        prediction = model.predict(normalized_input)
        crop_name = prediction[0]

        return render_template("result.html", prediction=f"Recommended Crop: {crop_name}")
    except Exception as e:
        return render_template("result.html", prediction=f"Error: {str(e)}")

if __name__ == "__main__":
    app.run(debug=True)
