  
from flask import Flask, request, jsonify  
import joblib  
import pandas as pd  

# Load the trained model  
model = joblib.load("random_forest_churn_model.pkl")  

# Initialize Flask app  
app = Flask(__name__)  

@app.route("/predict", methods=["POST"])  
def predict():  
    try:  
        data = request.get_json()  
        df = pd.DataFrame([data])  
        prediction = model.predict(df)  
        return jsonify({"churn_prediction": int(prediction[0])})  
    except Exception as e:  
        return jsonify({"error": str(e)})  

if __name__ == "__main__":  
    app.run(debug=True)  
