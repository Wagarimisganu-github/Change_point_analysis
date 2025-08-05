from flask import Flask, request, jsonify
import joblib
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from flask_cors import CORS  


app = Flask(__name__)


CORS(app)


model = joblib.load('brent_oil_lstm_model.pkl')


scaler = MinMaxScaler()

@app.route('/')
def home():
    return "Brent Oil Price Prediction API"

@app.route('/favicon.ico')
def favicon():
    return '', 204  

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json() 
    if not data or 'prices' not in data:
        return jsonify({'error': 'Invalid input, must contain "prices"'}), 400
    
    
    prices = np.array(data['prices']).reshape(-1, 1)
    
    
    scaled_prices = scaler.fit_transform(prices)

    
    X_input = []
    for i in range(60, len(scaled_prices)):
        X_input.append(scaled_prices[i-60:i, 0])
    X_input = np.array(X_input)
    X_input = np.reshape(X_input, (X_input.shape[0], X_input.shape[1], 1))


    predictions = model.predict(X_input)
    predicted_prices = scaler.inverse_transform(predictions)  

  
    return jsonify({'predicted_prices': predicted_prices.flatten().tolist()}), 200

if __name__ == '__main__':
    app.run(debug=True)
