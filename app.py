from flask import Flask, jsonify, render_template
import pandas as pd
import joblib
import requests
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

# Load model
model = joblib.load('random_forest_model.pkl')

DATA_URL = 'https://api.coincap.io/v2/assets/bitcoin/history'

def fetch_and_preprocess_data():
    END_DATE = datetime.utcnow()
    START_DATE = END_DATE - timedelta(days=15)
    params = {
        'start': int(START_DATE.timestamp() * 1000),
        'end': int(END_DATE.timestamp() * 1000),
        'interval': 'h1'
    }
    response = requests.get(DATA_URL, params=params)
    data = response.json()
    df = pd.DataFrame(data['data'])
    df['date'] = pd.to_datetime(df['time'], unit='ms')
    df.set_index('date', inplace=True)
    df['priceUsd'] = df['priceUsd'].astype(float)
    df['circulatingSupply'] = df['circulatingSupply'].astype(float)
    df['priceUsd_lag1'] = df['priceUsd'].shift(1)
    df['priceUsd_lag2'] = df['priceUsd'].shift(2)
    df['priceUsd_lag3'] = df['priceUsd'].shift(3)
    df.dropna(inplace=True)
    X = df[['priceUsd_lag1', 'priceUsd_lag2', 'priceUsd_lag3', 'circulatingSupply']]
    return X

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['GET'])
def predict():
    X = fetch_and_preprocess_data()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    latest_data = X_scaled[-1].reshape(1, -1)
    prediction = model.predict(latest_data)
    return jsonify({'predicted_price': prediction[0]})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)
