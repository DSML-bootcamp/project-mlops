import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import requests
from datetime import datetime, timedelta
from flask import Flask, request, jsonify, render_template
import joblib

app = Flask(__name__)

# Set up the URL and parameters
BASE_URL = 'https://api.coincap.io/v2/assets/bitcoin/history'
END_DATE = datetime.utcnow()
START_DATE = END_DATE - timedelta(days=15)
END_TIMESTAMP = int(END_DATE.timestamp() * 1000)
START_TIMESTAMP = int(START_DATE.timestamp() * 1000)
DATA_URL = f"{BASE_URL}?interval=h1&start={START_TIMESTAMP}&end={END_TIMESTAMP}"

# Define constants
MODEL_FILE = "random_forest_model.pkl"
SCALER_FILE = "scaler.pkl"

# Load initial model and scaler
model = joblib.load(MODEL_FILE)
scaler = joblib.load(SCALER_FILE)

# Load and preprocess data
def load_data(url):
    response = requests.get(url)
    data = response.json()['data']
    df = pd.DataFrame(data)
    df['date'] = pd.to_datetime(df['time'], unit='ms')
    df.set_index('date', inplace=True)
    df.sort_index(inplace=True)
    df.rename(columns={'priceUsd': 'priceUsd', 'circulatingSupply': 'circulatingSupply'}, inplace=True)
    return df[['priceUsd', 'circulatingSupply']]

def preprocess_data(data):
    data['priceUsd_lag1'] = data['priceUsd'].shift(1)
    data['priceUsd_lag2'] = data['priceUsd'].shift(2)
    data['priceUsd_lag3'] = data['priceUsd'].shift(3)
    data.dropna(inplace=True)
    X = data[['priceUsd_lag1', 'priceUsd_lag2', 'priceUsd_lag3', 'circulatingSupply']]
    y = data['priceUsd']
    return X, y

def train_model(X_train, y_train, n_estimators=100, random_state=42):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    model = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state)
    model.fit(X_train_scaled, y_train)
    return model, scaler

def update_model():
    global model, scaler
    data = load_data(DATA_URL)
    X, y = preprocess_data(data)
    model, scaler = train_model(X, y)
    joblib.dump(model, MODEL_FILE)
    joblib.dump(scaler, SCALER_FILE)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Update model with the latest data
    update_model()

    # Prepare the latest input for prediction
    latest_data = load_data(DATA_URL).iloc[-1:]
    X_latest = latest_data[['priceUsd_lag1', 'priceUsd_lag2', 'priceUsd_lag3', 'circulatingSupply']]
    X_latest_scaled = scaler.transform(X_latest)
    prediction = model.predict(X_latest_scaled)

    return jsonify({'prediction': prediction[0]})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
