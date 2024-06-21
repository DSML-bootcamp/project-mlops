import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Load the data
def load_data(file_path):
    data = pd.read_csv(file_path, parse_dates=['date'], index_col='date')
    data.sort_index(inplace=True)
    return data

# Preprocess the data
def preprocess_data(data):
    data['priceUsd_lag1'] = data['priceUsd'].shift(1)
    data['priceUsd_lag2'] = data['priceUsd'].shift(2)
    data['priceUsd_lag3'] = data['priceUsd'].shift(3)
    data.dropna(inplace=True)
    X = data[['priceUsd_lag1', 'priceUsd_lag2', 'priceUsd_lag3', 'circulatingSupply']]
    y = data['priceUsd']
    return X, y

# Train the model
def train_model(X_train, y_train, n_estimators=100, random_state=42):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)  # Fit and transform on training data
    model = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state)
    model.fit(X_train_scaled, y_train)
    return model, scaler

if __name__ == "__main__":
    # Load data
    data_file = 'btc_usd_hourly.csv'
    data = load_data(data_file)

    # Preprocess data
    X, y = preprocess_data(data)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    # Train the model and get the fitted scaler
    model, scaler = train_model(X_train, y_train)

    # Save the model and scaler
    joblib.dump(model, 'random_forest_model.pkl')
    joblib.dump(scaler, 'scaler.pkl')

    # Evaluate the model using the trained model and scaler
    X_test_scaled = scaler.transform(X_test)
    y_pred = model.predict(X_test_scaled)

    mse, rmse, mae, r2 = mean_squared_error(y_test, y_pred), mean_squared_error(y_test, y_pred, squared=False), mean_absolute_error(y_test, y_pred), r2_score(y_test, y_pred)

    print(f'Mean Squared Error (MSE): {mse}')
    print(f'Root Mean Squared Error (RMSE): {rmse}')
    print(f'Mean Absolute Error (MAE): {mae}')
    print(f'R-squared (R2): {r2}')
