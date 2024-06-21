MLops Project

**Step 1: Connect to an API to Retrieve Cryptocurrency Information**
Objective
Connect to an API to fetch historical BTC/USD price data on an hourly basis.

## Steps
Choose an API provider: We have several options listed: CryptoCompare, Binance, CoinCap, and CoinPaprika. For this project, we will use CryptoCompare for simplicity and ease of access.

Fetch Historical BTC/USD Data: We will write a script to fetch historical BTC/USD data on an hourly basis from the CryptoCompare API.

Store the Data: We will store the fetched data in a structured format (e.g., a CSV file).

## Implementation ##
**Step 1.1: Set Up Your Environment**
First, let's set up a virtual environment for the project and install the necessary packages.

Create a virtual environment:
 python -m venv btc-predictor-env

Activate the virtual environment:
 source btc-predictor-env/Scripts/activate

Install necessary packages:
 pip install requests pandas


**Step 1.2: Fetch Historical BTC/USD Data**
We will write a Python script to fetch the data from CryptoCompare API.

## Sign up for an API key

## Write the script:
Create a file named fetch_btc_data.py.

## Write a script to fetch historical BTC/USD price data on an hourly basis using the CryptoCompare API

## Step 1.3: Run the Script
Run the script in bash:
 python fetch_btc_data.py

## Verify the data: 
Open btc_usd_hourly.csv to ensure the data has been correctly fetched and saved.


**Step 2: Build a Model to Predict BTC/USD Prices on an Hourly Basis** 
Objective
Build a machine learning model to predict BTC/USD prices using historical data. The model should be able to handle the high volatility and noise typical of cryptocurrency markets.

## Load and Preprocess the Data: Load the CSV file, handle missing values, and create relevant features.
## Split the Data: Split the data into training and testing sets.
## Train a Machine Learning Model: Choose a suitable model and train it on the historical data.
## Evaluate the Model: Evaluate the model's performance on the test set.

**Implementation**
**Step 2.1: Load and Preprocess the Data**

## Load the CSV file:
Create a file named predict_btc_price.py.

## Run the script in bash:
 python predict_btc_price.py


**Step 3: Deploying the Model Using MLOps Concepts on AWS**
