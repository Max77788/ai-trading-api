import os
import requests
import json

# Import dotenv to load environment variables
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

# Configuration for external APIs
RANK_API_URL = "https://api.ranktrading.cloud/agent/ask"

# You might want to store this secret in an environment variable for security.
RANK_API_SECRET = os.environ.get("RANK_API_SECRET")

url = "https://api.ranktrading.cloud/agent/ask"
headers = {
    "Content-Type": "application/json",
    "Authorization": f"RANK-API-SECRET:{RANK_API_SECRET}"
}

def get_trading_signals():
    response = requests.get(url, headers=headers)

    data = response.json()
    
    return data

def get_market_data(ticker):
    url = f"https://min-api.cryptocompare.com/data/price?fsym={ticker.replace("USDT","")}&tsyms=USDT"
    response = requests.get(url)
    data = response.json()
    
    print(f"Response from Binance API: {data}")
    
    # print(f"Current price of {ticker}: {data['price']}")
    
    return data
