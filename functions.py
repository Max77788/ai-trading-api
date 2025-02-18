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
    "Authorization": "RANK-API-SECRET:vtXb58EZAX24NYL1-T4Frb6p3Td349GFS-4ZfYEjfyi9tslF5C"
}

def get_trading_signals():
    response = requests.get(url, headers=headers)

    data = response.json()
    
    return data
