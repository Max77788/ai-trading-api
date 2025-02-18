import os
from flask import Flask, request, jsonify
import requests
import json

# Import dotenv to load environment variables
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

from functions import get_trading_signals

# --- External libraries for AI and memory ---
import pinecone
# import openrouter
# from openrouter import ChatOpenRouter, ChatMessage
# pydanticai is a hypothetical library that extends Pydantic with AI-related features.
# Its usage may vary depending on your specific needs.
from pydantic_ai import Agent
from openai import OpenAI

app = Flask(__name__)

# Debug mode based on environment
IS_DEBUG = os.environ.get("NODE_ENV") == "development"

# Configuration for external APIs
RANK_API_URL = "https://api.ranktrading.cloud/agent/ask"
OPENROUTER_API_URL = "https://openrouter.ai/api/v1"
# You might want to store this secret in an environment variable for security.
RANK_API_SECRET = os.environ.get("RANK_API_SECRET")
BINANCE_API_URL = "https://api.binance.com/api/v3/ticker/price"

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")

AGENT_MAIN_INSTRUCTIONS="""
You are the AI assistant in AI trading app. Your goal is to help users make wise investment decisions based on their requests and market situations.

You are given the summary of the recommendation regarding the particular asset and the action the user should undertake regarding the asset (sell/hold/buy)

"""

CRYPTO_DETECTION_INSTRUCTION=""" 
Your current task is to define whether the user mentions any specific cryptocurrency.
Return the following json as the response { crypto_mentioned: bool, ticker: string }
"""

if OPENROUTER_API_KEY:
    client = OpenAI(
    base_url=OPENROUTER_API_URL,
    api_key=OPENROUTER_API_KEY,
    )

# ----------------------------------------------------------------------
# Example Pinecone initialization
# ----------------------------------------------------------------------
# Note: Make sure you set PINECONE_API_KEY and PINECONE_ENV in your .env file
pinecone_api_key = os.environ.get("PINECONE_API_KEY", "")
pinecone_environment = os.environ.get("PINECONE_ENV", "")

if pinecone_api_key and pinecone_environment:
    pinecone.init(api_key=pinecone_api_key, environment=pinecone_environment)
    # Create or reference an existing Pinecone index here:
    # example_index = pinecone.Index("my-embeddings-index")

@app.route('/')
def index():
    return "Welcome to the AI Trading Assistant API! Ready to make some market moves?"

@app.route('/daily_signals', methods=['GET'])
def get_daily_signals():
    """
    Fetches a trading signal from the Rank Market Intelligence API.
    """
    try:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"RANK-API-SECRET:{RANK_API_SECRET}"
        }
        response = requests.get(RANK_API_URL, headers=headers, timeout=5)
        response.raise_for_status()
        data = response.json()

        # The example response is a single signal. Wrap it in a list for consistency.
        signal = {
            "asset": data["asset"],
            "action": data["action"],
            "entry_price": data["entryPrice"],
            "profit_price": data["profitPrice"],
            "stop_loss_price": data["stopLossPrice"],
            "duration": data["duration"],
            "strength": data["strength"]
        }

        return jsonify({"status": "success", "signals": [signal]})
    except requests.exceptions.RequestException as re:
        return jsonify({"status": "error", "message": str(re)}), 500
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/chat', methods=['POST'])
def chat():
    """
    Handles chat messages for trading queries.
    Integrates with OpenRouter to generate AI-driven responses.
    """
    try:
        data = request.get_json()
        
        inquiry = data["question"]
        
        # Q example: Should I buy ETH today
        
        completion = client.chat.completions.create(
        model="openai/gpt-4o",
        messages=[
            {"role": "developer", "content": AGENT_MAIN_INSTRUCTIONS+CRYPTO_DETECTION_INSTRUCTION},
            {"role": "user", "content": inquiry}
        ],
        response_format={"type": "json_object"}
        )
        
        response_json = json.loads(completion.choices[0].message.content)
        
        print(f"First Response: {response_json}")
        
        if response_json.get("crypto_mentioned"):
            rank_api_data = get_trading_signals()
        else:
            rank_api_data = get_trading_signals()
            
            print(f"Rank API Data: {rank_api_data}")
            
            prompt_to_pass = " provide the clear response to user based on this assets pair evaluation "+str(rank_api_data)
            
            completion = client.chat.completions.create(
            model="openai/gpt-4o",
            messages=[
                {"role": "developer", "content": AGENT_MAIN_INSTRUCTIONS+prompt_to_pass}
            ]
            )
            
            print(completion)
            
            reply = completion.choices[0].message.content
        
        
        
        return jsonify({"response": reply, "rank_api_response": rank_api_data})

        
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/technical_analysis', methods=['POST'])
def technical_analysis():
    """
    Performs technical analysis on a given asset.
    For now, returns a placeholder analysis.
    """
    try:
        data = request.get_json()
        analysis_request = TechnicalAnalysisRequest(**data)
        symbol = analysis_request.symbol

        # Placeholder: In production, replace this with actual TA logic or pydanticai usage
        # For example, you could use pydanticai's functionality to parse, validate, or even
        # generate AI-based analysis. The library usage would look something like:
        #
        # my_ai_model = pydanticai.AIModel(...)
        # analysis = my_ai_model.run_analysis(symbol)
        #
        # For now, just returning a dummy response.
        analysis = (
            f"Technical analysis for {symbol}: Trend is bullish with a side of awesome!"
        )
        return jsonify({"status": "success", "analysis": analysis})
    except ValidationError as ve:
        return jsonify({"status": "error", "message": ve.errors()}), 400
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=IS_DEBUG)