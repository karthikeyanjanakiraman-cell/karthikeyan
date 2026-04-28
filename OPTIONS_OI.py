
import os
import requests
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('Scanner')

def fetch_data_direct(symbol, expiry):
    # Use standard Fyers v3 REST endpoint
    url = "https://api.fyers.in/data-rest/v3/optionchain"
    headers = {
        "Authorization": f"{os.getenv('CLIENT_ID')}:{os.getenv('ACCESS_TOKEN')}",
        "Content-Type": "application/json"
    }
    # Ensure proper ticker format for Fyers: e.g., "NSE:RELIANCE-EQ"
    s = symbol.strip().upper().replace('NSE:', '').replace('-EQ', '')
    payload = {
        "symbol": f"NSE:{s}-EQ",
        "strikecount": 50,
        "expiry": expiry
    }

    try:
        response = requests.post(url, json=payload, headers=headers)
        return response.json()
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    # Test for a common symbol to verify connectivity
    res = fetch_data_direct("RELIANCE", os.getenv('ACTIVE_EXPIRY', '2026-04-28'))
    print(json.dumps(res, indent=2))
