import os
from dotenv import load_dotenv
import requests

load_dotenv()

api_key = os.getenv("DEEPSEEK_API_KEY")
print(f"API Key: {api_key}")
print(f"Length: {len(api_key) if api_key else 0}")

# Test API call
url = "https://api.deepseek.com/v1/chat/completions"
headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json"
}
payload = {
    "model": "deepseek-chat",
    "messages": [{"role": "user", "content": "Hello"}],
    "max_tokens": 10
}

response = requests.post(url, headers=headers, json=payload)
print(f"Status: {response.status_code}")
print(f"Response: {response.text}")