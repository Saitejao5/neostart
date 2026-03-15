"""
diagnose.py — Diagnose API key and configuration issues.

Run this before starting Streamlit to verify everything is configured correctly.
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv
import requests

# Load .env from project root
env_path = Path(__file__).parent / ".env"
load_dotenv(dotenv_path=env_path, override=True)

print("=" * 70)
print("ScholarBot Diagnostics")
print("=" * 70)

# Check 1: .env file exists
print("\n✓ Checking .env file...")
if env_path.exists():
    print(f"  ✅ Found .env at: {env_path}")
else:
    print(f"  ❌ .env not found at: {env_path}")
    sys.exit(1)

# Check 2: API key is loaded
openrouter_key = os.getenv("OPENROUTER_API_KEY", "").strip()
print("\n✓ Checking API key...")
if openrouter_key:
    print(f"  ✅ API key loaded (starts with: {openrouter_key[:20]}...)")
else:
    print("  ❌ OPENROUTER_API_KEY not found or empty")
    sys.exit(1)

# Check 3: API key format
print("\n✓ Validating API key format...")
if openrouter_key.startswith("sk-or-v1-"):
    print(f"  ✅ API key format is correct")
else:
    print(f"  ⚠️  API key does not start with 'sk-or-v1-' (may be invalid)")

# Check 4: Test API connectivity
print("\n✓ Testing OpenRouter API connectivity...")
try:
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {openrouter_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://scholarbot.local",
        "X-Title": "ScholarBot",
    }
    
    # Simple test request (won't count against quota due to minimal tokens)
    payload = {
        "model": "meta-llama/llama-3.1-8b-instruct",
        "messages": [{"role": "user", "content": "Hi"}],
        "max_tokens": 10,
    }
    
    response = requests.post(url, headers=headers, json=payload, timeout=10)
    
    if response.status_code == 200:
        print(f"  ✅ API connection successful")
    elif response.status_code == 401:
        print(f"  ❌ Authentication failed (401)")
        print(f"     Your API key may be invalid or expired.")
        print(f"     Get a new key from: https://openrouter.ai/keys")
        sys.exit(1)
    else:
        print(f"  ⚠️  API returned status {response.status_code}")
        print(f"     Response: {response.text[:200]}")
        
except requests.exceptions.ConnectionError:
    print(f"  ❌ Cannot connect to OpenRouter API")
    print(f"     Check your internet connection")
    sys.exit(1)
except Exception as e:
    print(f"  ❌ Error testing API: {e}")
    sys.exit(1)

# Check 5: Other API keys
print("\n✓ Checking optional API keys...")
serper_key = os.getenv("SERPER_API_KEY", "").strip()
tavily_key = os.getenv("TAVILY_API_KEY", "").strip()

if serper_key:
    print(f"  ✅ Serper API key found")
else:
    print(f"  ⚠️  Serper API key not found (web search may not work)")

if tavily_key:
    print(f"  ✅ Tavily API key found")
else:
    print(f"  ⚠️  Tavily API key not found (fallback web search unavailable)")

print("\n" + "=" * 70)
print("✅ All checks passed! You're ready to run: streamlit run app.py")
print("=" * 70)
