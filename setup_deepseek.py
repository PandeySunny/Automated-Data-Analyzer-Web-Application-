#!/usr/bin/env python3
"""
Setup script for DeepSeek API Key
This script helps you set the DEEPSEEK_API_KEY environment variable
"""

import os
import sys
import subprocess
import platform

def set_env_variable_windows(key):
    """Set environment variable on Windows"""
    try:
        # Use setx to set permanently
        subprocess.run(
            ["setx", "DEEPSEEK_API_KEY", key],
            check=True,
            capture_output=True
        )
        print("✓ Environment variable set successfully!")
        print("⚠️  Please close and reopen your terminal/IDE for changes to take effect.")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Failed to set environment variable: {e}")
        return False
    except Exception as e:
        print(f"✗ Error: {e}")
        return False

def create_env_file(key):
    """Create a .env file with the API key"""
    try:
        with open(".env", "w") as f:
            f.write(f"DEEPSEEK_API_KEY={key}\n")
        print("✓ Created .env file with your API key")
        print("  (This file is Git-ignored for security)")
        return True
    except Exception as e:
        print(f"✗ Failed to create .env file: {e}")
        return False

def test_api_key(key):
    """Test if the API key is valid"""
    try:
        import httpx
        headers = {
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": "deepseek-chat",
            "messages": [{"role": "user", "content": "Say 'API Key Valid'"}],
            "max_tokens": 10
        }
        
        response = httpx.post(
            "https://api.deepseek.com/chat/completions",
            headers=headers,
            json=payload,
            timeout=10.0
        )
        
        if response.status_code == 200:
            print("✓ API Key is valid and working!")
            data = response.json()
            print(f"  Response: {data['choices'][0]['message']['content']}")
            return True
        else:
            print(f"✗ API Error {response.status_code}: {response.text[:100]}")
            return False
    except Exception as e:
        print(f"✗ API Key test failed: {e}")
        return False

def main():
    print("=" * 60)
    print("DeepSeek API Key Setup")
    print("=" * 60)
    print()
    
    # Check if already set
    existing_key = os.getenv("DEEPSEEK_API_KEY", "")
    if existing_key:
        print(f"✓ DEEPSEEK_API_KEY already set: {existing_key[:20]}...")
        test = input("\nWould you like to test or change it? (test/change/no): ").lower()
        if test == "test":
            test_api_key(existing_key)
            return
        elif test != "change":
            print("Skipping setup.")
            return
        else:
            print("Entering new key...")
    
    print()
    print("Get your API key from: https://platform.deepseek.com/api-keys")
    print()
    
    while True:
        api_key = input("Enter your DeepSeek API key: ").strip()
        
        if not api_key:
            print("✗ API key cannot be empty")
            continue
        
        # Let user choose setup method
        print()
        print("Setup Methods:")
        print("1. Environment Variable (Permanent, requires terminal restart)")
        print("2. .env File (Simple, local to this project)")
        print("3. Both (Recommended)")
        
        choice = input("Choose setup method (1-3): ").strip()
        
        if choice in ["1", "3"]:
            if platform.system() == "Windows":
                if set_env_variable_windows(api_key):
                    print()
            else:
                print("Please set DEEPSEEK_API_KEY manually on your system")
                print(f"  export DEEPSEEK_API_KEY={api_key}")
        
        if choice in ["2", "3"]:
            if create_env_file(api_key):
                print()
        
        # Test the key
        print("\nTesting API key...")
        if test_api_key(api_key):
            print()
            print("=" * 60)
            print("✓ Setup Complete!")
            print("=" * 60)
            print()
            print("Next steps:")
            print("1. Restart your Flask app")
            print("2. Upload a CSV file")
            print("3. See DeepSeek AI insights in the analysis results!")
            break
        else:
            retry = input("\nRetry with different key? (yes/no): ").lower()
            if retry != "yes":
                print("Setup cancelled.")
                break

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nSetup cancelled.")
        sys.exit(0)
