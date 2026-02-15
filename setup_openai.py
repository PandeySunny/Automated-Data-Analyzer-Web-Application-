#!/usr/bin/env python3
"""
Easy Setup Script for OpenAI API Key
This script helps you set the OPENAI_API_KEY environment variable
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
            ["setx", "OPENAI_API_KEY", key],
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
            f.write(f"OPENAI_API_KEY={key}\n")
        print("✓ Created .env file with your API key")
        print("  (This file is Git-ignored for security)")
        return True
    except Exception as e:
        print(f"✗ Failed to create .env file: {e}")
        return False

def test_api_key(key):
    """Test if the API key is valid"""
    try:
        from openai import OpenAI
        client = OpenAI(api_key=key)
        # Try a simple API call
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Say 'API Key Valid'"}],
            max_tokens=10
        )
        print("✓ API Key is valid and working!")
        print(f"  Response: {response.choices[0].message.content}")
        return True
    except Exception as e:
        print(f"✗ API Key test failed: {e}")
        return False

def main():
    print("=" * 60)
    print("OpenAI API Key Setup")
    print("=" * 60)
    print()
    
    # Check if already set
    existing_key = os.getenv("OPENAI_API_KEY", "")
    if existing_key:
        print(f"✓ OPENAI_API_KEY already set: {existing_key[:20]}...")
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
    print("Get your API key from: https://platform.openai.com/api-keys")
    print()
    
    while True:
        api_key = input("Enter your OpenAI API key: ").strip()
        
        if not api_key:
            print("✗ API key cannot be empty")
            continue
        
        if not api_key.startswith("sk-"):
            print("⚠️  Warning: API keys usually start with 'sk-'")
            confirm = input("Continue anyway? (yes/no): ").lower()
            if confirm != "yes":
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
                print("Please set OPENAI_API_KEY manually on your system")
                print(f"  export OPENAI_API_KEY={api_key}")
        
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
            print("3. See AI-powered insights in the analysis results!")
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
