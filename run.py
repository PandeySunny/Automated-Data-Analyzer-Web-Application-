#!/usr/bin/env python
"""
Simple runner for the Flask app with error handling
"""
import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(__file__))

# Ensure OpenAI is available
try:
    import openai
except ImportError:
    print("Installing openai package...")
    os.system(f"{sys.executable} -m pip install openai python-dotenv")

try:
    from app import app
    print("✓ Flask app loaded successfully")
    print("=" * 60)
    print("Starting Flask development server...")
    print("=" * 60)
    print("Visit: http://localhost:5000")
    print("=" * 60)
    app.run(debug=False, host="0.0.0.0", port=5000, threaded=True)
except ImportError as e:
    print(f"❌ Import Error: {e}")
    print("\nMake sure all dependencies are installed:")
    print(f"  {sys.executable} -m pip install -r requirements.txt")
    sys.exit(1)
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

