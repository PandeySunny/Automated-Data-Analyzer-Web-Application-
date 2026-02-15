# DeepSeek API Setup Guide

Your Data Analysis App now uses **DeepSeek** for AI-powered insights instead of OpenAI!

## ✨ Benefits of DeepSeek:
- 🚀 Faster response times
- 💰 More affordable pricing
- 🔓 Open-source friendly
- 🌍 Global availability

---

## 📋 Setup Instructions

### Step 1: Get Your DeepSeek API Key

1. Visit: https://platform.deepseek.com/api-keys
2. Sign up or log in to your DeepSeek account
3. Click **"Create new API key"**
4. Copy the key (starts with `sk-...`)
5. **Save it somewhere safe!**

### Step 2: Configure the API Key

**Option A: Automatic Setup (Recommended)**
```powershell
cd "C:\Users\SUNNY\Desktop\Data analysis automation Projects\Data analysis automation Projects"
py -3.10 setup_deepseek.py
```

Then follow the prompts to configure your API key.

**Option B: Manual Setup**

**Windows PowerShell:**
```powershell
# Set permanently
setx DEEPSEEK_API_KEY "sk-your-key-here"

# Or for current session only
$env:DEEPSEEK_API_KEY = "sk-your-key-here"
```

**Option C: Using .env File**

Create/edit `.env` in your project folder:
```
DEEPSEEK_API_KEY=sk-your-key-here
```

### Step 3: Restart Your App

```powershell
py -3.10 run.py
```

Visit: http://localhost:5000

### Step 4: Test It

1. Upload a CSV file
2. Look for **"AI-Powered Insights"** section
3. You should see:
   - ✅ Overall Summary
   - ✅ Key Findings  
   - ✅ Anomaly Insights
   - ✅ Segment Insights
   - ✅ Business Recommendations

---

## 📊 Pricing

DeepSeek API pricing is extremely affordable:
- **Input**: ~$0.14 per 1M tokens
- **Output**: ~$0.28 per 1M tokens

For typical datasets, expect **$0.001-$0.01 per analysis**.

Check usage: https://platform.deepseek.com/account/usage

---

## 🔧 Troubleshooting

**"AI insights unavailable"**
- ✓ Check DEEPSEEK_API_KEY is set correctly
- ✓ Restart the app
- ✓ Check your API key hasn't expired

**"Error 401 - Unauthorized"**
- ✗ Invalid or expired API key
- ✓ Generate a new one from the dashboard

**"Error 429 - Rate Limited"**
- ⏱️ Too many requests too quickly
- ✓ Wait a moment and retry

**"Connection timeout"**
- 🌐 Network issue or API is down
- ✓ Check your internet connection
- ✓ Try again in a moment

---

## 📚 More Information

- DeepSeek Docs: https://platform.deepseek.com/docs
- API Reference: https://platform.deepseek.com/docs/api-reference
- Status Page: https://status.deepseek.com

---

## 🎯 Features

Your app now provides:
✨ **Automatic Data Analysis** - Instant insights from your CSV files
🤖 **AI Interpretation** - DeepSeek explains what your data means
📊 **Smart Recommendations** - Actionable business advice
🎯 **Anomaly Detection** - Identifies unusual patterns & potential fraud
👥 **Customer Segmentation** - Groups similar customers automatically

---

**Enjoy your AI-powered data analysis! 🚀**
