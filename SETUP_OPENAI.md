# Setting Up OpenAI Integration for AI Insights

The Data Analysis Automation app now includes **AI-powered insights** powered by ChatGPT! This feature analyzes your data and provides intelligent summaries about what your data is telling you.

## Setup Instructions

### 1. Get Your OpenAI API Key

1. Visit: https://platform.openai.com/signup
2. Sign up or log in to your account
3. Go to: https://platform.openai.com/api-keys
4. Click **"Create new secret key"**
5. Copy the key (starts with `sk-...`)
6. **Save it somewhere safe!** (You won't be able to see it again)

### 2. Configure the API Key

#### Option A: Using Environment Variable (Recommended)

**Windows PowerShell:**
```powershell
# Set the environment variable
$env:OPENAI_API_KEY = "sk-your-api-key-here"

# Verify it's set
$env:OPENAI_API_KEY
```

**Windows Command Prompt:**
```cmd
# Set the environment variable
setx OPENAI_API_KEY "sk-your-api-key-here"

# Note: You'll need to restart your terminal or IDE for the change to take effect
```

#### Option B: Using .env File

1. Create a `.env` file in your project root:
```
OPENAI_API_KEY=sk-your-api-key-here
```

2. Install python-dotenv:
```bash
pip install python-dotenv
```

3. The app will automatically load it on startup.

### 3. Test the Integration

1. Restart your Flask app:
```bash
py -3.10 run.py
```

2. Upload a CSV file
3. After analysis, you should see **"AI-Powered Insights"** section with:
   - **Overall Summary** - What the data tells you
   - **Key Findings** - Top 3 insights
   - **Anomaly Insights** - Unusual patterns detected
   - **Segment Insights** - Customer/group breakdowns
   - **Business Recommendations** - Actionable advice

### 4. Cost Information

- OpenAI API calls cost money, but are extremely affordable
- Average analysis costs: **$0.001 - $0.01 per dataset**
- You can set usage limits in your OpenAI account settings
- Visit: https://platform.openai.com/account/usage/overview

### 5. Troubleshooting

**"AI insights unavailable (OpenAI API key not configured)"**
- Your `OPENAI_API_KEY` environment variable is not set
- Check that you've set it correctly and restarted your app

**"Error: Invalid API key"**
- Your API key is incorrect or expired
- Generate a new one from https://platform.openai.com/api-keys

**"Rate limit exceeded"**
- Too many requests in a short time
- Wait a minute and try again

**"Model not found"**
- The API key account may not have access to gpt-3.5-turbo
- Try upgrading your OpenAI account

## Features Powered by AI

✨ **Automatic Data Analysis**
- Generates summaries of your datasets
- Identifies patterns and trends
- Detects anomalies

📊 **Business Intelligence**
- Provides actionable recommendations
- Offers insights on customer segments
- Highlights unusual patterns

🎯 **Smart Insights**
- Explains what your data means
- Suggests next steps
- Contextualizes findings

---

**Questions?** Visit the OpenAI documentation: https://platform.openai.com/docs
