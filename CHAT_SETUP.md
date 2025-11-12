# OpenAI Chat Integration Setup Guide

## ✅ What's Been Implemented

### Backend (`chat_service.py`)
- **ChatService class** with LangChain + GPT-4 integration
- Queries Firestore `image_classifications` collection for context
- Maintains conversation history per session
- Provides statistics, classification breakdowns, and sample tile data to GPT

### API Endpoints (`app.py`)
- **POST `/api/chat`** - Send messages, get GPT-4 responses with Firestore context
- **POST `/api/chat/clear`** - Clear conversation history for new session

### Frontend Updates
- Chat interface now calls `/api/chat` API endpoint
- Added "New Conversation" button to reset chat history
- Shows typing indicator while waiting for GPT response
- Better error handling with informative messages

## 🔧 Setup Required

### 1. Get OpenAI API Key

1. Go to https://platform.openai.com/api-keys
2. Sign in or create account
3. Click "Create new secret key"
4. Copy the key (starts with `sk-...`)

### 2. Set Environment Variable

**Windows PowerShell:**
```powershell
$env:OPENAI_API_KEY='sk-your-key-here'
```

**For permanent setup, add to system environment variables:**
1. Search "Environment Variables" in Windows
2. Add new variable: `OPENAI_API_KEY` = `sk-your-key-here`

**Linux/Mac:**
```bash
export OPENAI_API_KEY='sk-your-key-here'
```

### 3. Verify Installation

Run the test script:
```bash
python test_chat_service.py
```

Should show:
```
✓ langchain installed
✓ langchain_openai installed  
✓ openai installed
✓ OPENAI_API_KEY is set
✓ chat_service module imported successfully
```

## 🚀 Usage

### Start the Flask App
```bash
python app.py
```

### Ask Questions in Chat

The chat can now answer questions about your LADI classification data:

**Statistics Questions:**
- "How many tiles show flooding?"
- "What's the most common classification?"
- "How many tiles have buildings_affected?"

**Analysis Questions:**
- "Summarize the damage in this dataset"
- "What percentage of tiles show debris?"
- "Compare flooding vs buildings classifications"

**Data Exploration:**
- "What classification types are available?"
- "Which labels have the highest confidence scores?"
- "Describe the spatial distribution of damage"

### How It Works

1. User asks question in chat
2. Frontend sends to `/api/chat`
3. Backend queries Firestore for all LADI classifications
4. Builds context with:
   - Total tile count
   - Classification breakdown by label
   - Confidence score ranges
   - Sample tile locations
5. Sends to GPT-4 with context
6. GPT analyzes and responds
7. Response shown in chat

## 💰 Cost Information

- **Model**: GPT-4 (standard, not vision)
- **Cost**: ~$0.03 per 1K input tokens, ~$0.06 per 1K output tokens
- **Typical query**: $0.001 - $0.01 per question
- With 100 tiles in context: ~$0.002 per question

## 🔒 Security Notes

- Never commit your API key to git
- Add to `.gitignore`: `.env`, `config.py` with keys
- Use environment variables for production
- Consider rate limiting for production deployment

## 📝 Example Conversation

```
User: How many tiles show flooding?

Bot: Based on the current data, there are 47 tiles classified 
with flooding_any, which represents 31% of the total 150 tiles. 
Additionally, 23 tiles show flooding_structures specifically, 
indicating flooding that affects buildings.

User: What areas have the most damage?

Bot: Looking at the spatial distribution, the highest concentration 
of damage classifications is in the northern section of the mapped 
area. This region shows:
- 18 tiles with buildings_affected_or_greater
- 15 tiles with debris_any
- 12 tiles with flooding_structures

The southern area shows less severe damage with mostly 
buildings_minor classifications.
```

## 🐛 Troubleshooting

**Error: "OPENAI_API_KEY environment variable not set"**
- Set the environment variable (see Setup step 2)
- Restart terminal/VS Code after setting

**Error: "Failed to process chat message"**
- Check Firestore is initialized (firebase-credentials.json exists)
- Verify image_classifications collection has data
- Check Flask app logs for detailed error

**Chat not responding:**
- Open browser console (F12) to check for errors
- Verify `/api/chat` endpoint is accessible
- Check network tab for 500 errors

## 🎯 Next Steps (Optional Enhancements)

1. **Add Vision Support**: For "show me what flooding looks like" queries
2. **Viewport Filtering**: Only analyze tiles in current map view
3. **Export Conversations**: Download chat history as PDF/text
4. **Suggested Questions**: Show relevant questions based on data
5. **Usage Tracking**: Monitor API costs and set budgets
