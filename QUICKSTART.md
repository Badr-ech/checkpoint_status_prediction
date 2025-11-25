# Checkpoint Status Prediction System - Quick Start Guide

## 🚀 Quick Start

### 1. Initial Setup (5 minutes)

Run the setup script:
```bash
setup.bat
```

This will:
- Create virtual environment
- Install dependencies
- Initialize database
- Add 8 major checkpoints

### 2. Configure API Keys

Edit `.env` file and add your credentials:

**Telegram API** (Required for social media monitoring):
1. Visit https://my.telegram.org
2. Login with your phone number
3. Go to "API Development Tools"
4. Create a new application
5. Copy `api_id` and `api_hash` to `.env`

**Reddit API** (Optional but recommended):
1. Visit https://www.reddit.com/prefs/apps
2. Click "create another app"
3. Choose "script" type
4. Copy `client_id` and `client_secret` to `.env`

**Google Maps API** (Required for dashboard):
1. Visit https://console.cloud.google.com
2. Enable "Maps JavaScript API"
3. Create credentials (API key)
4. Copy to `.env` and update `static/index.html`

### 3. Collect Data (7+ days recommended)

Start data collectors in separate terminals:

**Terminal 1 - Telegram:**
```bash
venv\Scripts\activate
python -m src.collectors.telegram_collector
```

**Terminal 2 - Reddit:**
```bash
venv\Scripts\activate
python -m src.collectors.reddit_collector
```

These will run continuously collecting data. Minimum 7 days recommended for training.

### 4. Train Models

After collecting sufficient data:
```bash
python -m src.models.train
```

### 5. Start API Server

```bash
run.bat
```

Or manually:
```bash
python -m src.api.main
```

### 6. Access Dashboard

Open browser to: **http://localhost:8000**

API documentation: **http://localhost:8000/docs**

---

## 📊 System Architecture

```
┌─────────────────────────────────────────────────────────┐
│                   Data Collection Layer                  │
├──────────────────┬──────────────────┬───────────────────┤
│ Telegram Monitor │ Reddit Scraper   │ Manual Reports    │
│  (telethon)      │  (praw)          │  (API endpoint)   │
└────────┬─────────┴────────┬─────────┴──────────┬────────┘
         │                  │                     │
         └──────────────────┼─────────────────────┘
                            ▼
                 ┌──────────────────────┐
                 │   SQLite Database    │
                 │  - Checkpoints       │
                 │  - Social Media      │
                 │  - Status History    │
                 │  - Predictions       │
                 └──────────┬───────────┘
                            │
         ┌──────────────────┼──────────────────┐
         │                  │                  │
         ▼                  ▼                  ▼
┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│ NLP Pipeline │  │   Feature    │  │   ML Models  │
│ - Sentiment  │  │ Engineering  │  │ - Short-term │
│ - Status     │  │ - Temporal   │  │ - Long-term  │
│ - Language   │  │ - Social     │  │ (RandomForest)│
└──────┬───────┘  └──────┬───────┘  └──────┬───────┘
       │                 │                  │
       └─────────────────┼──────────────────┘
                         ▼
              ┌────────────────────┐
              │   FastAPI Backend  │
              │  - REST Endpoints  │
              │  - Predictions     │
              └─────────┬──────────┘
                        │
                        ▼
              ┌────────────────────┐
              │  Web Dashboard     │
              │  - Google Maps     │
              │  - Real-time View  │
              └────────────────────┘
```

---

## 🎯 Key Features

### Dual-Horizon Predictions
- **Short-term (1-3 hours)**: Based on recent social media activity
- **Long-term (12-24 hours)**: Based on historical patterns

### Multi-Source Data Collection
- **Telegram**: Real-time reports from Palestinian channels
- **Reddit**: Community discussions and reports
- **Manual**: User-submitted status updates

### Smart Analysis
- **Arabic NLP**: Sentiment analysis for Arabic content
- **Pattern Recognition**: Historical closure patterns
- **Confidence Scores**: Probability estimates for predictions

---

## 📁 Project Structure

```
checkpoint_status_prediction/
├── src/
│   ├── api/                  # FastAPI backend
│   │   └── main.py           # API endpoints
│   ├── collectors/           # Data collection
│   │   ├── telegram_collector.py
│   │   ├── reddit_collector.py
│   │   └── init_checkpoints.py
│   ├── database/             # Database models
│   │   ├── models.py         # SQLAlchemy models
│   │   └── database.py       # Connection management
│   ├── models/               # ML models
│   │   ├── predictor.py      # Dual-horizon predictor
│   │   └── train.py          # Training script
│   ├── nlp/                  # NLP pipeline
│   │   ├── sentiment_analyzer.py
│   │   └── feature_extractor.py
│   └── utils/                # Utilities
│       └── logger.py
├── static/                   # Frontend
│   └── index.html            # Dashboard
├── data/                     # Database storage
├── models/                   # Trained models
├── logs/                     # Application logs
├── requirements.txt          # Python dependencies
├── .env                      # Configuration
├── setup.bat                 # Setup script
└── run.bat                   # Quick start script
```

---

## 🔧 Troubleshooting

### "No trained models found"
- Run data collectors for at least 7 days
- Then run `python -m src.models.train`

### "Telegram authentication failed"
- Check API credentials in `.env`
- Ensure phone number is in international format (+972...)

### "Database locked" error
- Stop all running collectors
- Restart one at a time

### "Insufficient data for training"
- Need minimum 100 status records
- Run collectors for longer period

---

## 🌍 Checkpoints Included

1. **Qalandiya** - Jerusalem/Ramallah
2. **Bethlehem 300** - Bethlehem/Jerusalem
3. **Huwwara** - South of Nablus
4. **Jaba** - Northeast Jerusalem
5. **Container** - Bethlehem/Hebron
6. **Tunnels** - Bethlehem north
7. **Za'tara** - Nablus/Ramallah
8. **Beit El** - North Ramallah

---

## 📞 Support

For issues or questions:
1. Check logs in `logs/` directory
2. Review API docs at `/docs` endpoint
3. Check database with: `python -m src.collectors.init_checkpoints --list`

---

## ⚠️ Disclaimer

This tool is for informational purposes only. Always verify checkpoint status through official sources before traveling. Predictions are based on historical patterns and social media analysis, which may not reflect sudden changes due to security situations or other factors.
