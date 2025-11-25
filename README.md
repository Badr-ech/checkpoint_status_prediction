# 🚧 Checkpoint Status Prediction System

A machine learning system that predicts Palestinian checkpoint status (open/closed) using social media feeds (Telegram, Reddit) and historical traffic patterns. Provides both short-term (1-3h) and long-term (12-24h) predictions.

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104-green.svg)](https://fastapi.tiangolo.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 🎯 Features

### Dual-Horizon Predictions
- **Short-term (1-3 hours)**: Real-time predictions based on recent social media activity, time patterns, and current sentiment
- **Long-term (12-24 hours)**: Advance forecasting using historical patterns, day-of-week trends, and political events

### Multi-Source Data Collection
- **Telegram**: Monitors Palestinian news channels and community groups for real-time reports
- **Reddit**: Analyzes discussions from r/Palestine, r/Palestinians, and related subreddits
- **Manual Reports**: API endpoint for user-submitted status updates

### Advanced NLP & Analysis
- **Arabic Sentiment Analysis**: Using multilingual BERT models optimized for Arabic
- **Status Inference**: Keyword-based and ML-based status extraction from text
- **Feature Engineering**: 50+ features including temporal, social media, and historical patterns
- **Confidence Scoring**: Probabilistic predictions with confidence levels

### Interactive Dashboard
- **Google Maps Integration**: Visual map showing all checkpoints
- **Real-time Updates**: Auto-refresh every 5 minutes
- **Dual Predictions Display**: Side-by-side short-term and long-term forecasts
- **Historical Data**: View past status records and social media mentions

### REST API
- Full-featured REST API with Swagger documentation
- Endpoints for predictions, historical data, social media mentions
- Real-time status reporting

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│                   Data Collection Layer                  │
├──────────────────┬──────────────────┬───────────────────┤
│ Telegram Monitor │ Reddit Scraper   │ Manual Reports    │
└────────┬─────────┴────────┬─────────┴──────────┬────────┘
         └──────────────────┼─────────────────────┘
                            ▼
                 ┌──────────────────────┐
                 │   SQLite Database    │
                 └──────────┬───────────┘
         ┌──────────────────┼──────────────────┐
         ▼                  ▼                  ▼
┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│ NLP Pipeline │  │   Feature    │  │   ML Models  │
│              │  │ Engineering  │  │ RandomForest │
└──────┬───────┘  └──────┬───────┘  └──────┬───────┘
       └─────────────────┼──────────────────┘
                         ▼
              ┌────────────────────┐
              │   FastAPI Backend  │
              └─────────┬──────────┘
                        ▼
              ┌────────────────────┐
              │  Web Dashboard     │
              └────────────────────┘
```

## 📁 Project Structure

```
checkpoint_status_prediction/
├── src/
│   ├── api/                      # FastAPI endpoints
│   │   └── main.py               # REST API implementation
│   ├── collectors/               # Data collection modules
│   │   ├── telegram_collector.py # Telegram monitoring
│   │   ├── reddit_collector.py   # Reddit scraping
│   │   └── init_checkpoints.py   # Checkpoint initialization
│   ├── models/                   # ML models
│   │   ├── predictor.py          # Dual-horizon predictor
│   │   └── train.py              # Training pipeline
│   ├── nlp/                      # NLP pipeline
│   │   ├── sentiment_analyzer.py # Sentiment analysis
│   │   └── feature_extractor.py  # Feature engineering
│   ├── database/                 # Database layer
│   │   ├── models.py             # SQLAlchemy models
│   │   └── database.py           # Connection management
│   └── utils/                    # Utilities
│       └── logger.py             # Logging setup
├── static/                       # Frontend assets
│   └── index.html                # Dashboard UI
├── data/                         # Database storage
├── models/                       # Trained ML models
├── logs/                         # Application logs
├── requirements.txt              # Python dependencies
├── .env.example                  # Environment template
├── Dockerfile                    # Docker configuration
├── docker-compose.yml            # Multi-container setup
├── setup.bat                     # Windows setup script
├── run.bat                       # Quick start script
├── QUICKSTART.md                 # Quick start guide
└── README.md                     # This file
```

## 🚀 Quick Start

### Option 1: Automated Setup (Recommended for Windows)

```bash
# Run the setup script
setup.bat

# Follow the prompts to configure API keys in .env
# Then start data collection for 7+ days
```

### Option 2: Manual Setup

#### 1. Prerequisites
- Python 3.11+
- Telegram account (for API access)
- Reddit account (for API access)
- Google Maps API key

#### 2. Installation

```bash
# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

#### 3. Configuration

Copy `.env.example` to `.env` and configure:

**Telegram API:**
1. Visit https://my.telegram.org
2. Create app → Get API ID and Hash
3. Add to `.env`

**Reddit API:**
1. Visit https://www.reddit.com/prefs/apps
2. Create script app → Get Client ID and Secret
3. Add to `.env`

**Google Maps API:**
1. Enable Maps JavaScript API at console.cloud.google.com
2. Create API key
3. Add to `.env` and update `static/index.html` line 6

#### 4. Initialize System

```bash
# Initialize database
python -m src.database.init_db

# Add checkpoints (8 major West Bank checkpoints)
python -m src.collectors.init_checkpoints --init
```

#### 5. Collect Data (7+ days recommended)

Run in separate terminals:

```bash
# Terminal 1: Telegram collector
python -m src.collectors.telegram_collector

# Terminal 2: Reddit collector
python -m src.collectors.reddit_collector
```

#### 6. Train Models

After collecting sufficient data:

```bash
python -m src.models.train
```

#### 7. Start API Server

```bash
# Quick start
run.bat

# Or manually
python -m src.api.main
```

#### 8. Access Dashboard

- **Dashboard**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs

## 📡 API Endpoints

### Checkpoints
- `GET /api/checkpoints` - List all checkpoints
- `GET /api/checkpoints/{id}` - Get checkpoint details
- `GET /api/checkpoints/{id}/predict` - Get dual-horizon predictions
- `GET /api/checkpoints/{id}/history` - Historical status records
- `GET /api/checkpoints/{id}/social-media` - Social media mentions
- `POST /api/checkpoints/{id}/status` - Report manual status

### System
- `GET /api/status` - System status and statistics
- `GET /api/predictions/recent` - Recent predictions
- `GET /health` - Health check

### Interactive Documentation
Visit `/docs` for full Swagger/OpenAPI documentation

## 🐳 Docker Deployment

```bash
# Build and start all services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

Services included:
- **app**: FastAPI server (port 8000)
- **db**: PostgreSQL database (port 5432)
- **telegram_collector**: Continuous Telegram monitoring
- **reddit_collector**: Continuous Reddit monitoring

## 🎓 How It Works

### Data Collection
1. **Telegram**: Monitors configured channels for checkpoint mentions
2. **Reddit**: Scrapes posts/comments from Palestine-related subreddits
3. **Processing**: Each mention is analyzed for:
   - Checkpoint identification (multi-language keyword matching)
   - Status inference (closed/open/partial/unknown)
   - Sentiment analysis (Arabic and English)
   - Confidence scoring

### Feature Engineering
Extracts 50+ features per checkpoint:
- **Temporal**: Hour, day of week, holidays, peak times
- **Social Media**: Mention count, sentiment scores, source diversity
- **Historical**: Closure rates, day-specific patterns, recent trends
- **Checkpoint**: Type, location, region

### Machine Learning
- **Models**: Random Forest classifiers (easily interpretable)
- **Training**: Minimum 7 days of data, 100+ samples
- **Separate Models**: Independent short-term and long-term models
- **Features**: Different feature weights for each time horizon
- **Output**: Status prediction + confidence score (0-1)

### Prediction Process
1. Extract current features for checkpoint
2. Run through both models (short-term & long-term)
3. Generate probability distributions
4. Return predictions with confidence scores
5. Store predictions for future evaluation

## 🌍 Included Checkpoints

8 major West Bank checkpoints pre-configured:

1. **Qalandiya** - Jerusalem/Ramallah (busiest)
2. **Bethlehem 300** - Bethlehem/Jerusalem
3. **Huwwara** - South of Nablus
4. **Jaba** - Northeast Jerusalem
5. **Container** - Bethlehem/Hebron
6. **Tunnels** - Bethlehem north entrance
7. **Za'tara** - Nablus/Ramallah, Route 60
8. **Beit El** - North Ramallah

## 🔧 Configuration

Key environment variables in `.env`:

```bash
# Telegram
TELEGRAM_API_ID=your_api_id
TELEGRAM_API_HASH=your_hash
TELEGRAM_CHANNELS=@channel1,@channel2

# Reddit
REDDIT_CLIENT_ID=your_client_id
REDDIT_CLIENT_SECRET=your_secret
REDDIT_SUBREDDITS=Palestine,Palestinians

# Google Maps
GOOGLE_MAPS_API_KEY=your_api_key

# Database
DATABASE_URL=sqlite:///./checkpoint_data.db

# ML Settings
MIN_DATA_DAYS_FOR_TRAINING=7
MODEL_RETRAIN_INTERVAL_HOURS=24
```

## 📊 Performance Metrics

Expected model performance (with 30 days of data):
- **Accuracy**: 75-85% (depends on data quality)
- **Short-term**: Higher accuracy (more recent signals)
- **Long-term**: Lower accuracy (more uncertainty)
- **Confidence calibration**: Scores reflect actual accuracy

Feature importance (typical):
1. Recent social media mentions (1h, 3h)
2. Historical closure rate for same time
3. Day of week patterns
4. Sentiment scores
5. Holiday proximity

## 🛠️ Development

```bash
# Run tests
pytest

# Code formatting
black src/

# Type checking
mypy src/

# Lint
flake8 src/
```

## 📝 Data Schema

### Key Tables
- **checkpoints**: Checkpoint locations and metadata
- **social_media_posts**: Collected posts with analysis
- **checkpoint_status_history**: Ground truth status records
- **predictions**: Model predictions with timestamps
- **training_jobs**: Model training logs and metrics

## 🔍 Troubleshooting

### Common Issues

**"No trained models found"**
- Solution: Collect data for 7+ days, then run `python -m src.models.train`

**"Insufficient data for training"**
- Solution: Ensure collectors are running, wait for more data

**"Telegram authentication failed"**
- Solution: Check API credentials, use international phone format (+972...)

**"Database locked"**
- Solution: Stop all collectors, restart one at a time

**Low prediction accuracy**
- Solution: Collect more data, check social media channels are active

## 🚧 Limitations

- **Data dependency**: Requires active social media discussion
- **Language bias**: Best for Arabic/English content
- **Real-time lag**: Predictions based on recent but not instant data
- **Coverage**: Limited to configured checkpoints
- **Political events**: Sudden security situations may not be predictable

## 🔮 Future Enhancements

- [ ] Integration with Waze traffic data
- [ ] WhatsApp/Signal channel monitoring
- [ ] Israeli news source integration
- [ ] Mobile app (React Native)
- [ ] SMS/Telegram bot notifications
- [ ] Deep learning models (LSTM/Transformer)
- [ ] Multi-language support (Hebrew)
- [ ] Checkpoint crowdsourcing platform

## 📄 License

MIT License - see LICENSE file

## ⚠️ Disclaimer

**This tool is for informational purposes only.**

- Predictions are probabilistic and may be incorrect
- Always verify status through official sources before traveling
- Not affiliated with any government or official organization
- Use at your own risk
- Political situations can change rapidly

## 🤝 Contributing

Contributions welcome! Areas of interest:
- Additional data sources
- Improved NLP models
- Better feature engineering
- Mobile app development
- Deployment optimization

## 📧 Contact

For questions, issues, or collaboration:
- Open an issue on GitHub
- Check documentation at `/docs` endpoint
- Review logs in `logs/` directory

## 🙏 Acknowledgments

Built using:
- FastAPI, SQLAlchemy, scikit-learn
- Transformers (Hugging Face)
- Telethon, PRAW
- Google Maps JavaScript API

Data sources:
- UN OCHA Humanitarian Data Exchange
- OpenStreetMap
- Telegram public channels
- Reddit communities

---

**Made with ❤️ for the Palestinian people**
