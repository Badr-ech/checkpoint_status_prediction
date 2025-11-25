"""
FastAPI backend for checkpoint status prediction system
"""
from fastapi import FastAPI, Depends, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from sqlalchemy.orm import Session
from datetime import datetime, timedelta
from typing import List, Optional
from pathlib import Path
import os

from database import (
    get_db, Checkpoint, CheckpointStatusHistory,
    SocialMediaPost, Prediction, CheckpointStatus
)
try:
    from models.predictor import CheckpointPredictor
except ImportError:
    from src.models.predictor import CheckpointPredictor
from utils.logger import setup_logger

logger = setup_logger("api")

app = FastAPI(
    title="Checkpoint Status Prediction API",
    description="Predict Palestinian checkpoint status using social media and historical data",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
static_path = Path(__file__).parent.parent.parent / "static"
static_path.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(static_path)), name="static")

# Initialize predictor
predictor = CheckpointPredictor()
try:
    predictor.load_models()
    logger.info("Models loaded successfully")
except FileNotFoundError:
    logger.warning("No trained models found. Train models first using: python -m src.models.train")
except Exception as e:
    logger.error(f"Error loading models: {e}")


# Pydantic models for API responses
from pydantic import BaseModel, Field

class CheckpointResponse(BaseModel):
    id: int
    name: str
    name_ar: Optional[str]
    latitude: float
    longitude: float
    checkpoint_type: str
    location_description: Optional[str]
    governorate: Optional[str]
    is_active: bool
    
    class Config:
        from_attributes = True


class PredictionResponse(BaseModel):
    checkpoint_id: int
    checkpoint_name: str
    timestamp: datetime
    short_term: dict
    long_term: dict


class StatusHistoryResponse(BaseModel):
    timestamp: datetime
    status: str
    source: str
    confidence: float
    
    class Config:
        from_attributes = True


class SystemStatusResponse(BaseModel):
    status: str
    checkpoints_count: int
    social_media_posts: int
    predictions_made: int
    models_loaded: bool
    last_data_collection: Optional[datetime]


# API Endpoints

@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the main dashboard"""
    html_path = static_path / "index.html"
    if html_path.exists():
        return FileResponse(html_path)
    return HTMLResponse(content="""
    <html>
        <head><title>Checkpoint Status Prediction</title></head>
        <body style="font-family: Arial; padding: 20px;">
            <h1>Checkpoint Status Prediction API</h1>
            <p>API is running! Visit <a href="/docs">/docs</a> for API documentation.</p>
            <p>Dashboard not yet built. Coming soon!</p>
        </body>
    </html>
    """)


@app.get("/api/status", response_model=SystemStatusResponse)
async def system_status(db: Session = Depends(get_db)):
    """Get system status and statistics"""
    checkpoints_count = db.query(Checkpoint).filter(Checkpoint.is_active == True).count()
    social_media_posts = db.query(SocialMediaPost).count()
    predictions_made = db.query(Prediction).count()
    
    last_post = db.query(SocialMediaPost).order_by(
        SocialMediaPost.collected_at.desc()
    ).first()
    
    return {
        "status": "operational",
        "checkpoints_count": checkpoints_count,
        "social_media_posts": social_media_posts,
        "predictions_made": predictions_made,
        "models_loaded": predictor.short_term_model is not None,
        "last_data_collection": last_post.collected_at if last_post else None
    }


@app.get("/api/checkpoints", response_model=List[CheckpointResponse])
async def list_checkpoints(
    active_only: bool = Query(True, description="Return only active checkpoints"),
    db: Session = Depends(get_db)
):
    """List all checkpoints"""
    query = db.query(Checkpoint)
    
    if active_only:
        query = query.filter(Checkpoint.is_active == True)
    
    checkpoints = query.order_by(Checkpoint.name).all()
    return checkpoints


@app.get("/api/checkpoints/{checkpoint_id}", response_model=CheckpointResponse)
async def get_checkpoint(checkpoint_id: int, db: Session = Depends(get_db)):
    """Get specific checkpoint details"""
    checkpoint = db.query(Checkpoint).filter(Checkpoint.id == checkpoint_id).first()
    
    if not checkpoint:
        raise HTTPException(status_code=404, detail="Checkpoint not found")
    
    return checkpoint


@app.get("/api/checkpoints/{checkpoint_id}/predict", response_model=PredictionResponse)
async def predict_checkpoint_status(
    checkpoint_id: int,
    db: Session = Depends(get_db)
):
    """Get status predictions for a checkpoint"""
    # Check if checkpoint exists
    checkpoint = db.query(Checkpoint).filter(Checkpoint.id == checkpoint_id).first()
    
    if not checkpoint:
        raise HTTPException(status_code=404, detail="Checkpoint not found")
    
    if predictor.short_term_model is None:
        raise HTTPException(
            status_code=503,
            detail="Models not trained yet. Please train models first."
        )
    
    try:
        # Make predictions
        prediction = predictor.predict(checkpoint_id)
        
        # Save predictions to database
        now = datetime.utcnow()
        
        # Short-term prediction
        short_pred = Prediction(
            checkpoint_id=checkpoint_id,
            predicted_status=CheckpointStatus(prediction['short_term']['status']),
            confidence=prediction['short_term']['confidence'],
            prediction_for=prediction['short_term']['prediction_for'],
            horizon_hours=prediction['short_term']['horizon_hours'],
            model_name="short_term_rf",
            created_at=now
        )
        db.add(short_pred)
        
        # Long-term prediction
        long_pred = Prediction(
            checkpoint_id=checkpoint_id,
            predicted_status=CheckpointStatus(prediction['long_term']['status']),
            confidence=prediction['long_term']['confidence'],
            prediction_for=prediction['long_term']['prediction_for'],
            horizon_hours=prediction['long_term']['horizon_hours'],
            model_name="long_term_rf",
            created_at=now
        )
        db.add(long_pred)
        db.commit()
        
        return {
            "checkpoint_id": checkpoint_id,
            "checkpoint_name": checkpoint.name,
            "timestamp": now,
            "short_term": prediction['short_term'],
            "long_term": prediction['long_term']
        }
    
    except Exception as e:
        logger.error(f"Error making prediction: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/checkpoints/{checkpoint_id}/history", response_model=List[StatusHistoryResponse])
async def get_checkpoint_history(
    checkpoint_id: int,
    hours: int = Query(24, description="Hours of history to retrieve"),
    db: Session = Depends(get_db)
):
    """Get historical status for a checkpoint"""
    checkpoint = db.query(Checkpoint).filter(Checkpoint.id == checkpoint_id).first()
    
    if not checkpoint:
        raise HTTPException(status_code=404, detail="Checkpoint not found")
    
    start_time = datetime.utcnow() - timedelta(hours=hours)
    
    history = db.query(CheckpointStatusHistory).filter(
        CheckpointStatusHistory.checkpoint_id == checkpoint_id,
        CheckpointStatusHistory.timestamp >= start_time
    ).order_by(CheckpointStatusHistory.timestamp.desc()).all()
    
    return history


@app.get("/api/checkpoints/{checkpoint_id}/social-media")
async def get_checkpoint_social_media(
    checkpoint_id: int,
    hours: int = Query(24, description="Hours of posts to retrieve"),
    limit: int = Query(50, description="Maximum number of posts"),
    db: Session = Depends(get_db)
):
    """Get recent social media posts mentioning a checkpoint"""
    checkpoint = db.query(Checkpoint).filter(Checkpoint.id == checkpoint_id).first()
    
    if not checkpoint:
        raise HTTPException(status_code=404, detail="Checkpoint not found")
    
    start_time = datetime.utcnow() - timedelta(hours=hours)
    
    posts = db.query(SocialMediaPost).filter(
        SocialMediaPost.checkpoint_id == checkpoint_id,
        SocialMediaPost.posted_at >= start_time
    ).order_by(SocialMediaPost.posted_at.desc()).limit(limit).all()
    
    return [{
        "id": post.id,
        "source": post.source.value,
        "text": (str(post.text) or "")[:200] + "..." if post.text is not None and len(str(post.text)) > 200 else (str(post.text) if post.text is not None else ""),
        "author": post.author,
        "posted_at": post.posted_at,
        "sentiment_score": post.sentiment_score,
        "inferred_status": post.inferred_status.value if post.inferred_status is not None else None,
        "confidence": post.confidence,
        "url": post.url
    } for post in posts]


@app.get("/api/predictions/recent")
async def get_recent_predictions(
    hours: int = Query(24, description="Hours of predictions to retrieve"),
    limit: int = Query(100, description="Maximum number of predictions"),
    db: Session = Depends(get_db)
):
    """Get recent predictions across all checkpoints"""
    start_time = datetime.utcnow() - timedelta(hours=hours)
    
    predictions = db.query(Prediction).filter(
        Prediction.created_at >= start_time
    ).order_by(Prediction.created_at.desc()).limit(limit).all()
    
    return [{
        "checkpoint_id": pred.checkpoint_id,
        "predicted_status": pred.predicted_status.value,
        "confidence": pred.confidence,
        "prediction_for": pred.prediction_for,
        "horizon_hours": pred.horizon_hours,
        "model_name": pred.model_name,
        "created_at": pred.created_at
    } for pred in predictions]


@app.post("/api/checkpoints/{checkpoint_id}/status")
async def report_checkpoint_status(
    checkpoint_id: int,
    status: str,
    source: str = "manual",
    notes: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """Manually report checkpoint status"""
    checkpoint = db.query(Checkpoint).filter(Checkpoint.id == checkpoint_id).first()
    
    if not checkpoint:
        raise HTTPException(status_code=404, detail="Checkpoint not found")
    
    try:
        status_enum = CheckpointStatus(status)
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid status. Must be one of: {[s.value for s in CheckpointStatus]}"
        )
    
    # Create status record
    from database import SourceType
    status_record = CheckpointStatusHistory(
        checkpoint_id=checkpoint_id,
        status=status_enum,
        source=SourceType(source) if source in [s.value for s in SourceType] else SourceType.MANUAL,
        confidence=1.0,
        notes=notes,
        verified=True,
        timestamp=datetime.utcnow()
    )
    
    db.add(status_record)
    db.commit()
    
    return {
        "success": True,
        "checkpoint_id": checkpoint_id,
        "status": status,
        "timestamp": status_record.timestamp
    }


# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.utcnow()}


if __name__ == "__main__":
    import uvicorn
    
    port = int(os.getenv("API_PORT", 8000))
    host = os.getenv("API_HOST", "0.0.0.0")
    
    logger.info(f"Starting API server on {host}:{port}")
    
    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        reload=True,
        log_level="info"
    )
