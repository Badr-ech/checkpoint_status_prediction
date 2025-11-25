"""
Database models for checkpoint status prediction system
"""
from datetime import datetime
from sqlalchemy import Column, Integer, String, Float, DateTime, Boolean, Text, ForeignKey, Enum
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
import enum

Base = declarative_base()


class CheckpointType(str, enum.Enum):
    """Types of checkpoints"""
    PERMANENT = "permanent"
    FLYING = "flying"
    TEMPORARY = "temporary"
    BARRIER = "barrier"


class CheckpointStatus(str, enum.Enum):
    """Possible checkpoint statuses"""
    OPEN = "open"
    CLOSED = "closed"
    PARTIAL = "partial"  # Partially open with restrictions
    UNKNOWN = "unknown"


class SourceType(str, enum.Enum):
    """Data source types"""
    TELEGRAM = "telegram"
    REDDIT = "reddit"
    TWITTER = "twitter"
    MANUAL = "manual"
    GOOGLE_MAPS = "google_maps"


class Checkpoint(Base):
    """Checkpoint location and metadata"""
    __tablename__ = "checkpoints"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), nullable=False, index=True)
    name_ar = Column(String(255), nullable=True)  # Arabic name
    name_he = Column(String(255), nullable=True)  # Hebrew name
    
    latitude = Column(Float, nullable=False)
    longitude = Column(Float, nullable=False)
    
    checkpoint_type = Column(Enum(CheckpointType), default=CheckpointType.PERMANENT)
    location_description = Column(Text, nullable=True)
    
    # Administrative info
    governorate = Column(String(100), nullable=True)  # e.g., "Ramallah", "Bethlehem"
    region = Column(String(100), nullable=True)  # "West Bank" or "Gaza"
    
    # Data sources
    ocha_id = Column(String(100), nullable=True, unique=True)
    osm_id = Column(String(100), nullable=True)
    
    # Metadata
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    status_history = relationship("CheckpointStatusHistory", back_populates="checkpoint")
    social_media_mentions = relationship("SocialMediaPost", back_populates="checkpoint")
    predictions = relationship("Prediction", back_populates="checkpoint")


class CheckpointStatusHistory(Base):
    """Historical checkpoint status records"""
    __tablename__ = "checkpoint_status_history"
    
    id = Column(Integer, primary_key=True, index=True)
    checkpoint_id = Column(Integer, ForeignKey("checkpoints.id"), nullable=False, index=True)
    
    status = Column(Enum(CheckpointStatus), nullable=False)
    confidence = Column(Float, default=1.0)  # 0.0 to 1.0
    
    source = Column(Enum(SourceType), nullable=False)
    source_reference = Column(String(500), nullable=True)  # URL or reference
    
    # Context
    notes = Column(Text, nullable=True)
    verified = Column(Boolean, default=False)
    
    timestamp = Column(DateTime, nullable=False, index=True, default=datetime.utcnow)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    checkpoint = relationship("Checkpoint", back_populates="status_history")


class SocialMediaPost(Base):
    """Social media posts mentioning checkpoints"""
    __tablename__ = "social_media_posts"
    
    id = Column(Integer, primary_key=True, index=True)
    checkpoint_id = Column(Integer, ForeignKey("checkpoints.id"), nullable=True, index=True)
    
    source = Column(Enum(SourceType), nullable=False)
    source_id = Column(String(255), nullable=False, unique=True)  # Platform-specific ID
    
    # Content
    text = Column(Text, nullable=False)
    language = Column(String(10), nullable=True)  # 'ar', 'en', 'he', etc.
    author = Column(String(255), nullable=True)
    
    # Social metrics
    likes = Column(Integer, default=0)
    shares = Column(Integer, default=0)
    comments = Column(Integer, default=0)
    
    # Analysis results
    sentiment_score = Column(Float, nullable=True)  # -1.0 to 1.0
    inferred_status = Column(Enum(CheckpointStatus), nullable=True)
    confidence = Column(Float, nullable=True)
    
    # Metadata
    url = Column(String(500), nullable=True)
    posted_at = Column(DateTime, nullable=False, index=True)
    collected_at = Column(DateTime, default=datetime.utcnow)
    processed = Column(Boolean, default=False)
    
    # Relationships
    checkpoint = relationship("Checkpoint", back_populates="social_media_mentions")


class Prediction(Base):
    """Model predictions for checkpoint status"""
    __tablename__ = "predictions"
    
    id = Column(Integer, primary_key=True, index=True)
    checkpoint_id = Column(Integer, ForeignKey("checkpoints.id"), nullable=False, index=True)
    
    # Prediction details
    predicted_status = Column(Enum(CheckpointStatus), nullable=False)
    confidence = Column(Float, nullable=False)  # 0.0 to 1.0
    
    # Time horizons
    prediction_for = Column(DateTime, nullable=False, index=True)  # When this prediction is for
    horizon_hours = Column(Integer, nullable=False)  # 1, 3, 12, or 24 hours
    
    # Model info
    model_name = Column(String(100), nullable=False)  # 'short_term_rf', 'long_term_rf', etc.
    model_version = Column(String(50), nullable=True)
    
    # Features used (stored as JSON string for reference)
    features_used = Column(Text, nullable=True)
    
    # Evaluation
    actual_status = Column(Enum(CheckpointStatus), nullable=True)  # Filled in later
    was_correct = Column(Boolean, nullable=True)
    
    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    
    # Relationships
    checkpoint = relationship("Checkpoint", back_populates="predictions")


class TrainingJob(Base):
    """Track model training jobs"""
    __tablename__ = "training_jobs"
    
    id = Column(Integer, primary_key=True, index=True)
    
    model_name = Column(String(100), nullable=False)
    model_version = Column(String(50), nullable=False)
    
    # Training parameters
    train_start_date = Column(DateTime, nullable=False)
    train_end_date = Column(DateTime, nullable=False)
    num_samples = Column(Integer, nullable=False)
    
    # Performance metrics
    accuracy = Column(Float, nullable=True)
    precision = Column(Float, nullable=True)
    recall = Column(Float, nullable=True)
    f1_score = Column(Float, nullable=True)
    
    # Status
    status = Column(String(50), default="running")  # running, completed, failed
    error_message = Column(Text, nullable=True)
    
    # Files
    model_path = Column(String(500), nullable=True)
    
    started_at = Column(DateTime, default=datetime.utcnow)
    completed_at = Column(DateTime, nullable=True)


class PalestinianHoliday(Base):
    """Palestinian holidays and special dates that may affect checkpoint status"""
    __tablename__ = "palestinian_holidays"
    
    id = Column(Integer, primary_key=True, index=True)
    
    name = Column(String(255), nullable=False)
    name_ar = Column(String(255), nullable=True)
    
    date = Column(DateTime, nullable=False, index=True)
    holiday_type = Column(String(100), nullable=True)  # religious, national, etc.
    
    # Impact on checkpoints
    expected_closure_increase = Column(Boolean, default=False)
    notes = Column(Text, nullable=True)
    
    created_at = Column(DateTime, default=datetime.utcnow)


class SystemLog(Base):
    """System activity and error logs"""
    __tablename__ = "system_logs"
    
    id = Column(Integer, primary_key=True, index=True)
    
    level = Column(String(20), nullable=False)  # INFO, WARNING, ERROR, etc.
    component = Column(String(100), nullable=False)  # telegram_collector, api, model, etc.
    message = Column(Text, nullable=False)
    
    details = Column(Text, nullable=True)  # JSON string for additional data
    
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
