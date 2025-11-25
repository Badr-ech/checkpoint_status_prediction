"""
Feature engineering for checkpoint status prediction
Extracts temporal, social media, and historical features
"""
import sys
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import pandas as pd
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database import (
    get_db_context, Checkpoint, SocialMediaPost,
    CheckpointStatusHistory, CheckpointStatus
)


class FeatureExtractor:
    """Extract features for checkpoint status prediction"""
    
    def __init__(self):
        self.palestinian_holidays = self._load_holidays()
    
    def _load_holidays(self) -> List[datetime]:
        """Load Palestinian holidays from database"""
        # In production, load from database
        # For now, return some common dates
        return [
            datetime(2025, 3, 31),  # Eid al-Fitr (approximate)
            datetime(2025, 6, 7),   # Eid al-Adha (approximate)
            datetime(2025, 11, 15), # Palestinian Independence Day
        ]
    
    def extract_temporal_features(self, timestamp: datetime) -> Dict:
        """
        Extract time-based features
        
        Returns:
            Dictionary of temporal features
        """
        features = {
            # Basic time features
            "hour": timestamp.hour,
            "day_of_week": timestamp.weekday(),  # 0=Monday, 6=Sunday
            "day_of_month": timestamp.day,
            "month": timestamp.month,
            "week_of_year": timestamp.isocalendar()[1],
            
            # Categorical time features
            "is_weekend": int(timestamp.weekday() >= 5),  # Friday-Saturday
            "is_morning": int(6 <= timestamp.hour < 12),
            "is_afternoon": int(12 <= timestamp.hour < 18),
            "is_evening": int(18 <= timestamp.hour < 22),
            "is_night": int(timestamp.hour >= 22 or timestamp.hour < 6),
            
            # Peak hours (common travel times)
            "is_peak_morning": int(6 <= timestamp.hour <= 9),
            "is_peak_evening": int(16 <= timestamp.hour <= 19),
            
            # Friday (important day for closures)
            "is_friday": int(timestamp.weekday() == 4),
            
            # Holiday proximity
            "is_holiday": int(any(abs((h - timestamp).days) == 0 for h in self.palestinian_holidays)),
            "days_to_next_holiday": self._days_to_next_holiday(timestamp),
            "days_from_last_holiday": self._days_from_last_holiday(timestamp),
        }
        
        # Cyclical encoding for circular features
        features["hour_sin"] = np.sin(2 * np.pi * timestamp.hour / 24)
        features["hour_cos"] = np.cos(2 * np.pi * timestamp.hour / 24)
        features["day_sin"] = np.sin(2 * np.pi * timestamp.weekday() / 7)
        features["day_cos"] = np.cos(2 * np.pi * timestamp.weekday() / 7)
        features["month_sin"] = np.sin(2 * np.pi * timestamp.month / 12)
        features["month_cos"] = np.cos(2 * np.pi * timestamp.month / 12)
        
        return features
    
    def _days_to_next_holiday(self, timestamp: datetime) -> int:
        """Calculate days to next holiday"""
        future_holidays = [h for h in self.palestinian_holidays if h > timestamp]
        if not future_holidays:
            return 365
        return (min(future_holidays) - timestamp).days
    
    def _days_from_last_holiday(self, timestamp: datetime) -> int:
        """Calculate days from last holiday"""
        past_holidays = [h for h in self.palestinian_holidays if h < timestamp]
        if not past_holidays:
            return 365
        return (timestamp - max(past_holidays)).days
    
    def extract_social_media_features(
        self,
        checkpoint_id: int,
        reference_time: datetime,
        lookback_hours: int = 24
    ) -> Dict:
        """
        Extract features from social media posts
        
        Args:
            checkpoint_id: Checkpoint ID
            reference_time: Time to extract features for
            lookback_hours: How many hours to look back
            
        Returns:
            Dictionary of social media features
        """
        with get_db_context() as db:
            # Get posts within lookback window
            start_time = reference_time - timedelta(hours=lookback_hours)
            
            posts = db.query(SocialMediaPost).filter(
                SocialMediaPost.checkpoint_id == checkpoint_id,
                SocialMediaPost.posted_at >= start_time,
                SocialMediaPost.posted_at <= reference_time
            ).all()
            
            if not posts:
                return self._empty_social_features()
            
            # Calculate features
            # Extract sentiment scores as floats to avoid Column type issues
            # SQLAlchemy Column access - type: ignore
            sentiment_scores = [float(p.sentiment_score) if p.sentiment_score is not None else 0.0 for p in posts]  # type: ignore
            confidence_scores = [float(p.confidence) if p.confidence is not None else 0.0 for p in posts]  # type: ignore
            
            features = {
                # Volume features
                f"mentions_last_{lookback_hours}h": len(posts),
                f"mentions_last_1h": len([p for p in posts if (reference_time - p.posted_at).total_seconds() <= 3600]),
                f"mentions_last_3h": len([p for p in posts if (reference_time - p.posted_at).total_seconds() <= 10800]),
                f"mentions_last_6h": len([p for p in posts if (reference_time - p.posted_at).total_seconds() <= 21600]),
                
                # Sentiment features
                f"avg_sentiment_{lookback_hours}h": float(np.mean(sentiment_scores)) if sentiment_scores else 0.0,
                f"min_sentiment_{lookback_hours}h": float(np.min(sentiment_scores)) if sentiment_scores else 0.0,
                f"max_sentiment_{lookback_hours}h": float(np.max(sentiment_scores)) if sentiment_scores else 0.0,
                f"std_sentiment_{lookback_hours}h": float(np.std(sentiment_scores)) if sentiment_scores else 0.0,
                
                # Status inference from social media
                f"closed_mentions_{lookback_hours}h": sum(1 for p in posts if str(p.inferred_status) == str(CheckpointStatus.CLOSED)),
                f"open_mentions_{lookback_hours}h": sum(1 for p in posts if str(p.inferred_status) == str(CheckpointStatus.OPEN)),
                f"partial_mentions_{lookback_hours}h": sum(1 for p in posts if str(p.inferred_status) == str(CheckpointStatus.PARTIAL)),
                
                # Confidence features
                f"avg_confidence_{lookback_hours}h": float(np.mean(confidence_scores)) if confidence_scores else 0.0,
                
                # Source diversity
                f"telegram_mentions_{lookback_hours}h": sum(1 for p in posts if str(p.source) == "telegram"),
                f"reddit_mentions_{lookback_hours}h": sum(1 for p in posts if str(p.source) == "reddit"),
                
                # Engagement features - type: ignore for SQLAlchemy Column conversion
                f"total_likes_{lookback_hours}h": sum(int(p.likes) if p.likes is not None else 0 for p in posts),  # type: ignore
                f"total_comments_{lookback_hours}h": sum(int(p.comments) if p.comments is not None else 0 for p in posts),  # type: ignore
            }
            
            # Rate of mentions (mentions per hour)
            features[f"mention_rate_{lookback_hours}h"] = len(posts) / lookback_hours
            
            # Weighted sentiment (by confidence)
            # Extract values to avoid Column type issues - type: ignore for SQLAlchemy
            weighted_items = [(float(p.sentiment_score), float(p.confidence))  # type: ignore
                            for p in posts 
                            if p.confidence is not None and p.sentiment_score is not None]
            if weighted_items:
                sentiments_list = [item[0] for item in weighted_items]
                confidences_list = [item[1] for item in weighted_items]
                features[f"weighted_sentiment_{lookback_hours}h"] = float(np.average(sentiments_list, weights=confidences_list))
            else:
                features[f"weighted_sentiment_{lookback_hours}h"] = 0.0
            
            return features
    
    def _empty_social_features(self) -> Dict:
        """Return empty/zero social media features"""
        lookback_hours = 24
        return {
            f"mentions_last_{lookback_hours}h": 0,
            f"mentions_last_1h": 0,
            f"mentions_last_3h": 0,
            f"mentions_last_6h": 0,
            f"avg_sentiment_{lookback_hours}h": 0.0,
            f"min_sentiment_{lookback_hours}h": 0.0,
            f"max_sentiment_{lookback_hours}h": 0.0,
            f"std_sentiment_{lookback_hours}h": 0.0,
            f"closed_mentions_{lookback_hours}h": 0,
            f"open_mentions_{lookback_hours}h": 0,
            f"partial_mentions_{lookback_hours}h": 0,
            f"avg_confidence_{lookback_hours}h": 0.0,
            f"telegram_mentions_{lookback_hours}h": 0,
            f"reddit_mentions_{lookback_hours}h": 0,
            f"total_likes_{lookback_hours}h": 0,
            f"total_comments_{lookback_hours}h": 0,
            f"mention_rate_{lookback_hours}h": 0.0,
            f"weighted_sentiment_{lookback_hours}h": 0.0,
        }
    
    def extract_historical_features(
        self,
        checkpoint_id: int,
        reference_time: datetime,
        lookback_days: int = 30
    ) -> Dict:
        """
        Extract historical pattern features
        
        Args:
            checkpoint_id: Checkpoint ID
            reference_time: Time to extract features for
            lookback_days: How many days of history to use
            
        Returns:
            Dictionary of historical features
        """
        with get_db_context() as db:
            start_time = reference_time - timedelta(days=lookback_days)
            
            # Get historical status records
            history = db.query(CheckpointStatusHistory).filter(
                CheckpointStatusHistory.checkpoint_id == checkpoint_id
            ).filter(
                CheckpointStatusHistory.timestamp >= start_time
            ).filter(
                CheckpointStatusHistory.timestamp < reference_time
            ).all()
            
            if not history:
                return self._empty_historical_features()
            
            # Calculate closure rate - convert to string for comparison
            closed_count = sum(1 for h in history if str(h.status) == str(CheckpointStatus.CLOSED))
            open_count = sum(1 for h in history if str(h.status) == str(CheckpointStatus.OPEN))
            total_count = len(history)
            
            features = {
                # Overall closure statistics
                "historical_closure_rate": closed_count / total_count if total_count > 0 else 0.5,
                "historical_open_rate": open_count / total_count if total_count > 0 else 0.5,
                "total_historical_records": total_count,
                
                # Time-specific patterns
                "closure_rate_same_hour": self._calculate_time_specific_rate(
                    history, reference_time, CheckpointStatus.CLOSED, by="hour"
                ),
                "closure_rate_same_dow": self._calculate_time_specific_rate(
                    history, reference_time, CheckpointStatus.CLOSED, by="dow"
                ),
                "closure_rate_weekend": self._calculate_weekend_rate(history, CheckpointStatus.CLOSED),
                
                # Recent trend
                "closures_last_7_days": sum(1 for h in history if str(h.status) == str(CheckpointStatus.CLOSED) and (reference_time - h.timestamp).days <= 7),
                "closures_last_3_days": sum(1 for h in history if str(h.status) == str(CheckpointStatus.CLOSED) and (reference_time - h.timestamp).days <= 3),
                
                # Last known status
                "hours_since_last_status": self._hours_since_last_status(history, reference_time),
            }
            
            # Last known status as one-hot encoding
            last_status = self._get_last_known_status(history)
            features["last_status_was_closed"] = int(str(last_status) == str(CheckpointStatus.CLOSED))
            features["last_status_was_open"] = int(last_status == CheckpointStatus.OPEN)
            features["last_status_was_partial"] = int(last_status == CheckpointStatus.PARTIAL)
            
            return features
    
    def _empty_historical_features(self) -> Dict:
        """Return empty historical features"""
        return {
            "historical_closure_rate": 0.5,
            "historical_open_rate": 0.5,
            "total_historical_records": 0,
            "closure_rate_same_hour": 0.5,
            "closure_rate_same_dow": 0.5,
            "closure_rate_weekend": 0.5,
            "closures_last_7_days": 0,
            "closures_last_3_days": 0,
            "hours_since_last_status": 999,
            "last_status_was_closed": 0,
            "last_status_was_open": 0,
            "last_status_was_partial": 0,
        }
    
    def _calculate_time_specific_rate(
        self,
        history: List,
        reference_time: datetime,
        status: CheckpointStatus,
        by: str = "hour"
    ) -> float:
        """Calculate closure rate for specific time (hour or day of week)"""
        if by == "hour":
            relevant = [h for h in history if h.timestamp.hour == reference_time.hour]
        elif by == "dow":
            relevant = [h for h in history if h.timestamp.weekday() == reference_time.weekday()]
        else:
            return 0.5
        
        if not relevant:
            return 0.5
        
        status_count = len([h for h in relevant if h.status == status])
        return status_count / len(relevant)
    
    def _calculate_weekend_rate(self, history: List, status: CheckpointStatus) -> float:
        """Calculate closure rate on weekends"""
        weekend = [h for h in history if h.timestamp.weekday() >= 5]
        if not weekend:
            return 0.5
        status_count = len([h for h in weekend if h.status == status])
        return status_count / len(weekend)
    
    def _hours_since_last_status(self, history: List, reference_time: datetime) -> float:
        """Calculate hours since last status update"""
        if not history:
            return 999.0
        sorted_history = sorted(history, key=lambda x: x.timestamp, reverse=True)
        time_diff = reference_time - sorted_history[0].timestamp
        return time_diff.total_seconds() / 3600
    
    def _get_last_known_status(self, history: List) -> Optional[CheckpointStatus]:
        """Get the last known status"""
        if not history:
            return None
        sorted_history = sorted(history, key=lambda x: x.timestamp, reverse=True)
        return sorted_history[0].status
    
    def extract_all_features(
        self,
        checkpoint_id: int,
        reference_time: datetime,
        lookback_hours_social: int = 24,
        lookback_days_historical: int = 30
    ) -> Dict:
        """
        Extract all features for a checkpoint at a specific time
        
        Returns:
            Dictionary with all features combined
        """
        features = {}
        
        # Temporal features
        features.update(self.extract_temporal_features(reference_time))
        
        # Social media features
        features.update(self.extract_social_media_features(
            checkpoint_id, reference_time, lookback_hours_social
        ))
        
        # Historical features
        features.update(self.extract_historical_features(
            checkpoint_id, reference_time, lookback_days_historical
        ))
        
        # Checkpoint-specific features (static)
        features.update(self._extract_checkpoint_features(checkpoint_id))
        
        return features
    
    def _extract_checkpoint_features(self, checkpoint_id: int) -> Dict:
        """Extract static checkpoint features"""
        with get_db_context() as db:
            checkpoint = db.query(Checkpoint).filter(
                Checkpoint.id == checkpoint_id
            ).first()
            
            if not checkpoint:
                return {}
            
            # One-hot encoding for checkpoint type
            return {
                "is_permanent_checkpoint": int(checkpoint.checkpoint_type.value == "permanent"),
                "is_flying_checkpoint": int(checkpoint.checkpoint_type.value == "flying"),
                "is_temporary_checkpoint": int(checkpoint.checkpoint_type.value == "temporary"),
                "checkpoint_latitude": checkpoint.latitude,
                "checkpoint_longitude": checkpoint.longitude,
            }


# Example usage
if __name__ == "__main__":
    extractor = FeatureExtractor()
    
    # Test with current time
    now = datetime.now()
    checkpoint_id = 1  # Assuming checkpoint exists
    
    print("Testing Feature Extraction:\n")
    
    # Temporal features
    temporal = extractor.extract_temporal_features(now)
    print(f"Temporal features: {len(temporal)} features")
    print(f"  Hour: {temporal['hour']}, Day of week: {temporal['day_of_week']}")
    print(f"  Is weekend: {temporal['is_weekend']}, Is Friday: {temporal['is_friday']}")
    
    # All features
    all_features = extractor.extract_all_features(checkpoint_id, now)
    print(f"\nTotal features: {len(all_features)}")
    print(f"Feature names: {list(all_features.keys())[:10]}...")
