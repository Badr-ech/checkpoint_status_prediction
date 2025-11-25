"""
Machine learning models for checkpoint status prediction
Dual-horizon prediction: short-term (1-3h) and long-term (12-24h)
"""
import sys
import os
from datetime import datetime, timedelta
from typing import Tuple, Optional, Dict, List
import pickle
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
import joblib

from database import (
    get_db_context, Checkpoint, CheckpointStatusHistory,
    CheckpointStatus, Prediction
)
from nlp import FeatureExtractor
from utils.logger import setup_logger

logger = setup_logger("ml_models")


class CheckpointPredictor:
    """Dual-horizon checkpoint status predictor"""
    
    def __init__(self):
        self.short_term_model = None
        self.long_term_model = None
        self.scaler = StandardScaler()
        self.feature_extractor = FeatureExtractor()
        self.feature_names = None
        self.model_dir = Path("models")
        self.model_dir.mkdir(exist_ok=True)
        
    def prepare_training_data(
        self,
        start_date: datetime,
        end_date: datetime,
        min_samples_per_checkpoint: int = 10
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Prepare training data from historical records
        
        Returns:
            X_short, y_short, X_long, y_long
        """
        logger.info(f"Preparing training data from {start_date} to {end_date}")
        
        with get_db_context() as db:
            # Get all active checkpoints
            checkpoints = db.query(Checkpoint).filter(
                Checkpoint.is_active == True
            ).all()
            
            all_data = []
            
            for checkpoint in checkpoints:
                logger.info(f"Processing checkpoint: {checkpoint.name}")
                
                # Get historical status records
                records = db.query(CheckpointStatusHistory).filter(
                    CheckpointStatusHistory.checkpoint_id == checkpoint.id,
                    CheckpointStatusHistory.timestamp >= start_date,
                    CheckpointStatusHistory.timestamp <= end_date
                ).order_by(CheckpointStatusHistory.timestamp).all()
                
                if len(records) < min_samples_per_checkpoint:
                    logger.warning(f"Skipping {checkpoint.name}: only {len(records)} samples")
                    continue
                
                logger.info(f"Found {len(records)} records for {checkpoint.name}")
                
                # Create samples
                for i, record in enumerate(records):
                    # Skip if we can't get future labels
                    if i >= len(records) - 1:
                        continue
                    
                    # Extract features at this point in time
                    # SQLAlchemy Column access - type: ignore
                    cp_id: int = checkpoint.id  # type: ignore
                    rec_time: datetime = record.timestamp  # type: ignore
                    
                    features = self.feature_extractor.extract_all_features(
                        cp_id,
                        rec_time
                    )
                    
                    # Get labels (future status)
                    # Short-term: 1-3 hours ahead
                    short_term_label = self._get_status_at_time(
                        records,
                        rec_time + timedelta(hours=2),  # type: ignore  # Middle of 1-3h range
                        i
                    )
                    
                    # Long-term: 12-24 hours ahead
                    long_term_label = self._get_status_at_time(
                        records,
                        rec_time + timedelta(hours=18),  # type: ignore  # Middle of 12-24h range
                        i
                    )
                    
                    if short_term_label and long_term_label:
                        sample = {
                            **features,
                            'checkpoint_id': checkpoint.id,
                            'timestamp': record.timestamp,
                            'short_term_status': short_term_label.value,
                            'long_term_status': long_term_label.value
                        }
                        all_data.append(sample)
            
            if not all_data:
                raise ValueError("No training data could be prepared!")
            
            logger.info(f"Created {len(all_data)} training samples")
            
            # Convert to DataFrame
            df = pd.DataFrame(all_data)
            
            # Separate features and labels
            feature_cols = [col for col in df.columns if col not in [
                'checkpoint_id', 'timestamp', 'short_term_status', 'long_term_status'
            ]]
            
            self.feature_names = feature_cols
            
            X = df[feature_cols]
            y_short = df['short_term_status']
            y_long = df['long_term_status']
            
            # Handle any missing values
            X = X.fillna(0)
            
            logger.info(f"Feature matrix shape: {X.shape}")
            logger.info(f"Features: {len(feature_cols)}")
            
            # Split into train/test
            X_train, X_test, y_short_train, y_short_test = train_test_split(
                X, y_short, test_size=0.2, random_state=42
            )
            _, _, y_long_train, y_long_test = train_test_split(
                X, y_long, test_size=0.2, random_state=42
            )
            
            return X_train, y_short_train, X_test, y_short_test
    
    def _get_status_at_time(
        self,
        records: List,
        target_time: datetime,
        current_index: int
    ) -> Optional[CheckpointStatus]:
        """Get the status at a specific future time"""
        # Find the closest record to target_time (that comes after current_index)
        future_records = [r for r in records[current_index+1:] if r.timestamp >= target_time]
        
        if not future_records:
            return None
        
        # Return the first status at or after target_time
        return future_records[0].status
    
    def train_models(
        self,
        X_train: pd.DataFrame,
        y_short: pd.Series,
        y_long: pd.Series,
        test_size: float = 0.2
    ) -> Dict:
        """
        Train both short-term and long-term models
        
        Returns:
            Dictionary with training metrics
        """
        logger.info("Starting model training...")
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X_train)
        
        # Train-test split
        X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(
            X_scaled, y_short, test_size=test_size, random_state=42, stratify=y_short
        )
        
        X_train_l, X_test_l, y_train_l, y_test_l = train_test_split(
            X_scaled, y_long, test_size=test_size, random_state=42, stratify=y_long
        )
        
        metrics = {}
        
        # Train short-term model
        logger.info("Training short-term model (1-3 hours)...")
        self.short_term_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=15,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=42,
            n_jobs=-1,
            class_weight='balanced'
        )
        
        self.short_term_model.fit(X_train_s, y_train_s)
        y_pred_s = self.short_term_model.predict(X_test_s)
        
        metrics['short_term'] = self._calculate_metrics(y_test_s, y_pred_s, "Short-term")
        
        # Train long-term model
        logger.info("Training long-term model (12-24 hours)...")
        self.long_term_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=15,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=42,
            n_jobs=-1,
            class_weight='balanced'
        )
        
        self.long_term_model.fit(X_train_l, y_train_l)
        y_pred_l = self.long_term_model.predict(X_test_l)
        
        metrics['long_term'] = self._calculate_metrics(y_test_l, y_pred_l, "Long-term")
        
        logger.info("Model training complete!")
        
        return metrics
    
    def _calculate_metrics(self, y_true, y_pred, model_name: str) -> Dict:
        """Calculate and log metrics"""
        accuracy = accuracy_score(y_true, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average='weighted', zero_division=0
        )
        
        logger.info(f"\n{model_name} Model Performance:")
        logger.info(f"  Accuracy:  {accuracy:.3f}")
        logger.info(f"  Precision: {precision:.3f}")
        logger.info(f"  Recall:    {recall:.3f}")
        logger.info(f"  F1 Score:  {f1:.3f}")
        
        # Detailed classification report
        logger.info(f"\n{model_name} Classification Report:")
        logger.info(f"\n{classification_report(y_true, y_pred)}")
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }
    
    def predict(
        self,
        checkpoint_id: int,
        reference_time: Optional[datetime] = None
    ) -> Dict:
        """
        Make predictions for a checkpoint
        
        Returns:
            Dictionary with short-term and long-term predictions
        """
        if reference_time is None:
            reference_time = datetime.utcnow()
        
        if self.short_term_model is None or self.long_term_model is None:
            raise ValueError("Models not trained! Call train_models() first or load_models()")
        
        # Extract features
        features = self.feature_extractor.extract_all_features(
            checkpoint_id, reference_time
        )
        
        # Convert to DataFrame with correct feature order
        X = pd.DataFrame([features])[self.feature_names]
        X = X.fillna(0)
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        # Short-term prediction
        short_pred = self.short_term_model.predict(X_scaled)[0]
        short_proba = self.short_term_model.predict_proba(X_scaled)[0]
        short_confidence = np.max(short_proba)
        
        # Long-term prediction
        long_pred = self.long_term_model.predict(X_scaled)[0]
        long_proba = self.long_term_model.predict_proba(X_scaled)[0]
        long_confidence = np.max(long_proba)
        
        return {
            'checkpoint_id': checkpoint_id,
            'reference_time': reference_time,
            'short_term': {
                'status': short_pred,
                'confidence': float(short_confidence),
                'prediction_for': reference_time + timedelta(hours=2),
                'horizon_hours': 2
            },
            'long_term': {
                'status': long_pred,
                'confidence': float(long_confidence),
                'prediction_for': reference_time + timedelta(hours=18),
                'horizon_hours': 18
            }
        }
    
    def save_models(self, version: Optional[str] = None):
        """Save trained models to disk"""
        if version is None:
            version = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        models_data = {
            'short_term_model': self.short_term_model,
            'long_term_model': self.long_term_model,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'version': version,
            'trained_at': datetime.now()
        }
        
        filepath = self.model_dir / f"checkpoint_models_{version}.joblib"
        joblib.dump(models_data, filepath)
        
        logger.info(f"Models saved to {filepath}")
        
        # Also save as "latest"
        latest_path = self.model_dir / "checkpoint_models_latest.joblib"
        joblib.dump(models_data, latest_path)
        logger.info(f"Models also saved as latest: {latest_path}")
        
        return filepath
    
    def load_models(self, filepath: Optional[str] = None):
        """Load trained models from disk"""
        if filepath is None:
            filepath = str(self.model_dir / "checkpoint_models_latest.joblib")
        
        filepath_obj = Path(filepath)
        
        if not filepath_obj.exists():
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        logger.info(f"Loading models from {filepath}")
        
        models_data = joblib.load(filepath_obj)
        
        self.short_term_model = models_data['short_term_model']
        self.long_term_model = models_data['long_term_model']
        self.scaler = models_data['scaler']
        self.feature_names = models_data['feature_names']
        
        logger.info(f"Models loaded successfully (version: {models_data.get('version', 'unknown')})")
        logger.info(f"Trained at: {models_data.get('trained_at', 'unknown')}")
    
    def get_feature_importance(self, top_n: int = 20) -> Dict:
        """Get feature importance from models"""
        if self.short_term_model is None or self.long_term_model is None:
            return {}
        
        short_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.short_term_model.feature_importances_
        }).sort_values('importance', ascending=False).head(top_n)
        
        long_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.long_term_model.feature_importances_
        }).sort_values('importance', ascending=False).head(top_n)
        
        return {
            'short_term': short_importance.to_dict('records'),
            'long_term': long_importance.to_dict('records')
        }


# Example usage
if __name__ == "__main__":
    predictor = CheckpointPredictor()
    
    # Check if we have enough data
    with get_db_context() as db:
        status_count = db.query(CheckpointStatusHistory).count()
        logger.info(f"Total status records in database: {status_count}")
        
        if status_count < 100:
            logger.warning("Not enough data for training! Need at least 100 status records.")
            logger.info("Run data collectors first to gather historical data.")
        else:
            # Prepare data
            end_date = datetime.now()
            start_date = end_date - timedelta(days=30)
            
            X, y_short, _, y_long = predictor.prepare_training_data(start_date, end_date)
            
            # Ensure y values are Series - type: ignore for pandas conversion
            y_short_series = y_short.iloc[:, 0] if hasattr(y_short, 'iloc') and len(y_short.shape) > 1 else y_short  # type: ignore
            y_long_series = y_long.iloc[:, 0] if hasattr(y_long, 'iloc') and len(y_long.shape) > 1 else y_long  # type: ignore
            
            # Train models
            metrics = predictor.train_models(X, y_short_series, y_long_series)  # type: ignore
            
            # Save models
            predictor.save_models()
            
            # Show feature importance
            importance = predictor.get_feature_importance()
            logger.info("\nTop 10 features for short-term prediction:")
            for feat in importance['short_term'][:10]:
                logger.info(f"  {feat['feature']}: {feat['importance']:.4f}")
