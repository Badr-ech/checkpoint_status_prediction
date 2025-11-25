"""
Training script for checkpoint prediction models
Run this script after collecting sufficient data (7+ days)
"""
import sys
import os
from datetime import datetime, timedelta
import argparse

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from predictor import CheckpointPredictor
from database import get_db_context, CheckpointStatusHistory, TrainingJob
from utils.logger import setup_logger

logger = setup_logger("train")


def check_data_availability():
    """Check if enough data is available for training"""
    with get_db_context() as db:
        total_records = db.query(CheckpointStatusHistory).count()
        
        if total_records == 0:
            logger.error("No status history found in database!")
            logger.error("Please run data collectors first:")
            logger.error("  python -m src.collectors.telegram_collector")
            logger.error("  python -m src.collectors.reddit_collector")
            return False
        
        logger.info(f"Found {total_records} status records in database")
        
        # Check date range
        oldest = db.query(CheckpointStatusHistory).order_by(
            CheckpointStatusHistory.timestamp
        ).first()
        newest = db.query(CheckpointStatusHistory).order_by(
            CheckpointStatusHistory.timestamp.desc()
        ).first()
        
        if oldest and newest:
            days_of_data = (newest.timestamp - oldest.timestamp).days
            logger.info(f"Data spans {days_of_data} days")
            
            if days_of_data < 7:
                logger.warning(f"Only {days_of_data} days of data available.")
                logger.warning("Recommended: at least 7 days for reliable training.")
                return False
        
        return total_records >= 100  # Minimum samples needed


def train_models(lookback_days: int = 30, version: Optional[str] = None):
    """Train checkpoint prediction models"""
    logger.info("="*60)
    logger.info("Checkpoint Status Prediction Model Training")
    logger.info("="*60)
    
    # Check data availability
    if not check_data_availability():
        logger.error("Insufficient data for training. Exiting.")
        return False
    
    # Initialize predictor
    predictor = CheckpointPredictor()
    
    # Prepare training data
    end_date = datetime.now()
    start_date = end_date - timedelta(days=lookback_days)
    
    logger.info(f"\nPreparing training data:")
    logger.info(f"  Start date: {start_date}")
    logger.info(f"  End date:   {end_date}")
    logger.info(f"  Lookback:   {lookback_days} days")
    
    try:
        X, y_short, _, y_long = predictor.prepare_training_data(start_date, end_date)
        
        logger.info(f"\nTraining data prepared:")
        logger.info(f"  Samples:   {len(X)}")
        logger.info(f"  Features:  {len(X.columns)}")
        
        # Record training job
        with get_db_context() as db:
            job = TrainingJob(
                model_name="dual_horizon_rf",
                model_version=version or datetime.now().strftime("%Y%m%d_%H%M%S"),
                train_start_date=start_date,
                train_end_date=end_date,
                num_samples=len(X),
                status="running"
            )
            db.add(job)
            db.commit()
            job_id = job.id
        
        # Train models
        metrics = predictor.train_models(X, y_short, y_long)
        
        # Save models
        model_path = predictor.save_models(version)
        
        # Update training job with results
        with get_db_context() as db:
            job = db.query(TrainingJob).filter(TrainingJob.id == job_id).first()
            if job:
                # Use setattr to avoid SQLAlchemy Column type issues
                setattr(job, 'accuracy', (metrics['short_term']['accuracy'] + metrics['long_term']['accuracy']) / 2)
                setattr(job, 'precision', (metrics['short_term']['precision'] + metrics['long_term']['precision']) / 2)
                setattr(job, 'recall', (metrics['short_term']['recall'] + metrics['long_term']['recall']) / 2)
                setattr(job, 'f1_score', (metrics['short_term']['f1_score'] + metrics['long_term']['f1_score']) / 2)
                setattr(job, 'status', "completed")
                setattr(job, 'completed_at', datetime.now())
                setattr(job, 'model_path', str(model_path))
                db.commit()
        
        # Print feature importance
        logger.info("\n" + "="*60)
        logger.info("Feature Importance Analysis")
        logger.info("="*60)
        
        importance = predictor.get_feature_importance(top_n=15)
        
        logger.info("\nTop 15 features for SHORT-TERM prediction (1-3h):")
        for i, feat in enumerate(importance['short_term'][:15], 1):
            logger.info(f"  {i:2d}. {feat['feature']:40s} {feat['importance']:.4f}")
        
        logger.info("\nTop 15 features for LONG-TERM prediction (12-24h):")
        for i, feat in enumerate(importance['long_term'][:15], 1):
            logger.info(f"  {i:2d}. {feat['feature']:40s} {feat['importance']:.4f}")
        
        logger.info("\n" + "="*60)
        logger.info("Training Complete!")
        logger.info("="*60)
        logger.info(f"Models saved to: {model_path}")
        logger.info("\nYou can now start the API server:")
        logger.info("  python -m src.api.main")
        
        return True
        
    except Exception as e:
        logger.error(f"Error during training: {str(e)}", exc_info=True)
        
        # Update job status
        with get_db_context() as db:
            job = db.query(TrainingJob).filter(
                TrainingJob.id == job_id
            ).first()
            if job:
                # Use setattr to avoid SQLAlchemy Column type issues
                setattr(job, 'status', "failed")
                setattr(job, 'error_message', str(e))
                setattr(job, 'completed_at', datetime.now())
                db.commit()
        
        return False


def main():
    parser = argparse.ArgumentParser(description="Train checkpoint prediction models")
    parser.add_argument(
        "--lookback-days",
        type=int,
        default=30,
        help="Number of days of historical data to use (default: 30)"
    )
    parser.add_argument(
        "--version",
        type=str,
        default=None,
        help="Model version name (default: timestamp)"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force training even if data is insufficient"
    )
    
    args = parser.parse_args()
    
    if args.force or check_data_availability():
        success = train_models(args.lookback_days, args.version)
        exit(0 if success else 1)
    else:
        logger.error("Training aborted due to insufficient data.")
        logger.error("Use --force to train anyway (not recommended).")
        exit(1)


if __name__ == "__main__":
    main()
