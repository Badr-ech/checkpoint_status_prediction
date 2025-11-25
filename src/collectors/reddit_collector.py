"""
Reddit data collector for checkpoint status
Monitors specified subreddits for checkpoint-related posts and comments
"""
import sys
import os
from datetime import datetime, timedelta
from typing import List, Optional
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import praw
from praw.models import Submission, Comment
from dotenv import load_dotenv

from database import get_db_context, Checkpoint, SocialMediaPost, SourceType, CheckpointStatus
from utils.logger import setup_logger

load_dotenv()

# Configuration
CLIENT_ID = os.getenv("REDDIT_CLIENT_ID")
CLIENT_SECRET = os.getenv("REDDIT_CLIENT_SECRET")
USER_AGENT = os.getenv("REDDIT_USER_AGENT", "checkpoint_predictor_v1.0")
SUBREDDITS = os.getenv("REDDIT_SUBREDDITS", "Palestine,Palestinians").split(",")

logger = setup_logger("reddit_collector")


class RedditCollector:
    """Collects checkpoint-related posts from Reddit"""
    
    def __init__(self):
        self.reddit = None
        self.checkpoints: List[Checkpoint] = []
        self.checkpoint_keywords = {}
        
    def initialize(self):
        """Initialize Reddit client and load checkpoints"""
        if not CLIENT_ID or not CLIENT_SECRET:
            raise ValueError("Reddit API credentials not configured. Check .env file.")
        
        # Create Reddit client
        self.reddit = praw.Reddit(
            client_id=CLIENT_ID,
            client_secret=CLIENT_SECRET,
            user_agent=USER_AGENT
        )
        
        logger.info("Reddit client initialized successfully")
        
        # Load checkpoints
        self.load_checkpoints()
        
        logger.info(f"Loaded {len(self.checkpoints)} checkpoints")
    
    def load_checkpoints(self):
        """Load active checkpoints from database"""
        with get_db_context() as db:
            self.checkpoints = db.query(Checkpoint).filter(
                Checkpoint.is_active == True
            ).all()
            
            # Generate keywords for each checkpoint
            for checkpoint in self.checkpoints:
                keywords = self._generate_keywords(checkpoint)
                self.checkpoint_keywords[checkpoint.id] = keywords
    
    def _generate_keywords(self, checkpoint: Checkpoint) -> List[str]:
        """Generate search keywords for a checkpoint"""
        keywords = []
        
        # Convert Column types to strings to avoid type checking issues
        name = str(checkpoint.name) if checkpoint.name is not None else None
        if name:
            keywords.append(name.lower())
            base_name = name.lower().replace("checkpoint", "").replace("barrier", "").strip()
            if base_name:
                keywords.append(base_name)
        
        name_ar = str(checkpoint.name_ar) if checkpoint.name_ar is not None else None
        if name_ar:
            keywords.append(name_ar)
        
        governorate = str(checkpoint.governorate) if checkpoint.governorate is not None else None
        if governorate:
            keywords.append(governorate.lower())
        
        return list(set(keywords))
    
    def _find_mentioned_checkpoints(self, text: str) -> List[int]:
        """Find which checkpoints are mentioned in the text"""
        if not text:
            return []
        
        text_lower = text.lower()
        mentioned = []
        
        for checkpoint_id, keywords in self.checkpoint_keywords.items():
            for keyword in keywords:
                if keyword and keyword in text_lower:
                    mentioned.append(checkpoint_id)
                    break
        
        return list(set(mentioned))
    
    def _infer_status(self, text: str) -> tuple[Optional[CheckpointStatus], float]:
        """Infer checkpoint status from text"""
        if not text:
            return CheckpointStatus.UNKNOWN, 0.3
        
        text_lower = text.lower()
        
        closed_keywords = ["closed", "closure", "blocked", "shut", "shutdown", "inaccessible"]
        open_keywords = ["open", "opened", "accessible", "passage", "flowing"]
        partial_keywords = ["partial", "restricted", "limited", "delays", "slow", "queue"]
        
        closed_count = sum(1 for kw in closed_keywords if kw in text_lower)
        open_count = sum(1 for kw in open_keywords if kw in text_lower)
        partial_count = sum(1 for kw in partial_keywords if kw in text_lower)
        
        if closed_count > open_count and closed_count > partial_count:
            return CheckpointStatus.CLOSED, min(0.6 + (closed_count * 0.1), 0.9)
        elif open_count > closed_count and open_count > partial_count:
            return CheckpointStatus.OPEN, min(0.6 + (open_count * 0.1), 0.9)
        elif partial_count > 0:
            return CheckpointStatus.PARTIAL, min(0.5 + (partial_count * 0.1), 0.8)
        
        return CheckpointStatus.UNKNOWN, 0.3
    
    def collect_historical(self, subreddit_name: str, days_back: int = 7):
        """Collect historical posts from a subreddit"""
        try:
            logger.info(f"Collecting from r/{subreddit_name} ({days_back} days)")
            
            sub = self.reddit.subreddit(subreddit_name) if self.reddit else None
            if not sub:
                logger.error("Reddit client not initialized")
                return
            
            limit_timestamp = time.time() - (days_back * 24 * 60 * 60)
            
            posts_collected = 0
            posts_saved = 0
            
            # Collect from new, hot, and top posts
            for submission in sub.new(limit=100):
                if submission.created_utc < limit_timestamp:
                    continue
                
                posts_collected += 1
                
                # Check submission title and body
                text = f"{submission.title} {submission.selftext}"
                mentioned = self._find_mentioned_checkpoints(text)
                
                if mentioned:
                    for checkpoint_id in mentioned:
                        if self._save_post(submission, checkpoint_id):
                            posts_saved += 1
                
                # Check top comments
                submission.comments.replace_more(limit=0)
                for comment in submission.comments.list()[:20]:
                    if isinstance(comment, Comment) and comment.body:
                        mentioned = self._find_mentioned_checkpoints(comment.body)
                        if mentioned:
                            for checkpoint_id in mentioned:
                                if self._save_comment(comment, submission, checkpoint_id):
                                    posts_saved += 1
            
            logger.info(f"r/{subreddit_name}: collected {posts_collected}, saved {posts_saved}")
            
        except Exception as e:
            logger.error(f"Error collecting from r/{subreddit_name}: {str(e)}")
    
    def _save_post(self, submission: Submission, checkpoint_id: int) -> bool:
        """Save a Reddit post to the database"""
        try:
            with get_db_context() as db:
                source_id = f"reddit_post_{submission.id}"
                
                existing = db.query(SocialMediaPost).filter(
                    SocialMediaPost.source_id == source_id
                ).first()
                
                if existing:
                    return False
                
                text = f"{submission.title}\n\n{submission.selftext}"
                inferred_status, confidence = self._infer_status(text)
                
                post = SocialMediaPost(
                    checkpoint_id=checkpoint_id,
                    source=SourceType.REDDIT,
                    source_id=source_id,
                    text=text[:10000],
                    language="en",
                    author=str(submission.author) if submission.author else "[deleted]",
                    likes=submission.score,
                    comments=submission.num_comments,
                    url=f"https://reddit.com{submission.permalink}",
                    posted_at=datetime.fromtimestamp(submission.created_utc),
                    collected_at=datetime.utcnow(),
                    inferred_status=inferred_status,
                    confidence=confidence,
                    processed=False
                )
                
                db.add(post)
                db.commit()
                
                logger.debug(f"Saved post {submission.id} for checkpoint {checkpoint_id}")
                return True
                
        except Exception as e:
            logger.error(f"Error saving post: {str(e)}")
            return False
    
    def _save_comment(self, comment: Comment, submission: Submission, checkpoint_id: int) -> bool:
        """Save a Reddit comment to the database"""
        try:
            with get_db_context() as db:
                source_id = f"reddit_comment_{comment.id}"
                
                existing = db.query(SocialMediaPost).filter(
                    SocialMediaPost.source_id == source_id
                ).first()
                
                if existing:
                    return False
                
                inferred_status, confidence = self._infer_status(comment.body)
                
                post = SocialMediaPost(
                    checkpoint_id=checkpoint_id,
                    source=SourceType.REDDIT,
                    source_id=source_id,
                    text=comment.body[:10000],
                    language="en",
                    author=str(comment.author) if comment.author else "[deleted]",
                    likes=comment.score,
                    url=f"https://reddit.com{submission.permalink}{comment.id}/",
                    posted_at=datetime.fromtimestamp(comment.created_utc),
                    collected_at=datetime.utcnow(),
                    inferred_status=inferred_status,
                    confidence=confidence,
                    processed=False
                )
                
                db.add(post)
                db.commit()
                
                return True
                
        except Exception as e:
            logger.error(f"Error saving comment: {str(e)}")
            return False
    
    def monitor_continuously(self, interval_minutes: int = 30):
        """Continuously monitor subreddits"""
        logger.info(f"Starting continuous monitoring (interval: {interval_minutes} minutes)")
        
        while True:
            try:
                for subreddit_name in SUBREDDITS:
                    if subreddit_name.strip():
                        self.collect_historical(subreddit_name.strip(), days_back=1)
                
                logger.info(f"Sleeping for {interval_minutes} minutes...")
                time.sleep(interval_minutes * 60)
                
            except KeyboardInterrupt:
                logger.info("Monitoring stopped by user")
                break
            except Exception as e:
                logger.error(f"Error in monitoring loop: {str(e)}")
                time.sleep(60)


def main():
    """Main function"""
    collector = RedditCollector()
    
    try:
        collector.initialize()
        
        # Collect historical data
        logger.info("Starting historical data collection...")
        for subreddit in SUBREDDITS:
            if subreddit.strip():
                collector.collect_historical(subreddit.strip(), days_back=7)
        
        logger.info("Historical collection complete. Starting continuous monitoring...")
        
        # Start continuous monitoring
        collector.monitor_continuously(interval_minutes=30)
        
    except KeyboardInterrupt:
        logger.info("Shutting down...")
    except Exception as e:
        logger.error(f"Error: {str(e)}", exc_info=True)


if __name__ == "__main__":
    main()
