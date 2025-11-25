"""
Telegram data collector for checkpoint status
Monitors specified Telegram channels for checkpoint-related posts
"""
import sys
import os
import re
from datetime import datetime, timedelta
from typing import List, Optional, Dict
import asyncio

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from telethon import TelegramClient, events
from telethon.tl.types import Message
from dotenv import load_dotenv

from database import get_db_context, Checkpoint, SocialMediaPost, SourceType, CheckpointStatus
from utils.logger import setup_logger

load_dotenv()

# Configuration
API_ID = os.getenv("TELEGRAM_API_ID")
API_HASH = os.getenv("TELEGRAM_API_HASH")
PHONE = os.getenv("TELEGRAM_PHONE")
SESSION_NAME = os.getenv("TELEGRAM_SESSION_NAME", "checkpoint_monitor")
CHANNELS = os.getenv("TELEGRAM_CHANNELS", "").split(",")

logger = setup_logger("telegram_collector")


class TelegramCollector:
    """Collects checkpoint-related posts from Telegram channels"""
    
    def __init__(self):
        self.client = None
        self.checkpoints: List[Checkpoint] = []
        self.checkpoint_keywords: Dict[int, List[str]] = {}
        
    async def initialize(self):
        """Initialize Telegram client and load checkpoints"""
        if not API_ID or not API_HASH:
            raise ValueError("Telegram API credentials not configured. Check .env file.")
        
        # Create Telegram client
        self.client = TelegramClient(SESSION_NAME, int(API_ID) if API_ID else 0, API_HASH)
        if self.client:
            phone_input = PHONE if PHONE else lambda: input("Enter phone: ")
            await self.client.start(phone=phone_input)  # type: ignore
        
        logger.info("Telegram client initialized successfully")
        
        # Load checkpoints from database
        await self.load_checkpoints()
        
        logger.info(f"Loaded {len(self.checkpoints)} checkpoints")
    
    async def load_checkpoints(self):
        """Load active checkpoints from database"""
        with get_db_context() as db:
            self.checkpoints = db.query(Checkpoint).filter(
                Checkpoint.is_active == True
            ).all()
            
            # Generate keywords for each checkpoint
            for checkpoint in self.checkpoints:
                keywords = self._generate_keywords(checkpoint)
                # Use type: ignore for SQLAlchemy Column access
                cp_id: int = checkpoint.id  # type: ignore
                self.checkpoint_keywords[cp_id] = keywords
                logger.debug(f"Keywords for {checkpoint.name}: {keywords}")
    
    def _generate_keywords(self, checkpoint: Checkpoint) -> List[str]:
        """Generate search keywords for a checkpoint"""
        keywords = []
        
        # English name variations
        name = str(checkpoint.name) if checkpoint.name is not None else None
        if name:
            keywords.append(name.lower())
            # Remove "checkpoint" and "barrier" from name
            base_name = name.lower()
            base_name = base_name.replace("checkpoint", "").replace("barrier", "").strip()
            if base_name:
                keywords.append(base_name)
        
        # Arabic name
        name_ar = str(checkpoint.name_ar) if checkpoint.name_ar is not None else None
        if name_ar:
            keywords.append(name_ar)
            # Also add without "حاجز" (checkpoint in Arabic)
            ar_base = name_ar.replace("حاجز", "").strip()
            if ar_base:
                keywords.append(ar_base)
        
        # Hebrew name
        name_he = str(checkpoint.name_he) if checkpoint.name_he is not None else None
        if name_he:
            keywords.append(name_he)
            # Also add without "מחסום" (checkpoint in Hebrew)
            he_base = name_he.replace("מחסום", "").strip()
            if he_base:
                keywords.append(he_base)
        
        # Location-based keywords
        governorate = str(checkpoint.governorate) if checkpoint.governorate is not None else None
        if governorate:
            keywords.append(governorate.lower())
        
        # OCHA ID
        ocha_id = str(checkpoint.ocha_id) if checkpoint.ocha_id is not None else None
        if ocha_id:
            keywords.append(checkpoint.ocha_id.lower())
        
        return list(set(keywords))  # Remove duplicates
    
    def _find_mentioned_checkpoints(self, text: str) -> List[int]:
        """Find which checkpoints are mentioned in the text"""
        text_lower = text.lower()
        mentioned = []
        
        for checkpoint_id, keywords in self.checkpoint_keywords.items():
            for keyword in keywords:
                if keyword and keyword in text_lower:
                    mentioned.append(checkpoint_id)
                    break  # Found a match for this checkpoint
        
        return list(set(mentioned))  # Remove duplicates
    
    def _infer_status(self, text: str) -> tuple[Optional[CheckpointStatus], float]:
        """
        Infer checkpoint status from text using keyword matching
        Returns: (status, confidence)
        """
        text_lower = text.lower()
        
        # Keywords for closed status (English and Arabic)
        closed_keywords = [
            "closed", "closure", "blocked", "shut", "shutdown",
            "مغلق", "إغلاق", "مقفل", "أغلق", "اغلاق"
        ]
        
        # Keywords for open status
        open_keywords = [
            "open", "opened", "accessible", "passage", "flowing",
            "مفتوح", "مفتوحة", "يعمل", "سالك"
        ]
        
        # Keywords for partial/restricted
        partial_keywords = [
            "partial", "restricted", "limited", "delays", "slow",
            "جزئي", "محدود", "تأخير", "بطيء"
        ]
        
        # Count matches
        closed_count = sum(1 for kw in closed_keywords if kw in text_lower)
        open_count = sum(1 for kw in open_keywords if kw in text_lower)
        partial_count = sum(1 for kw in partial_keywords if kw in text_lower)
        
        # Determine status based on counts
        if closed_count > open_count and closed_count > partial_count:
            return CheckpointStatus.CLOSED, min(0.6 + (closed_count * 0.1), 0.9)
        elif open_count > closed_count and open_count > partial_count:
            return CheckpointStatus.OPEN, min(0.6 + (open_count * 0.1), 0.9)
        elif partial_count > 0:
            return CheckpointStatus.PARTIAL, min(0.5 + (partial_count * 0.1), 0.8)
        else:
            return CheckpointStatus.UNKNOWN, 0.3
    
    async def collect_historical(self, channel: str, days_back: int = 7):
        """Collect historical messages from a channel"""
        try:
            logger.info(f"Collecting historical data from {channel} ({days_back} days)")
            
            limit_date = datetime.utcnow() - timedelta(days=days_back)
            messages_collected = 0
            messages_saved = 0
            
            if not self.client:
                return
            
            async for message in self.client.iter_messages(channel, limit=1000):
                if message.date < limit_date:
                    break
                
                if not message.text:
                    continue
                
                messages_collected += 1
                
                # Find mentioned checkpoints
                mentioned = self._find_mentioned_checkpoints(message.text)
                
                if mentioned:
                    # Save message for each mentioned checkpoint
                    for checkpoint_id in mentioned:
                        saved = await self._save_message(message, channel, checkpoint_id)
                        if saved:
                            messages_saved += 1
            
            logger.info(f"Channel {channel}: collected {messages_collected}, saved {messages_saved}")
            
        except Exception as e:
            logger.error(f"Error collecting from {channel}: {str(e)}")
    
    async def _save_message(self, message: Message, channel: str, checkpoint_id: int) -> bool:
        """Save a message to the database"""
        try:
            with get_db_context() as db:
                # Check if message already exists
                source_id = f"telegram_{channel}_{message.id}"
                existing = db.query(SocialMediaPost).filter(
                    SocialMediaPost.source_id == source_id
                ).first()
                
                if existing:
                    return False
                
                # Infer status from text
                msg_text = str(getattr(message, 'text', '') or getattr(message, 'message', ''))
                inferred_status, confidence = self._infer_status(msg_text)
                
                # Detect language (basic detection)
                language = self._detect_language(msg_text)
                
                # Create post record
                post = SocialMediaPost(
                    checkpoint_id=checkpoint_id,
                    source=SourceType.TELEGRAM,
                    source_id=source_id,
                    text=msg_text[:10000],  # Limit text length
                    language=language,
                    author=channel,
                    likes=message.forwards or 0,
                    url=f"https://t.me/{channel}/{message.id}",
                    posted_at=message.date,
                    collected_at=datetime.utcnow(),
                    inferred_status=inferred_status,
                    confidence=confidence,
                    processed=False
                )
                
                db.add(post)
                db.commit()
                
                logger.debug(f"Saved message from {channel} mentioning checkpoint {checkpoint_id}")
                return True
                
        except Exception as e:
            logger.error(f"Error saving message: {str(e)}")
            return False
    
    def _detect_language(self, text: str) -> str:
        """Basic language detection"""
        # Check for Arabic characters
        arabic_pattern = re.compile(r'[\u0600-\u06FF]')
        if arabic_pattern.search(text):
            return "ar"
        
        # Check for Hebrew characters
        hebrew_pattern = re.compile(r'[\u0590-\u05FF]')
        if hebrew_pattern.search(text):
            return "he"
        
        return "en"
    
    async def start_monitoring(self):
        """Start real-time monitoring of channels"""
        logger.info(f"Starting real-time monitoring of {len(CHANNELS)} channels")
        
        if not self.client:
            logger.error("Client not initialized")
            return
        
        @self.client.on(events.NewMessage(chats=CHANNELS))
        async def handler(event):
            """Handle new messages"""
            msg_text = str(getattr(event.message, 'text', '') or getattr(event.message, 'message', ''))
            if not msg_text:
                return
            
            # Find mentioned checkpoints
            mentioned = self._find_mentioned_checkpoints(msg_text)
            
            if mentioned:
                logger.info(f"New message mentioning {len(mentioned)} checkpoints")
                
                for checkpoint_id in mentioned:
                    await self._save_message(
                        event.message,
                        event.chat.username or str(event.chat_id),
                        checkpoint_id
                    )
        
        logger.info("Real-time monitoring started. Press Ctrl+C to stop.")
        if self.client:
            await self.client.run_until_disconnected()  # type: ignore
    
    async def disconnect(self):
        """Disconnect from Telegram"""
        if self.client and hasattr(self.client, 'disconnect'):
            await self.client.disconnect()  # type: ignore
            logger.info("Telegram client disconnected")


async def main():
    """Main function"""
    collector = TelegramCollector()
    
    try:
        await collector.initialize()
        
        # Collect historical data first
        logger.info("Starting historical data collection...")
        for channel in CHANNELS:
            if channel.strip():
                await collector.collect_historical(channel.strip(), days_back=7)
        
        logger.info("Historical collection complete. Starting real-time monitoring...")
        
        # Start real-time monitoring
        await collector.start_monitoring()
        
    except KeyboardInterrupt:
        logger.info("Shutting down...")
    except Exception as e:
        logger.error(f"Error: {str(e)}", exc_info=True)
    finally:
        await collector.disconnect()


if __name__ == "__main__":
    asyncio.run(main())
