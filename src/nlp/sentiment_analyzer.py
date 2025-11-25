"""
NLP and sentiment analysis for checkpoint-related social media posts
"""
import sys
import os
from typing import Optional, Tuple
import re

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformers import pipeline, AutoTokenizer, AutoModel
import torch
from database import CheckpointStatus

# Lazy loading for models
_sentiment_analyzer = None
_arabic_sentiment_analyzer = None


def get_sentiment_analyzer():
    """Get or create sentiment analyzer for English"""
    global _sentiment_analyzer
    if _sentiment_analyzer is None:
        try:
            _sentiment_analyzer = pipeline(
                "sentiment-analysis",
                model="distilbert-base-uncased-finetuned-sst-2-english",
                device=-1  # CPU
            )
        except Exception as e:
            print(f"Warning: Could not load sentiment analyzer: {e}")
            _sentiment_analyzer = None
    return _sentiment_analyzer


def get_arabic_sentiment_analyzer():
    """Get or create sentiment analyzer for Arabic"""
    global _arabic_sentiment_analyzer
    if _arabic_sentiment_analyzer is None:
        try:
            # Using a lighter multilingual model for Arabic
            _arabic_sentiment_analyzer = pipeline(
                "sentiment-analysis",
                model="nlptown/bert-base-multilingual-uncased-sentiment",
                device=-1  # CPU
            )
        except Exception as e:
            print(f"Warning: Could not load Arabic sentiment analyzer: {e}")
            _arabic_sentiment_analyzer = None
    return _arabic_sentiment_analyzer


def detect_language(text: str) -> str:
    """
    Detect language of text (basic detection)
    
    Returns: 'ar' for Arabic, 'he' for Hebrew, 'en' for English
    """
    # Check for Arabic characters
    arabic_pattern = re.compile(r'[\u0600-\u06FF]')
    if arabic_pattern.search(text):
        return "ar"
    
    # Check for Hebrew characters
    hebrew_pattern = re.compile(r'[\u0590-\u05FF]')
    if hebrew_pattern.search(text):
        return "he"
    
    return "en"


def analyze_sentiment(text: str, language: Optional[str] = None) -> Tuple[float, float]:
    """
    Analyze sentiment of text
    
    Args:
        text: Text to analyze
        language: Optional language code ('ar', 'en', 'he'). Auto-detected if None.
        
    Returns:
        Tuple of (sentiment_score, confidence)
        sentiment_score: -1.0 (negative) to 1.0 (positive)
        confidence: 0.0 to 1.0
    """
    if not text or len(text.strip()) < 3:
        return 0.0, 0.0
    
    # Detect language if not provided
    if language is None:
        language = detect_language(text)
    
    try:
        # Truncate text if too long
        text = text[:512]
        
        if language == "ar":
            # Use Arabic/multilingual model
            analyzer = get_arabic_sentiment_analyzer()
            if analyzer:
                result = analyzer(text)
                if isinstance(result, list) and len(result) > 0:
                    item = result[0]
                    # Convert 1-5 star rating to -1 to 1 scale
                    label_text = str(item.get('label', '3'))
                    score = (float(label_text.split()[0]) - 3) / 2  # 1->-1, 3->0, 5->1
                    # Handle tensor scores properly
                    conf = item.get('score', 0.5)
                    confidence = float(conf.item()) if hasattr(conf, 'item') else float(conf)
                    return score, confidence
        else:
            # Use English model for English and Hebrew (fallback)
            analyzer = get_sentiment_analyzer()
            if analyzer:
                result = analyzer(text)
                if isinstance(result, list) and len(result) > 0:
                    item = result[0]
                    # Convert POSITIVE/NEGATIVE to -1 to 1 scale
                    score = 1.0 if item.get('label') == 'POSITIVE' else -1.0
                    # Handle tensor scores properly
                    conf = item.get('score', 0.5)
                    confidence = float(conf.item()) if hasattr(conf, 'item') else float(conf)
                    return score, confidence
        
        # Fallback to keyword-based sentiment
        return keyword_based_sentiment(text)
        
    except Exception as e:
        print(f"Error in sentiment analysis: {e}")
        return keyword_based_sentiment(text)


def keyword_based_sentiment(text: str) -> Tuple[float, float]:
    """
    Fallback keyword-based sentiment analysis
    
    Returns: (sentiment_score, confidence)
    """
    text_lower = text.lower()
    
    # Positive keywords
    positive_keywords = [
        "open", "opened", "good", "smooth", "easy", "quick", "accessible",
        "مفتوح", "جيد", "سهل", "سريع"
    ]
    
    # Negative keywords
    negative_keywords = [
        "closed", "blocked", "bad", "difficult", "slow", "restricted", "denied",
        "مغلق", "سيء", "صعب", "بطيء", "ممنوع"
    ]
    
    positive_count = sum(1 for kw in positive_keywords if kw in text_lower)
    negative_count = sum(1 for kw in negative_keywords if kw in text_lower)
    
    total = positive_count + negative_count
    
    if total == 0:
        return 0.0, 0.1  # Neutral, low confidence
    
    sentiment = (positive_count - negative_count) / total
    confidence = min(0.3 + (total * 0.1), 0.7)  # Max 0.7 for keyword-based
    
    return sentiment, confidence


def extract_status_from_text(text: str) -> Tuple[Optional[CheckpointStatus], float]:
    """
    Extract checkpoint status from text using NLP and keywords
    
    Returns:
        Tuple of (status, confidence)
    """
    if not text:
        return None, 0.0
    
    text_lower = text.lower()
    
    # Strong indicators for closed
    closed_patterns = [
        r'\b(closed|closure|shut\s+down|blocked|sealed)\b',
        r'\b(مغلق|إغلاق|مقفل)\b',
    ]
    
    # Strong indicators for open
    open_patterns = [
        r'\b(open|opened|accessible|passable|flowing)\b',
        r'\b(مفتوح|مفتوحة|يعمل)\b',
    ]
    
    # Partial/restricted indicators
    partial_patterns = [
        r'\b(partial|limited|restricted|delays|slow|queue)\b',
        r'\b(جزئي|محدود|تأخير)\b',
    ]
    
    closed_matches = sum(len(re.findall(p, text_lower)) for p in closed_patterns)
    open_matches = sum(len(re.findall(p, text_lower)) for p in open_patterns)
    partial_matches = sum(len(re.findall(p, text_lower)) for p in partial_patterns)
    
    # Use sentiment as additional signal
    sentiment, sent_conf = analyze_sentiment(text)
    
    # Decision logic
    if closed_matches > open_matches and closed_matches > 0:
        confidence = min(0.6 + (closed_matches * 0.15), 0.95)
        # Boost confidence if sentiment is negative
        if sentiment < -0.3:
            confidence = min(confidence + 0.1, 0.95)
        return CheckpointStatus.CLOSED, confidence
    
    elif open_matches > closed_matches and open_matches > 0:
        confidence = min(0.6 + (open_matches * 0.15), 0.95)
        # Boost confidence if sentiment is positive
        if sentiment > 0.3:
            confidence = min(confidence + 0.1, 0.95)
        return CheckpointStatus.OPEN, confidence
    
    elif partial_matches > 0:
        confidence = min(0.5 + (partial_matches * 0.1), 0.8)
        return CheckpointStatus.PARTIAL, confidence
    
    # Use sentiment as fallback
    elif abs(sentiment) > 0.5:
        if sentiment < 0:
            return CheckpointStatus.CLOSED, 0.4
        else:
            return CheckpointStatus.OPEN, 0.4
    
    return CheckpointStatus.UNKNOWN, 0.2


def process_social_media_post(text: str, language: Optional[str] = None) -> dict:
    """
    Process a social media post and extract all relevant information
    
    Returns:
        Dictionary with sentiment, status, and confidence scores
    """
    if language is None:
        language = detect_language(text)
    
    sentiment, sent_conf = analyze_sentiment(text, language)
    status, status_conf = extract_status_from_text(text)
    
    return {
        "language": language,
        "sentiment_score": sentiment,
        "sentiment_confidence": sent_conf,
        "inferred_status": status,
        "status_confidence": status_conf
    }


# Example usage and testing
if __name__ == "__main__":
    test_texts = [
        "Qalandiya checkpoint is closed today, major delays",
        "حاجز قلنديا مغلق اليوم، تأخير كبير",
        "Checkpoint is open and traffic is flowing smoothly",
        "Long queues at the checkpoint but it's partially open"
    ]
    
    print("Testing NLP Module:\n")
    for text in test_texts:
        result = process_social_media_post(text)
        print(f"Text: {text}")
        print(f"  Language: {result['language']}")
        print(f"  Sentiment: {result['sentiment_score']:.2f} (conf: {result['sentiment_confidence']:.2f})")
        print(f"  Status: {result['inferred_status']} (conf: {result['status_confidence']:.2f})")
        print()
