"""
NLP package for sentiment analysis and feature extraction
"""
from .sentiment_analyzer import (
    detect_language,
    analyze_sentiment,
    extract_status_from_text,
    process_social_media_post
)

from .feature_extractor import FeatureExtractor

__all__ = [
    "detect_language",
    "analyze_sentiment",
    "extract_status_from_text",
    "process_social_media_post",
    "FeatureExtractor",
]
