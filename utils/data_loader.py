import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional

from utils.data_collection import get_news_articles as fetch_news_articles
from utils.data_collection import get_social_media_data as fetch_social_media_data
from utils.database import (
    get_news_articles as db_get_news_articles,
    get_social_media_posts as db_get_social_media_posts,
    store_news_articles,
    store_social_media_posts,
    store_keywords,
    store_topic_model
)
from utils.analysis import (
    analyze_news_sentiment,
    analyze_social_sentiment
)

def get_data(
    start_date: datetime.date,
    end_date: datetime.date,
    news_sources: List[str],
    social_sources: List[str],
    regions: List[str],
    topics: List[str],
    force_refresh: bool = False
):
    """
    Get data from API/scraping or database based on parameters
    
    Args:
        start_date: Start date for data
        end_date: End date for data
        news_sources: List of news sources to include
        social_sources: List of social media platforms to include
        regions: List of regions to filter by
        topics: List of topics to filter by
        force_refresh: If True, fetch fresh data regardless of cache
        
    Returns:
        Tuple of news_df, social_df
    """
    # Check if data is available in database for the requested date range
    # Increased limit to 1000 articles to ensure we display more content
    news_db = db_get_news_articles(
        start_date=start_date,
        end_date=end_date,
        sources=news_sources,
        regions=regions,
        categories=topics,
        limit=1000
    )
    
    # Increased limit to 500 posts to ensure we display more content
    social_db = db_get_social_media_posts(
        start_date=start_date,
        end_date=end_date,
        platforms=social_sources,
        regions=regions,
        categories=topics,
        limit=500
    )
    
    # If we have data in database and not forcing refresh, use database data
    news_df = pd.DataFrame()
    social_df = pd.DataFrame()
    
    # For news data
    if not force_refresh and not news_db.empty:
        # Use database data
        news_df = news_db
    else:
        # Fetch fresh data
        news_df = fetch_news_articles(news_sources, start_date, end_date, topics, regions)
        
        # Analyze sentiment if needed
        if not news_df.empty and 'sentiment_category' not in news_df.columns:
            news_df = analyze_news_sentiment(news_df)
        
        # Store data in database
        if not news_df.empty:
            store_news_articles(news_df)
    
    # For social media data
    if not force_refresh and not social_db.empty:
        # Use database data
        social_df = social_db
    else:
        # Fetch fresh data
        social_df = fetch_social_media_data(social_sources, start_date, end_date, topics, regions)
        
        # Analyze sentiment if needed
        if not social_df.empty and 'sentiment_category' not in social_df.columns:
            social_df = analyze_social_sentiment(social_df)
        
        # Store data in database
        if not social_df.empty:
            store_social_media_posts(social_df)
    
    return news_df, social_df


def store_analysis_results(news_keywords=None, social_keywords=None, topic_model=None):
    """
    Store analysis results in database
    
    Args:
        news_keywords: List of dictionaries with keyword and count for news
        social_keywords: List of dictionaries with keyword and count for social media
        topic_model: List of dictionaries with topic modeling results
    """
    # Store keywords
    if news_keywords:
        store_keywords(news_keywords, source_type='news')
    
    if social_keywords:
        store_keywords(social_keywords, source_type='social')
    
    # Store topic model
    if topic_model:
        store_topic_model(topic_model)