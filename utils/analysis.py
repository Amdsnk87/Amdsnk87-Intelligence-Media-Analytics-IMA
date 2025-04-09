import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Any, Tuple, Optional
import re
from collections import Counter
from textblob import TextBlob
import streamlit as st
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import string
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from dateutil.parser import parse as date_parse
import matplotlib.pyplot as plt

# Download nltk resources
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
except:
    print("Could not download NLTK resources")

# Download spaCy model if needed
try:
    import spacy.cli
    spacy.cli.download("en_core_web_sm")
except:
    print("Could not download spaCy model")

# Set Indonesian stopwords
INDONESIAN_STOPWORDS = [
    "yang", "dan", "di", "dengan", "untuk", "dari", "ini", "dalam", "pada", "adalah", "ke", "tidak",
    "ada", "itu", "juga", "oleh", "saya", "kita", "akan", "atau", "bisa", "tersebut", "bahwa", "mereka",
    "dia", "jika", "saat", "sudah", "dua", "karena", "semua", "tahun", "banyak", "tak", "pun", "kami"
]

def preprocess_text(text: str) -> str:
    """
    Preprocess text for analysis: lowercase, remove punctuation, remove numbers, etc.
    
    Args:
        text: Input text
        
    Returns:
        Preprocessed text
    """
    if not text or not isinstance(text, str):
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r'http\S+', '', text)
    
    # Remove mentions (@username)
    text = re.sub(r'@\w+', '', text)
    
    # Remove hashtags as symbols but keep the words
    text = re.sub(r'#(\w+)', r'\1', text)
    
    # Remove punctuation
    text = ''.join([c for c in text if c not in string.punctuation])
    
    # Remove numbers
    text = re.sub(r'\d+', '', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def get_sentiment(text: str, lang: str = 'en') -> Dict[str, float]:
    """
    Analyze sentiment in text.
    
    Args:
        text: Text to analyze
        lang: Language code (en or id)
        
    Returns:
        Dictionary with polarity and subjectivity scores
    """
    if not text or not isinstance(text, str):
        return {"polarity": 0.0, "subjectivity": 0.0}
    
    # Preprocess text
    clean_text = preprocess_text(text)
    
    if not clean_text:
        return {"polarity": 0.0, "subjectivity": 0.0}
    
    # If Indonesian text, apply some basic translation of sentiment words to help TextBlob
    if lang == 'id':
        # Map of Indonesian sentiment words to English
        id_to_en_sentiment = {
            'bagus': 'good', 'baik': 'good', 'hebat': 'great', 'buruk': 'bad',
            'jelek': 'bad', 'indah': 'beautiful', 'senang': 'happy', 'sedih': 'sad',
            'marah': 'angry', 'kesal': 'upset', 'puas': 'satisfied', 'kecewa': 'disappointed',
            'sukses': 'success', 'gagal': 'fail', 'berhasil': 'succeed', 'bangga': 'proud',
            'takut': 'afraid', 'berani': 'brave', 'positif': 'positive', 'negatif': 'negative'
        }
        
        # Replace Indonesian sentiment words with English equivalents
        for id_word, en_word in id_to_en_sentiment.items():
            clean_text = re.sub(r'\b' + id_word + r'\b', en_word, clean_text)
    
    # Analyze sentiment with TextBlob
    analysis = TextBlob(clean_text)
    
    return {
        "polarity": float(analysis.sentiment.polarity),
        "subjectivity": float(analysis.sentiment.subjectivity)
    }

def analyze_news_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    """
    Analyze sentiment in news articles.
    
    Args:
        df: DataFrame containing news articles
        
    Returns:
        DataFrame with sentiment analysis results
    """
    if df.empty:
        return df
    
    # Create copies of dataframe columns to avoid SettingWithCopyWarning
    df = df.copy()
    
    # Add sentiment analysis columns
    sentiment_results = []
    
    for idx, row in df.iterrows():
        # Combine title and content for analysis
        text = f"{row.get('title', '')} {row.get('content', '')}"
        
        # Determine language (assume Indonesian if source is Indonesian)
        lang = 'id' if any(domain in str(row.get('source', '')).lower() for domain in ['kompas', 'detik', 'tempo', 'indonesia']) else 'en'
        
        sentiment = get_sentiment(text, lang)
        sentiment_results.append(sentiment)
    
    # Add sentiment results to DataFrame
    df['sentiment_polarity'] = [result['polarity'] for result in sentiment_results]
    df['sentiment_subjectivity'] = [result['subjectivity'] for result in sentiment_results]
    
    # Add sentiment category
    df['sentiment_category'] = df['sentiment_polarity'].apply(
        lambda x: 'positive' if x > 0.1 else ('negative' if x < -0.1 else 'neutral')
    )
    
    return df

def analyze_social_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    """
    Analyze sentiment in social media posts.
    
    Args:
        df: DataFrame containing social media posts
        
    Returns:
        DataFrame with sentiment analysis results
    """
    if df.empty:
        return df
    
    # Create copies of dataframe columns to avoid SettingWithCopyWarning
    df = df.copy()
    
    # Add sentiment analysis columns
    sentiment_results = []
    
    for idx, row in df.iterrows():
        # Get content for analysis
        text = row.get('content', '')
        
        # Determine language (assume Indonesian by default for Indonesian media platform)
        lang = 'id'
        
        sentiment = get_sentiment(text, lang)
        sentiment_results.append(sentiment)
    
    # Add sentiment results to DataFrame
    df['sentiment_polarity'] = [result['polarity'] for result in sentiment_results]
    df['sentiment_subjectivity'] = [result['subjectivity'] for result in sentiment_results]
    
    # Add sentiment category
    df['sentiment_category'] = df['sentiment_polarity'].apply(
        lambda x: 'positive' if x > 0.1 else ('negative' if x < -0.1 else 'neutral')
    )
    
    return df

@st.cache_data(ttl=3600)  # Cache for 1 hour
def extract_keywords(df: pd.DataFrame, column: str, top_n: int = 20) -> List[Dict[str, Any]]:
    """
    Extract the most common keywords from a text column.
    
    Args:
        df: DataFrame containing text data
        column: Column name containing text
        top_n: Number of top keywords to return
        
    Returns:
        List of dictionaries with keyword and count
    """
    if df.empty or column not in df.columns:
        return []
    
    # Combine all text in the specified column
    combined_text = ' '.join(df[column].fillna('').astype(str))
    
    # Preprocess the text
    clean_text = preprocess_text(combined_text)
    
    # Try to tokenize with NLTK, fallback to simple split if it fails
    try:
        tokens = word_tokenize(clean_text)
    except Exception as e:
        print(f"NLTK tokenization failed: {str(e)}")
        # Simple fallback tokenization
        tokens = clean_text.split()
    
    # Remove stopwords (English and Indonesian)
    try:
        english_stopwords = set(stopwords.words('english'))
    except:
        # Fallback if NLTK stopwords fail
        english_stopwords = set(['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 
                              'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 
                              'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 
                              'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', 
                              'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 
                              'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 
                              'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 
                              'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 
                              'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 
                              'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 
                              'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 
                              'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 
                              'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 
                              't', 'can', 'will', 'just', 'don', 'should', 'now'])
        
    stop_words = english_stopwords.union(set(INDONESIAN_STOPWORDS))
    filtered_tokens = [word for word in tokens if word.lower() not in stop_words and len(word) > 2]
    
    # Count the frequency of each token
    counter = Counter(filtered_tokens)
    
    # Get the top N keywords
    top_keywords = counter.most_common(top_n)
    
    # Convert to list of dictionaries
    return [{"keyword": keyword, "count": count} for keyword, count in top_keywords]

def analyze_trends(news_df: pd.DataFrame, social_df: pd.DataFrame, time_period: str = 'day') -> Tuple[Dict[str, List[Dict[str, Any]]], Dict[str, List[Dict[str, Any]]]]:
    """
    Analyze trends in news and social media over time.
    
    Args:
        news_df: DataFrame containing news articles
        social_df: DataFrame containing social media posts
        time_period: Time period for grouping ('day', 'week', 'hour')
        
    Returns:
        Tuple of two dictionaries containing trend data for news and social media
    """
    news_trends = {}
    social_trends = {}
    
    # Process news data if not empty
    if not news_df.empty and 'published_at' in news_df.columns:
        # Convert published_at to datetime
        news_df = news_df.copy()
        news_df['published_at'] = pd.to_datetime(news_df['published_at'], errors='coerce')
        
        # Drop rows with invalid dates
        news_df = news_df.dropna(subset=['published_at'])
        
        if not news_df.empty:
            # Group by time period
            if time_period == 'day':
                news_df['period'] = news_df['published_at'].dt.date
            elif time_period == 'week':
                news_df['period'] = news_df['published_at'].dt.to_period('W').apply(lambda x: x.start_time.date())
            elif time_period == 'hour':
                news_df['period'] = news_df['published_at'].dt.floor('H')
            
            # Count articles per period
            volume_by_period = news_df.groupby('period').size().reset_index(name='count')
            news_trends['volume'] = volume_by_period.to_dict('records')
            
            # Group by category
            if 'category' in news_df.columns:
                category_by_period = news_df.groupby(['period', 'category']).size().reset_index(name='count')
                news_trends['category'] = category_by_period.to_dict('records')
            
            # Group by sentiment
            if 'sentiment_category' in news_df.columns:
                sentiment_by_period = news_df.groupby(['period', 'sentiment_category']).size().reset_index(name='count')
                news_trends['sentiment'] = sentiment_by_period.to_dict('records')
            
            # Group by source
            if 'source' in news_df.columns:
                source_by_period = news_df.groupby(['period', 'source']).size().reset_index(name='count')
                news_trends['source'] = source_by_period.to_dict('records')
            
            # Group by region
            if 'region' in news_df.columns:
                region_by_period = news_df.groupby(['period', 'region']).size().reset_index(name='count')
                news_trends['region'] = region_by_period.to_dict('records')
    
    # Process social media data if not empty
    if not social_df.empty and 'posted_at' in social_df.columns:
        # Convert posted_at to datetime
        social_df = social_df.copy()
        social_df['posted_at'] = pd.to_datetime(social_df['posted_at'], errors='coerce')
        
        # Drop rows with invalid dates
        social_df = social_df.dropna(subset=['posted_at'])
        
        if not social_df.empty:
            # Group by time period
            if time_period == 'day':
                social_df['period'] = social_df['posted_at'].dt.date
            elif time_period == 'week':
                social_df['period'] = social_df['posted_at'].dt.to_period('W').apply(lambda x: x.start_time.date())
            elif time_period == 'hour':
                social_df['period'] = social_df['posted_at'].dt.floor('H')
            
            # Count posts per period
            volume_by_period = social_df.groupby('period').size().reset_index(name='count')
            social_trends['volume'] = volume_by_period.to_dict('records')
            
            # Group by platform
            if 'platform' in social_df.columns:
                platform_by_period = social_df.groupby(['period', 'platform']).size().reset_index(name='count')
                social_trends['platform'] = platform_by_period.to_dict('records')
            
            # Group by sentiment
            if 'sentiment_category' in social_df.columns:
                sentiment_by_period = social_df.groupby(['period', 'sentiment_category']).size().reset_index(name='count')
                social_trends['sentiment'] = sentiment_by_period.to_dict('records')
            
            # Group by category
            if 'category' in social_df.columns:
                category_by_period = social_df.groupby(['period', 'category']).size().reset_index(name='count')
                social_trends['category'] = category_by_period.to_dict('records')
            
            # Group by engagement
            if all(col in social_df.columns for col in ['likes', 'retweets']):
                # Calculate total engagement per period
                engagement_df = social_df.groupby('period').agg({
                    'likes': 'sum',
                    'retweets': 'sum'
                }).reset_index()
                
                # Add total engagement column
                engagement_df['total_engagement'] = engagement_df['likes'] + engagement_df['retweets']
                
                social_trends['engagement'] = engagement_df.to_dict('records')
    
    return news_trends, social_trends

def identify_influencers(social_df: pd.DataFrame, top_n: int = 10) -> List[Dict[str, Any]]:
    """
    Identify top social media influencers based on followers and engagement.
    
    Args:
        social_df: DataFrame containing social media posts
        top_n: Number of top influencers to return
        
    Returns:
        List of dictionaries with influencer data
    """
    if social_df.empty or 'username' not in social_df.columns:
        return []
    
    # Create a copy to avoid warnings
    df = social_df.copy()
    
    # Calculate engagement metrics
    if 'user_followers' in df.columns:
        # Group by username and aggregate metrics
        influencers = df.groupby('username').agg({
            'user_followers': 'first',  # Get follower count
            'content': 'count',  # Count posts
            'likes': 'sum',  # Sum likes
        }).reset_index()
        
        # Add retweets/shares if available
        if 'retweets' in df.columns:
            retweets = df.groupby('username')['retweets'].sum().reset_index()
            influencers = influencers.merge(retweets, on='username', how='left')
        
        # Calculate total engagement (likes + retweets/shares)
        engagement_cols = ['likes']
        if 'retweets' in influencers.columns:
            engagement_cols.append('retweets')
        
        influencers['total_engagement'] = influencers[engagement_cols].sum(axis=1)
        
        # Calculate engagement rate (engagement / followers)
        influencers['engagement_rate'] = influencers['total_engagement'] / influencers['user_followers'].replace(0, 1)
        
        # Sort by total engagement 
        influencers = influencers.sort_values('total_engagement', ascending=False)
        
        # Get top N influencers
        top_influencers = influencers.head(top_n).to_dict('records')
        
        return top_influencers
    
    return []

@st.cache_data(ttl=3600)  # Cache for 1 hour
def topic_modeling(df: pd.DataFrame, text_column: str, n_topics: int = 5, n_words: int = 10) -> List[Dict[str, Any]]:
    """
    Perform topic modeling on text data.
    
    Args:
        df: DataFrame containing text data
        text_column: Column name containing text
        n_topics: Number of topics to extract
        n_words: Number of words per topic
        
    Returns:
        List of dictionaries with topic words and weights
    """
    if df.empty or text_column not in df.columns:
        return []
    
    # Prepare texts
    texts = df[text_column].fillna('').astype(str).apply(preprocess_text).tolist()
    
    if not texts or all(not text for text in texts):
        return []
        
    # Make sure texts are properly cleaned and sanitized
    texts = [text if isinstance(text, str) and text.strip() else "" for text in texts]
    
    try:
        # TF-IDF vectorization
        try:
            english_stopwords = list(stopwords.words('english'))
        except:
            # Fallback if NLTK stopwords fail
            english_stopwords = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 
                              'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 
                              'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 
                              'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', 
                              'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 
                              'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 
                              'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 
                              'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 
                              'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 
                              'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 
                              'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 
                              'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 
                              'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 
                              't', 'can', 'will', 'just', 'don', 'should', 'now']

        # Combine English and Indonesian stopwords
        combined_stopwords = english_stopwords + INDONESIAN_STOPWORDS
        
        vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words=combined_stopwords,
            min_df=2
        )
        
        X = vectorizer.fit_transform(texts)
        
        if X.shape[1] == 0:  # No features extracted
            return []
        
        # LDA for topic modeling
        lda = LatentDirichletAllocation(
            n_components=n_topics,
            max_iter=10,
            learning_method='online',
            random_state=42
        )
        
        lda.fit(X)
        
        # Get feature names
        feature_names = vectorizer.get_feature_names_out()
        
        # Extract topics
        topics = []
        for topic_idx, topic in enumerate(lda.components_):
            top_words_idx = topic.argsort()[:-n_words-1:-1]
            top_words = [feature_names[i] for i in top_words_idx]
            top_weights = [float(topic[i]) for i in top_words_idx]
            
            topics.append({
                "id": topic_idx,
                "words": top_words,
                "weights": top_weights
            })
        
        return topics
    
    except Exception as e:
        print(f"Error in topic modeling: {str(e)}")
        return []

def calculate_media_share(news_df: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    Calculate media share (percentage) by source.
    
    Args:
        news_df: DataFrame containing news articles
        
    Returns:
        List of dictionaries with source and share percentage
    """
    if news_df.empty or 'source' not in news_df.columns:
        return []
    
    # Count articles by source
    source_counts = news_df['source'].value_counts().reset_index()
    source_counts.columns = ['source', 'count']
    
    # Calculate percentage
    total_articles = source_counts['count'].sum()
    source_counts['percentage'] = (source_counts['count'] / total_articles * 100).round(2)
    
    # Convert to list of dictionaries
    return source_counts.to_dict('records')

@st.cache_data(ttl=3600)  # Cache for 1 hour
def create_word_cloud_data(df: pd.DataFrame, text_column: str, max_words: int = 100) -> List[Dict[str, Any]]:
    """
    Create data for a word cloud visualization.
    
    Args:
        df: DataFrame containing text data
        text_column: Column name containing text
        max_words: Maximum number of words in the cloud
        
    Returns:
        List of dictionaries with word and weight
    """
    if df.empty or text_column not in df.columns:
        return []
    
    # Combine all text
    combined_text = ' '.join(df[text_column].fillna('').astype(str))
    
    # Preprocess
    clean_text = preprocess_text(combined_text)
    
    # Try to tokenize with NLTK, fallback to simple split if it fails
    try:
        tokens = word_tokenize(clean_text)
    except Exception as e:
        print(f"NLTK tokenization failed in word cloud: {str(e)}")
        # Simple fallback tokenization
        tokens = clean_text.split()
    
    # Remove stopwords
    try:
        english_stopwords = set(stopwords.words('english'))
    except:
        # Fallback if NLTK stopwords fail
        english_stopwords = set(['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 
                              'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 
                              'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 
                              'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', 
                              'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 
                              'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 
                              'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 
                              'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 
                              'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 
                              'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 
                              'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 
                              'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 
                              'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 
                              't', 'can', 'will', 'just', 'don', 'should', 'now'])
        
    stop_words = english_stopwords.union(set(INDONESIAN_STOPWORDS))
    filtered_tokens = [word for word in tokens if word.lower() not in stop_words and len(word) > 2]
    
    # Count frequencies
    counter = Counter(filtered_tokens)
    
    # Get top words
    top_words = counter.most_common(max_words)
    
    # Calculate weights (for visualization sizing)
    max_count = max([count for _, count in top_words]) if top_words else 1
    
    # Create result
    result = [
        {"text": word, "value": count, "weight": (count / max_count) * 100}
        for word, count in top_words
    ]
    
    return result

def search_content(df: pd.DataFrame, query: str, columns: List[str]) -> pd.DataFrame:
    """
    Search for content matching a query in specified columns.
    
    Args:
        df: DataFrame to search
        query: Search query
        columns: Columns to search in
        
    Returns:
        DataFrame with matching rows
    """
    if df.empty or not query or not columns:
        return df
    
    # Make sure columns exist in the DataFrame
    valid_columns = [col for col in columns if col in df.columns]
    
    if not valid_columns:
        return df.head(0)  # Return empty DataFrame with the same structure
    
    # Create pattern for case-insensitive search
    pattern = re.compile(f'.*{re.escape(query)}.*', re.IGNORECASE)
    
    # Initialize mask
    mask = pd.Series(False, index=df.index)
    
    # Search in each column
    for column in valid_columns:
        column_mask = df[column].fillna('').astype(str).str.contains(pattern, regex=True)
        mask = mask | column_mask
    
    # Return matching rows
    return df[mask]

def media_coverage_timeline(news_df: pd.DataFrame, keywords: List[str], 
                           start_date: datetime, end_date: datetime) -> Dict[str, List[Dict[str, Any]]]:
    """
    Create timeline data for media coverage of specific keywords.
    
    Args:
        news_df: DataFrame with news articles
        keywords: List of keywords to track
        start_date: Start date for analysis
        end_date: End date for analysis
        
    Returns:
        Dictionary with timeline data for each keyword
    """
    if news_df.empty or 'published_at' not in news_df.columns:
        return {keyword: [] for keyword in keywords}
    
    # Create a copy of the DataFrame
    df = news_df.copy()
    
    # Convert published_at to datetime
    df['published_at'] = pd.to_datetime(df['published_at'], errors='coerce')
    
    # Filter by date range
    df = df[(df['published_at'] >= pd.Timestamp(start_date)) & 
            (df['published_at'] <= pd.Timestamp(end_date))]
    
    if df.empty:
        return {keyword: [] for keyword in keywords}
    
    # Add date column
    df['date'] = df['published_at'].dt.date
    
    # Create combined text column for searching
    df['combined_text'] = df['title'].fillna('') + ' ' + df['content'].fillna('')
    
    # Initialize result
    result = {}
    
    # Track coverage for each keyword
    for keyword in keywords:
        # Create mask for articles containing the keyword
        pattern = re.compile(f'.*{re.escape(keyword)}.*', re.IGNORECASE)
        mask = df['combined_text'].str.contains(pattern, regex=True)
        
        # Count articles by date
        timeline = df[mask].groupby('date').size().reset_index(name='count')
        
        # Convert to list of dictionaries
        result[keyword] = timeline.to_dict('records')
    
    return result
