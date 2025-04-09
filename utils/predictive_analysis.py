import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation, NMF
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import warnings
warnings.filterwarnings("ignore")

def identify_emerging_topics(df, text_column='content', date_column='published_at', 
                           n_topics=5, n_words=10, window_days=30):
    """
    Identify emerging topics in recent content compared to older content.
    
    Args:
        df: DataFrame containing text data
        text_column: Column containing text to analyze
        date_column: Column containing dates
        n_topics: Number of topics to identify
        n_words: Number of words per topic
        window_days: Number of days to consider recent
        
    Returns:
        Dictionary with emerging topics and trend strength
    """
    if df.empty or text_column not in df.columns or date_column not in df.columns:
        return []
    
    # Ensure date is in datetime format
    df[date_column] = pd.to_datetime(df[date_column])
    
    # Split into recent and older content
    latest_date = df[date_column].max()
    cutoff_date = latest_date - timedelta(days=window_days)
    
    recent_df = df[df[date_column] >= cutoff_date]
    older_df = df[df[date_column] < cutoff_date]
    
    # If either dataset is too small, return empty
    if len(recent_df) < 10 or len(older_df) < 10:
        return []
    
    # Extract topics from recent content
    recent_topics = extract_topics_lda(recent_df, text_column, n_topics, n_words)
    
    # Extract topics from older content
    older_topics = extract_topics_lda(older_df, text_column, n_topics, n_words)
    
    # Compare topics and identify emerging ones
    emerging_topics = []
    for topic_idx, topic in enumerate(recent_topics):
        # Check topic similarity with older topics
        max_similarity = max([topic_similarity(topic['words'], old_topic['words']) 
                             for old_topic in older_topics])
        
        # If topic is new or growing
        if max_similarity < 0.3:  # Low similarity means new topic
            topic['trend'] = 'new'
            topic['strength'] = 0.9  # High strength for new topics
            emerging_topics.append(topic)
        elif 0.3 <= max_similarity < 0.7:  # Medium similarity means evolving topic
            topic['trend'] = 'evolving'
            topic['strength'] = 0.7  # Medium strength for evolving topics
            emerging_topics.append(topic)
    
    # Sort by strength
    emerging_topics = sorted(emerging_topics, key=lambda x: x['strength'], reverse=True)
    
    return emerging_topics[:n_topics]

def extract_topics_lda(df, text_column, n_topics=5, n_words=10):
    """
    Extract topics using Latent Dirichlet Allocation.
    
    Args:
        df: DataFrame containing text data
        text_column: Column containing text to analyze
        n_topics: Number of topics to extract
        n_words: Number of words per topic
        
    Returns:
        List of dictionaries with topic words and weights
    """
    if df.empty or text_column not in df.columns or len(df) < 5:
        return []
    
    # Fill NaN values
    df[text_column] = df[text_column].fillna('')
    
    # Vectorize text
    tfidf_vectorizer = TfidfVectorizer(
        max_df=0.95, 
        min_df=2,
        max_features=1000,
        stop_words='english'
    )
    
    try:
        tfidf = tfidf_vectorizer.fit_transform(df[text_column])
        
        # Apply LDA
        lda = LatentDirichletAllocation(
            n_components=n_topics,
            max_iter=10,
            learning_method='online',
            random_state=42
        )
        
        lda.fit(tfidf)
        
        # Get feature names
        feature_names = tfidf_vectorizer.get_feature_names_out()
        
        # Extract topics
        topics = []
        for topic_idx, topic in enumerate(lda.components_):
            top_words_idx = topic.argsort()[:-n_words-1:-1]
            top_words = [feature_names[i] for i in top_words_idx]
            top_weights = [topic[i] for i in top_words_idx]
            
            topics.append({
                'id': topic_idx,
                'words': top_words,
                'weights': top_weights
            })
        
        return topics
    
    except Exception as e:
        print(f"Error in topic extraction: {e}")
        return []

def topic_similarity(words1, words2):
    """
    Calculate similarity between two topics based on word overlap.
    
    Args:
        words1: List of words in first topic
        words2: List of words in second topic
        
    Returns:
        Similarity score (0-1)
    """
    set1 = set(words1)
    set2 = set(words2)
    
    # Jaccard similarity
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    
    return intersection / union if union > 0 else 0

def predict_topic_trends(df, topic_words, date_column='published_at', 
                        prediction_days=7, smoothing=True):
    """
    Predict future trend for a specific topic.
    
    Args:
        df: DataFrame containing text data
        topic_words: List of words defining the topic
        date_column: Column with date information
        prediction_days: Number of days to predict ahead
        smoothing: Whether to apply smoothing
        
    Returns:
        Dictionary with prediction results and data for visualization
    """
    if df.empty or date_column not in df.columns:
        return None
    
    # Ensure date is in datetime format
    df[date_column] = pd.to_datetime(df[date_column])
    
    # Create a time series for the topic
    # Count documents containing topic words by day
    topic_series = create_topic_time_series(df, topic_words, date_column)
    
    if topic_series.empty or len(topic_series) < 5:
        return None
    
    # Apply smoothing if requested
    if smoothing and len(topic_series) >= 7:
        topic_series = topic_series.rolling(window=3, center=True).mean().dropna()
    
    # Ensure we have enough data
    if len(topic_series) < 5:
        return None
    
    # Prepare data for forecasting
    try:
        # Try ARIMA forecasting
        prediction_result = arima_forecast(topic_series, prediction_days)
        
        # If ARIMA fails, try exponential smoothing
        if prediction_result is None:
            prediction_result = exp_smoothing_forecast(topic_series, prediction_days)
        
        # If that also fails, try linear regression
        if prediction_result is None:
            prediction_result = linear_regression_forecast(topic_series, prediction_days)
        
        return prediction_result
    
    except Exception as e:
        print(f"Error in trend prediction: {e}")
        return None

def create_topic_time_series(df, topic_words, date_column):
    """
    Create a time series for a topic based on word frequency.
    
    Args:
        df: DataFrame with articles
        topic_words: Words defining the topic
        date_column: Date column name
        
    Returns:
        Time series of topic frequency
    """
    # Ensure date is in datetime format
    df[date_column] = pd.to_datetime(df[date_column])
    
    # Function to check if any topic word is in the content
    def contains_topic(text):
        if not isinstance(text, str):
            return 0
        return any(word.lower() in text.lower() for word in topic_words)
    
    # Apply function to all text columns
    text_columns = ['title', 'content', 'description']
    contains_topic_words = False
    
    for col in text_columns:
        if col in df.columns:
            df[f'contains_{col}'] = df[col].apply(contains_topic)
            contains_topic_words = contains_topic_words | df[f'contains_{col}']
    
    if not contains_topic_words.any():
        return pd.Series()
    
    # Mark articles that contain any topic word
    df['contains_topic'] = contains_topic_words
    
    # Group by date and count articles with topic
    topic_by_date = df.groupby(df[date_column].dt.date)['contains_topic'].sum()
    
    # Fill missing dates with zeros
    date_range = pd.date_range(start=topic_by_date.index.min(), end=topic_by_date.index.max())
    topic_series = topic_by_date.reindex(date_range, fill_value=0)
    
    return topic_series

def arima_forecast(series, prediction_days=7):
    """
    Use ARIMA model for forecasting.
    
    Args:
        series: Time series data
        prediction_days: Days to forecast
        
    Returns:
        Dictionary with prediction results
    """
    try:
        # Define model with appropriate parameters
        model = ARIMA(series, order=(5,1,0))
        model_fit = model.fit()
        
        # Make prediction
        forecast = model_fit.forecast(steps=prediction_days)
        
        # Generate dates for prediction
        last_date = series.index[-1]
        forecast_dates = pd.date_range(start=last_date + timedelta(days=1), periods=prediction_days)
        
        # Create forecast series
        forecast_series = pd.Series(forecast, index=forecast_dates)
        
        # Calculate trend direction and strength
        recent_avg = series[-7:].mean() if len(series) >= 7 else series.mean()
        forecast_avg = forecast.mean()
        
        trend_change = ((forecast_avg - recent_avg) / recent_avg) * 100 if recent_avg > 0 else 0
        
        # Determine trend direction
        if trend_change > 10:
            trend = "rising"
        elif trend_change < -10:
            trend = "falling"
        else:
            trend = "stable"
        
        # Calculate trend strength (0-1)
        trend_strength = min(abs(trend_change) / 50, 1.0)
        
        return {
            'model': 'ARIMA',
            'forecast': forecast,
            'forecast_series': forecast_series,
            'historical': series,
            'trend': trend,
            'trend_change_pct': trend_change,
            'trend_strength': trend_strength
        }
    
    except Exception as e:
        print(f"ARIMA forecast error: {e}")
        return None

def exp_smoothing_forecast(series, prediction_days=7):
    """
    Use Exponential Smoothing for forecasting.
    
    Args:
        series: Time series data
        prediction_days: Days to forecast
        
    Returns:
        Dictionary with prediction results
    """
    try:
        # Define model
        model = ExponentialSmoothing(
            series,
            trend='add',
            seasonal='add',
            seasonal_periods=7
        )
        model_fit = model.fit()
        
        # Make prediction
        forecast = model_fit.forecast(prediction_days)
        
        # Generate dates for prediction
        last_date = series.index[-1]
        forecast_dates = pd.date_range(start=last_date + timedelta(days=1), periods=prediction_days)
        
        # Create forecast series
        forecast_series = pd.Series(forecast, index=forecast_dates)
        
        # Calculate trend direction and strength
        recent_avg = series[-7:].mean() if len(series) >= 7 else series.mean()
        forecast_avg = forecast.mean()
        
        trend_change = ((forecast_avg - recent_avg) / recent_avg) * 100 if recent_avg > 0 else 0
        
        # Determine trend direction
        if trend_change > 10:
            trend = "rising"
        elif trend_change < -10:
            trend = "falling"
        else:
            trend = "stable"
        
        # Calculate trend strength (0-1)
        trend_strength = min(abs(trend_change) / 50, 1.0)
        
        return {
            'model': 'Exponential Smoothing',
            'forecast': forecast,
            'forecast_series': forecast_series,
            'historical': series,
            'trend': trend,
            'trend_change_pct': trend_change,
            'trend_strength': trend_strength
        }
    
    except Exception as e:
        print(f"Exponential smoothing forecast error: {e}")
        return None

def linear_regression_forecast(series, prediction_days=7):
    """
    Use Linear Regression for simple forecasting.
    
    Args:
        series: Time series data
        prediction_days: Days to forecast
        
    Returns:
        Dictionary with prediction results
    """
    try:
        # Create features (days from start)
        X = np.array(range(len(series))).reshape(-1, 1)
        y = series.values
        
        # Fit model
        model = LinearRegression()
        model.fit(X, y)
        
        # Predict future values
        X_future = np.array(range(len(series), len(series) + prediction_days)).reshape(-1, 1)
        forecast = model.predict(X_future)
        
        # Generate dates for prediction
        last_date = series.index[-1]
        forecast_dates = pd.date_range(start=last_date + timedelta(days=1), periods=prediction_days)
        
        # Create forecast series
        forecast_series = pd.Series(forecast, index=forecast_dates)
        
        # Calculate trend direction and strength
        recent_avg = series[-7:].mean() if len(series) >= 7 else series.mean()
        forecast_avg = np.mean(forecast)
        
        trend_change = ((forecast_avg - recent_avg) / recent_avg) * 100 if recent_avg > 0 else 0
        
        # Determine trend direction
        if trend_change > 10:
            trend = "rising"
        elif trend_change < -10:
            trend = "falling"
        else:
            trend = "stable"
        
        # Calculate trend strength (0-1)
        trend_strength = min(abs(trend_change) / 50, 1.0)
        
        return {
            'model': 'Linear Regression',
            'forecast': forecast,
            'forecast_series': forecast_series,
            'historical': series,
            'trend': trend,
            'trend_change_pct': trend_change,
            'trend_strength': trend_strength
        }
    
    except Exception as e:
        print(f"Linear regression forecast error: {e}")
        return None

def identify_seasonal_patterns(df, text_column='content', date_column='published_at', 
                             topic_words=None):
    """
    Identify seasonal patterns in topic popularity.
    
    Args:
        df: DataFrame with news articles
        text_column: Column with text content
        date_column: Column with dates
        topic_words: Words defining the topic (optional)
        
    Returns:
        Dictionary with seasonal pattern information
    """
    if df.empty or date_column not in df.columns:
        return None
    
    # Ensure date is in datetime format
    df[date_column] = pd.to_datetime(df[date_column])
    
    # If no specific topic, analyze overall volume
    if topic_words is None:
        # Group by date and count articles
        articles_by_date = df.groupby(df[date_column].dt.date).size()
    else:
        # Create time series for the specific topic
        articles_by_date = create_topic_time_series(df, topic_words, date_column)
    
    # Need at least 30 days of data for seasonal analysis
    if len(articles_by_date) < 30:
        return None
    
    # Add day of week and month
    articles_df = articles_by_date.reset_index()
    articles_df.columns = ['date', 'count']
    articles_df['day_of_week'] = articles_df['date'].dt.dayofweek
    articles_df['month'] = articles_df['date'].dt.month
    
    # Analyze patterns by day of week
    day_pattern = articles_df.groupby('day_of_week')['count'].mean()
    day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    day_pattern.index = day_names
    
    # Find peak days
    max_day = day_pattern.idxmax()
    min_day = day_pattern.idxmin()
    
    # Analyze patterns by month (if enough data)
    month_pattern = None
    max_month = None
    min_month = None
    
    if len(articles_df['date'].dt.month.unique()) > 6:
        month_pattern = articles_df.groupby('month')['count'].mean()
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                      'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        month_pattern.index = month_names
        
        max_month = month_pattern.idxmax()
        min_month = month_pattern.idxmin()
    
    # Calculate variability
    day_variability = day_pattern.std() / day_pattern.mean() if day_pattern.mean() > 0 else 0
    
    return {
        'day_pattern': day_pattern.to_dict(),
        'month_pattern': month_pattern.to_dict() if month_pattern is not None else None,
        'peak_day': max_day,
        'low_day': min_day,
        'peak_month': max_month,
        'low_month': min_month,
        'day_variability': day_variability
    }