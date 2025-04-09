"""
Predictive analytics module for trend prediction and analysis.
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Any, Tuple, Optional
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
import warnings
warnings.filterwarnings('ignore')

@st.cache_data(ttl=3600)  # Cache for 1 hour
def aggregate_time_series(
    df: pd.DataFrame, 
    time_column: str, 
    groupby_column: str = None, 
    freq: str = 'D',
    min_periods: int = 3
) -> pd.DataFrame:
    """
    Aggregate data into a time series for analysis.
    
    Args:
        df: DataFrame containing the data
        time_column: Column containing datetime data
        groupby_column: Column to group by (e.g., 'keyword', 'category')
        freq: Frequency for resampling ('D' for daily, 'H' for hourly, etc.)
        min_periods: Minimum number of periods required for analysis
        
    Returns:
        DataFrame with time series data
    """
    # Ensure datetime format
    df[time_column] = pd.to_datetime(df[time_column])
    df = df.sort_values(time_column)
    
    # Group by time and optional column
    if groupby_column:
        # Count occurrences by time and groupby column
        grouped = df.groupby([pd.Grouper(key=time_column, freq=freq), groupby_column]).size().reset_index()
        grouped.columns = [time_column, groupby_column, 'count']
        
        # Pivot to get time series for each value in groupby_column
        pivoted = grouped.pivot(index=time_column, columns=groupby_column, values='count')
        pivoted = pivoted.fillna(0)
        
        # Keep only columns with sufficient data points
        keep_cols = [col for col in pivoted.columns if pivoted[col].sum() >= min_periods]
        pivoted = pivoted[keep_cols]
        
        return pivoted
    else:
        # Simple time series without grouping
        time_series = df.groupby(pd.Grouper(key=time_column, freq=freq)).size()
        time_series = pd.DataFrame(time_series, columns=['count'])
        return time_series

@st.cache_data(ttl=3600)  # Cache for 1 hour
def predict_trend_linear(
    time_series: pd.DataFrame, 
    column: str = 'count',
    days_to_predict: int = 7
) -> Tuple[pd.DataFrame, float, float]:
    """
    Predict future trend using linear regression.
    
    Args:
        time_series: DataFrame with time series data
        column: Column name to predict
        days_to_predict: Number of days to predict into the future
        
    Returns:
        Tuple of (DataFrame with predictions, RMSE, R²)
    """
    # Prepare data
    df = time_series.reset_index()
    if column not in df.columns and len(df.columns) == 2:
        # If the column doesn't exist but there are only 2 columns (index and value)
        column = df.columns[1]
        
    # Create features: convert dates to numeric (days since first date)
    first_date = df.iloc[0, 0]
    df['date_numeric'] = (df.iloc[:, 0] - first_date).dt.total_seconds() / (24 * 3600)
    
    # Split data
    X = df[['date_numeric']]
    y = df[column]
    
    # Handle small datasets - if less than 4 data points, don't split
    if len(df) < 4:
        X_train, y_train = X, y
        X_test, y_test = X, y
        # Default metrics since we can't properly evaluate
        rmse = 0.0
        r2 = 1.0
    else:
        # Dynamic test size: smaller test size for smaller datasets
        test_size = min(0.3, 1/len(df))
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        
        # Train linear regression model
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        # Evaluate model
        y_pred = model.predict(X_test)
        rmse = mean_squared_error(y_test, y_pred, squared=False)
        r2 = r2_score(y_test, y_pred)
    
    # If we used the entire dataset for training, train the model again
    if len(df) < 4:
        model = LinearRegression()
        model.fit(X, y)
    
    # Predict future values
    last_date = df.iloc[-1, 0]
    future_dates = [last_date + timedelta(days=i+1) for i in range(days_to_predict)]
    future_date_numeric = [(date - first_date).total_seconds() / (24 * 3600) for date in future_dates]
    future_df = pd.DataFrame({
        'date': future_dates,
        'date_numeric': future_date_numeric
    })
    
    # Make predictions
    future_df[f'predicted_{column}'] = model.predict(future_df[['date_numeric']])
    future_df[f'predicted_{column}'] = future_df[f'predicted_{column}'].apply(lambda x: max(0, x))  # No negative counts
    
    # Combine historical and prediction
    historical = df.iloc[:, [0, df.columns.get_loc(column)]]
    historical.columns = ['date', column]
    historical['type'] = 'historical'
    
    prediction = future_df[['date', f'predicted_{column}']]
    prediction.columns = ['date', column]
    prediction['type'] = 'prediction'
    
    result = pd.concat([historical, prediction])
    
    return result, rmse, r2

def predict_trend_arima(
    time_series: pd.DataFrame, 
    column: str = 'count',
    days_to_predict: int = 7,
    order: Tuple[int, int, int] = (1, 1, 1)
) -> Tuple[pd.DataFrame, float]:
    """
    Predict future trend using ARIMA model.
    
    Args:
        time_series: DataFrame with time series data
        column: Column name to predict
        days_to_predict: Number of days to predict into the future
        order: ARIMA model order (p,d,q)
        
    Returns:
        Tuple of (DataFrame with predictions, RMSE)
    """
    # Prepare data
    df = time_series.reset_index()
    if column not in df.columns and len(df.columns) == 2:
        # If the column doesn't exist but there are only 2 columns (index and value)
        column = df.columns[1]
    
    # Ensure datetime index
    data = df.set_index(df.columns[0])
    data = data[[column]].asfreq('D')
    data = data.fillna(method='ffill')
    
    # Define training data
    train_data = data[column].values
    
    # Fit ARIMA model
    try:
        model = ARIMA(train_data, order=order)
        model_fit = model.fit()
        
        # Make predictions
        forecast = model_fit.forecast(steps=days_to_predict)
        forecast = np.maximum(forecast, 0)  # No negative counts
        
        # Calculate error on training data
        predictions = model_fit.predict(start=0, end=len(train_data)-1)
        rmse = mean_squared_error(train_data[1:], predictions[1:], squared=False)
        
        # Create result DataFrame
        last_date = df.iloc[-1, 0]
        future_dates = [last_date + timedelta(days=i+1) for i in range(days_to_predict)]
        
        # Combine historical and prediction
        historical = df.iloc[:, [0, df.columns.get_loc(column)]]
        historical.columns = ['date', column]
        historical['type'] = 'historical'
        
        prediction = pd.DataFrame({
            'date': future_dates,
            column: forecast,
            'type': 'prediction'
        })
        
        result = pd.concat([historical, prediction])
        
        return result, rmse
    except:
        # If ARIMA fails, fall back to linear regression
        print("ARIMA failed, falling back to linear regression")
        result, rmse, _ = predict_trend_linear(time_series, column, days_to_predict)
        return result, rmse

@st.cache_data(ttl=3600)  # Cache for 1 hour
def detect_emerging_keywords(
    df: pd.DataFrame, 
    text_column: str, 
    time_column: str,
    min_count: int = 3, 
    growth_threshold: float = 1.5,
    days_to_compare: int = 3
) -> List[Dict[str, Any]]:
    """
    Detect emerging keywords/topics based on growth rate.
    
    Args:
        df: DataFrame with text data
        text_column: Column containing text
        time_column: Column containing datetime
        min_count: Minimum count to consider a keyword significant
        growth_threshold: Minimum growth rate to consider a keyword "emerging"
        days_to_compare: Number of days to compare for growth calculation
        
    Returns:
        List of dictionaries with emerging keywords and metrics
    """
    from utils.analysis import extract_keywords
    
    # Ensure datetime format
    df[time_column] = pd.to_datetime(df[time_column])
    
    # Define time periods for comparison
    latest_date = df[time_column].max()
    cutoff_date = latest_date - timedelta(days=days_to_compare)
    
    # Split data into recent and older
    recent_df = df[df[time_column] > cutoff_date]
    older_df = df[(df[time_column] <= cutoff_date) & (df[time_column] >= cutoff_date - timedelta(days=days_to_compare))]
    
    # Handle empty dataframes
    if recent_df.empty or older_df.empty:
        return []
    
    # Extract keywords for each period
    recent_keywords = extract_keywords(recent_df, text_column, top_n=50)
    older_keywords = extract_keywords(older_df, text_column, top_n=50)
    
    # Convert to dictionaries for easier lookup
    recent_dict = {k['keyword']: k['count'] for k in recent_keywords}
    older_dict = {k['keyword']: k['count'] for k in older_keywords}
    
    # Find emerging keywords
    emerging = []
    for keyword, recent_count in recent_dict.items():
        if recent_count < min_count:
            continue
            
        older_count = older_dict.get(keyword, 0)
        
        # Calculate growth
        if older_count == 0:
            # New keyword (wasn't in older period)
            growth = float(recent_count)
            emerging.append({
                'keyword': keyword,
                'recent_count': recent_count,
                'previous_count': older_count,
                'growth': 'new',
                'growth_factor': growth
            })
        else:
            growth = recent_count / older_count
            if growth >= growth_threshold:
                emerging.append({
                    'keyword': keyword,
                    'recent_count': recent_count,
                    'previous_count': older_count,
                    'growth': f"{growth:.2f}x",
                    'growth_factor': growth
                })
    
    # Sort by growth factor
    emerging = sorted(emerging, key=lambda x: x['growth_factor'], reverse=True)
    return emerging

def predict_topic_trends(
    df: pd.DataFrame,
    time_column: str,
    category_column: str,
    days_to_predict: int = 7
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    Predict trends for different topics/categories.
    
    Args:
        df: DataFrame with the data
        time_column: Column containing datetime
        category_column: Column containing topic/category
        days_to_predict: Number of days to predict
        
    Returns:
        Tuple of (DataFrame with predictions, Dict of RMSE values by category)
    """
    # Aggregate by category and time
    time_series = aggregate_time_series(df, time_column, category_column, 'D')
    
    # Make predictions for each category
    all_predictions = []
    rmse_values = {}
    
    for category in time_series.columns:
        category_ts = pd.DataFrame(time_series[category])
        
        # Predict using linear regression (more stable than ARIMA for sparse data)
        try:
            prediction, rmse, _ = predict_trend_linear(
                category_ts, 
                column=category,
                days_to_predict=days_to_predict
            )
            prediction['category'] = category
            all_predictions.append(prediction)
            rmse_values[category] = rmse
        except Exception as e:
            print(f"Error predicting trend for {category}: {e}")
            continue
    
    # Combine all predictions
    if all_predictions:
        result = pd.concat(all_predictions)
        return result, rmse_values
    else:
        return pd.DataFrame(), {}

def predict_seasonal_patterns(
    df: pd.DataFrame,
    time_column: str,
    period: int = 7  # 7 for weekly, 24 for daily, etc.
) -> Dict[str, Any]:
    """
    Identify and predict seasonal patterns in the data.
    
    Args:
        df: DataFrame with time series data
        time_column: Column containing datetime
        period: Number of time units in a seasonal cycle
        
    Returns:
        Dictionary with seasonal components and pattern information
    """
    # Ensure datetime format
    df[time_column] = pd.to_datetime(df[time_column])
    
    # Create time series
    time_series = df.groupby(pd.Grouper(key=time_column, freq='D')).size()
    time_series = pd.DataFrame(time_series, columns=['count'])
    
    # Check if we have enough data for seasonal decomposition
    if len(time_series) < period * 2:
        return {"error": "Not enough data for seasonal analysis"}
    
    # Fill missing values if any
    time_series = time_series.asfreq('D').fillna(method='ffill').fillna(0)
    
    try:
        # Perform seasonal decomposition
        result = seasonal_decompose(time_series['count'], model='additive', period=period)
        
        # Get components
        trend = result.trend
        seasonal = result.seasonal
        residual = result.resid
        
        # Find days of the week/times with highest activity
        seasonal_df = pd.DataFrame(seasonal)
        seasonal_df.index.name = 'date'
        seasonal_df = seasonal_df.reset_index()
        seasonal_df['dayofweek'] = seasonal_df['date'].dt.day_name()
        
        # Get average seasonal value by day of week
        day_avg = seasonal_df.groupby('dayofweek')['seasonal'].mean()
        
        # Find highest and lowest activity days
        highest_day = day_avg.idxmax()
        lowest_day = day_avg.idxmin()
        
        # Create result
        result_dict = {
            "highest_activity_day": highest_day,
            "lowest_activity_day": lowest_day,
            "seasonal_strength": seasonal.std() / time_series['count'].std(),
            "has_seasonal_pattern": seasonal.std() / time_series['count'].std() > 0.1,
            "weekday_patterns": day_avg.to_dict()
        }
        
        return result_dict
    except Exception as e:
        print(f"Error in seasonal analysis: {e}")
        return {"error": str(e)}

def get_trending_predictions(
    news_df: pd.DataFrame,
    social_df: pd.DataFrame,
    days_to_predict: int = 7
) -> Dict[str, Any]:
    """
    Get comprehensive trending predictions for multiple aspects of the data.
    
    Args:
        news_df: DataFrame with news articles
        social_df: DataFrame with social media posts
        days_to_predict: Number of days to predict
        
    Returns:
        Dictionary with various predictions and trending indicators
    """
    result = {}
    
    # Skip if no data
    if news_df.empty and social_df.empty:
        return {"error": "No data available for predictions"}
    
    # 1. Overall volume predictions
    if not news_df.empty and 'published_at' in news_df.columns:
        try:
            # Predict overall news volume
            news_ts = aggregate_time_series(news_df, 'published_at')
            news_prediction, news_rmse, news_r2 = predict_trend_linear(news_ts, days_to_predict=days_to_predict)
            
            result['news_volume_prediction'] = {
                'data': news_prediction.to_dict(orient='records'),
                'rmse': news_rmse,
                'r2': news_r2
            }
        except Exception as e:
            print(f"Error predicting news volume: {e}")
    
    if not social_df.empty and 'posted_at' in social_df.columns:
        try:
            # Predict overall social media volume
            social_ts = aggregate_time_series(social_df, 'posted_at')
            social_prediction, social_rmse, social_r2 = predict_trend_linear(social_ts, days_to_predict=days_to_predict)
            
            result['social_volume_prediction'] = {
                'data': social_prediction.to_dict(orient='records'),
                'rmse': social_rmse,
                'r2': social_r2
            }
        except Exception as e:
            print(f"Error predicting social volume: {e}")
    
    # 2. Topic/category trends
    if not news_df.empty and 'published_at' in news_df.columns and 'category' in news_df.columns:
        try:
            topic_predictions, topic_rmse = predict_topic_trends(
                news_df, 'published_at', 'category', days_to_predict
            )
            
            result['topic_predictions'] = {
                'data': topic_predictions.to_dict(orient='records'),
                'rmse_by_topic': topic_rmse
            }
        except Exception as e:
            print(f"Error predicting topic trends: {e}")
    
    # 3. Emerging keywords
    if not news_df.empty and 'content' in news_df.columns and 'published_at' in news_df.columns:
        try:
            emerging_keywords = detect_emerging_keywords(
                news_df, 'content', 'published_at', min_count=2, growth_threshold=1.2
            )
            
            result['emerging_keywords'] = emerging_keywords
        except Exception as e:
            print(f"Error detecting emerging keywords: {e}")
    
    # 4. Seasonal patterns
    if not news_df.empty and 'published_at' in news_df.columns:
        try:
            seasonal_patterns = predict_seasonal_patterns(news_df, 'published_at')
            result['seasonal_patterns'] = seasonal_patterns
        except Exception as e:
            print(f"Error in seasonal analysis: {e}")
    
    return result