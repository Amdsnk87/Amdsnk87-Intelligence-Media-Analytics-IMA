import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go

from utils.data_loader import get_data
from utils.analysis import (
    analyze_news_sentiment, 
    analyze_social_sentiment, 
    extract_keywords,
    analyze_trends,
    media_coverage_timeline
)
from utils.visualization import (
    plot_sentiment_timeline,
    plot_volume_timeline,
    plot_comparison_chart,
    plot_stacked_bar,
    plot_word_cloud_data
)

def show_trends(
    start_date,
    end_date,
    news_sources,
    social_sources,
    regions,
    topics
):
    """
    Display trend analysis for media content over time.
    
    Args:
        start_date: Start date for filtering data
        end_date: End date for filtering data
        news_sources: List of selected news sources
        social_sources: List of selected social media platforms
        regions: List of selected regions
        topics: List of selected topics
    """
    st.header("Media Trend Analysis")
    
    # Get data from session state or database
    if 'news_data' in st.session_state and 'social_data' in st.session_state:
        news_df = st.session_state.news_data
        social_df = st.session_state.social_data
    else:
        # Use data loader to fetch from database
        news_df, social_df = get_data(
            start_date=start_date,
            end_date=end_date,
            news_sources=news_sources,
            social_sources=social_sources,
            regions=regions,
            topics=topics,
            force_refresh=False
        )
        
        # Update session state
        st.session_state.news_data = news_df
        st.session_state.social_data = social_df
    
    # If data is not available, show message
    if news_df.empty and social_df.empty:
        st.warning("No data available in the database. Please refresh data from the dashboard.")
        return
    
    # Time interval selector
    st.subheader("Time Period Analysis")
    time_period = st.radio(
        "Analyze trends by:",
        ["Day", "Week", "Hour"],
        horizontal=True,
        key="trend_time_period"
    )
    
    # Map selection to time period parameter
    period_param = time_period.lower()
    
    # Analyze trends
    with st.spinner("Analyzing trends..."):
        news_trends, social_trends = analyze_trends(news_df, social_df, period_param)
    
    # Volume Trends
    st.subheader("Media Volume Trends")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # News volume trends
        if 'volume' in news_trends and news_trends['volume']:
            # Convert to DataFrame for plotting
            volume_df = pd.DataFrame(news_trends['volume'])
            
            # Ensure 'period' is datetime
            volume_df['period'] = pd.to_datetime(volume_df['period'])
            
            # Create plot
            volume_fig = plot_volume_timeline(
                volume_df, 
                'period', 
                "News Volume Over Time"
            )
            st.plotly_chart(volume_fig, use_container_width=True)
        else:
            st.info("No news volume data available for the selected period")
    
    with col2:
        # Social media volume trends
        if 'volume' in social_trends and social_trends['volume']:
            # Convert to DataFrame for plotting
            social_volume_df = pd.DataFrame(social_trends['volume'])
            
            # Ensure 'period' is datetime
            social_volume_df['period'] = pd.to_datetime(social_volume_df['period'])
            
            # Create plot
            social_volume_fig = plot_volume_timeline(
                social_volume_df, 
                'period', 
                "Social Media Volume Over Time"
            )
            st.plotly_chart(social_volume_fig, use_container_width=True)
        else:
            st.info("No social media volume data available for the selected period")
    
    # Keyword Analysis
    st.subheader("Keyword Trends Analysis")
    
    # Let user input keywords to track
    keyword_input = st.text_input(
        "Enter keywords to analyze (comma separated)",
        "politik, ekonomi, covid, pendidikan",
        help="Enter keywords to analyze their presence in media coverage over time"
    )
    
    # Process keywords
    if keyword_input:
        keywords = [k.strip() for k in keyword_input.split(",")]
        
        # Get keyword timeline data
        timeline_data = media_coverage_timeline(
            news_df, 
            keywords, 
            start_date, 
            end_date
        )
        
        # Display timeline chart
        if timeline_data:
            # Transform data for plotting
            flat_data = []
            for keyword, items in timeline_data.items():
                for item in items:
                    item_copy = item.copy()
                    item_copy['keyword'] = keyword
                    flat_data.append(item_copy)
            
            if flat_data:
                timeline_df = pd.DataFrame(flat_data)
                
                # Convert date to datetime
                timeline_df['date'] = pd.to_datetime(timeline_df['date'])
                
                # Create comparison chart
                timeline_fig = plot_comparison_chart(
                    timeline_data,
                    'date',
                    'count',
                    'keyword',
                    "Keyword Mentions Over Time"
                )
                st.plotly_chart(timeline_fig, use_container_width=True)
            else:
                st.info("No keyword mentions found in the specified date range")
        else:
            st.info("No keyword data available")
    
    # Sentiment Trends
    st.subheader("Sentiment Trends")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # News sentiment trends
        if 'sentiment' in news_trends and news_trends['sentiment']:
            # Convert to DataFrame for plotting
            sentiment_df = pd.DataFrame(news_trends['sentiment'])
            
            # Ensure 'period' is datetime
            sentiment_df['period'] = pd.to_datetime(sentiment_df['period'])
            
            # Create stacked bar chart
            sentiment_fig = plot_stacked_bar(
                sentiment_df,
                'period',
                'count',
                'sentiment_category',
                "News Sentiment Trends"
            )
            st.plotly_chart(sentiment_fig, use_container_width=True)
        else:
            st.info("No news sentiment trend data available")
    
    with col2:
        # Social media sentiment trends
        if 'sentiment' in social_trends and social_trends['sentiment']:
            # Convert to DataFrame for plotting
            social_sentiment_df = pd.DataFrame(social_trends['sentiment'])
            
            # Ensure 'period' is datetime
            social_sentiment_df['period'] = pd.to_datetime(social_sentiment_df['period'])
            
            # Create stacked bar chart
            social_sentiment_fig = plot_stacked_bar(
                social_sentiment_df,
                'period',
                'count',
                'sentiment_category',
                "Social Media Sentiment Trends"
            )
            st.plotly_chart(social_sentiment_fig, use_container_width=True)
        else:
            st.info("No social media sentiment trend data available")
    
    # Topic Trends
    st.subheader("Topic Trends")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # News topic trends
        if 'category' in news_trends and news_trends['category']:
            # Convert to DataFrame for plotting
            category_df = pd.DataFrame(news_trends['category'])
            
            # Ensure 'period' is datetime
            category_df['period'] = pd.to_datetime(category_df['period'])
            
            # Create stacked bar chart
            category_fig = plot_stacked_bar(
                category_df,
                'period',
                'count',
                'category',
                "News Topic Trends"
            )
            st.plotly_chart(category_fig, use_container_width=True)
        else:
            st.info("No news topic trend data available")
    
    with col2:
        # Social media topic trends
        if 'category' in social_trends and social_trends['category']:
            # Convert to DataFrame for plotting
            social_category_df = pd.DataFrame(social_trends['category'])
            
            # Ensure 'period' is datetime
            social_category_df['period'] = pd.to_datetime(social_category_df['period'])
            
            # Create stacked bar chart
            social_category_fig = plot_stacked_bar(
                social_category_df,
                'period',
                'count',
                'category',
                "Social Media Topic Trends"
            )
            st.plotly_chart(social_category_fig, use_container_width=True)
        else:
            st.info("No social media topic trend data available")
    
    # Source/Media Trends
    st.subheader("Media Source Trends")
    
    # News source trends
    if 'source' in news_trends and news_trends['source']:
        # Convert to DataFrame for plotting
        source_df = pd.DataFrame(news_trends['source'])
        
        # Ensure 'period' is datetime
        source_df['period'] = pd.to_datetime(source_df['period'])
        
        # Create stacked bar chart
        source_fig = plot_stacked_bar(
            source_df,
            'period',
            'count',
            'source',
            "News Source Trends"
        )
        st.plotly_chart(source_fig, use_container_width=True)
    else:
        st.info("No news source trend data available")
    
    # Social Media Engagement Trends
    st.subheader("Social Media Engagement Trends")
    
    # Social media engagement trends
    if 'engagement' in social_trends and social_trends['engagement']:
        # Convert to DataFrame for plotting
        engagement_df = pd.DataFrame(social_trends['engagement'])
        
        # Ensure 'period' is datetime
        engagement_df['period'] = pd.to_datetime(engagement_df['period'])
        
        # Create engagement metrics plot
        fig = go.Figure()
        
        # Add individual engagement metrics
        if 'likes' in engagement_df.columns:
            fig.add_trace(go.Scatter(
                x=engagement_df['period'],
                y=engagement_df['likes'],
                mode='lines+markers',
                name='Likes'
            ))
        
        if 'retweets' in engagement_df.columns:
            fig.add_trace(go.Scatter(
                x=engagement_df['period'],
                y=engagement_df['retweets'],
                mode='lines+markers',
                name='Retweets/Shares'
            ))
        
        if 'total_engagement' in engagement_df.columns:
            fig.add_trace(go.Scatter(
                x=engagement_df['period'],
                y=engagement_df['total_engagement'],
                mode='lines+markers',
                name='Total Engagement',
                line=dict(width=3, dash='dash')
            ))
        
        # Update layout
        fig.update_layout(
            title="Social Media Engagement Over Time",
            xaxis_title="Date",
            yaxis_title="Count",
            legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5)
        )
        
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No social media engagement trend data available")
    
    # Regional Trends
    st.subheader("Regional Coverage Trends")
    
    # Regional news trends
    if 'region' in news_trends and news_trends['region']:
        # Convert to DataFrame for plotting
        region_df = pd.DataFrame(news_trends['region'])
        
        # Ensure 'period' is datetime
        region_df['period'] = pd.to_datetime(region_df['period'])
        
        # Create stacked bar chart for regions
        region_fig = plot_stacked_bar(
            region_df,
            'period',
            'count',
            'region',
            "Regional News Coverage Trends"
        )
        st.plotly_chart(region_fig, use_container_width=True)
    else:
        st.info("No regional trend data available")
    
    # Comparative Analysis
    st.subheader("Comparative Media Analysis")
    
    # Allow user to select metrics to compare
    comparison_options = [
        "News vs. Social Media Volume",
        "News vs. Social Media Sentiment",
        "Platform Comparison (Social Media)"
    ]
    
    comparison_type = st.selectbox(
        "Select comparison type",
        comparison_options
    )
    
    if comparison_type == "News vs. Social Media Volume":
        # Create comparison of news and social media volume
        if 'volume' in news_trends and news_trends['volume'] and 'volume' in social_trends and social_trends['volume']:
            # Convert to DataFrames
            news_volume_df = pd.DataFrame(news_trends['volume'])
            social_volume_df = pd.DataFrame(social_trends['volume'])
            
            # Ensure 'period' is datetime
            news_volume_df['period'] = pd.to_datetime(news_volume_df['period'])
            social_volume_df['period'] = pd.to_datetime(social_volume_df['period'])
            
            # Add source column
            news_volume_df['source'] = 'News'
            social_volume_df['source'] = 'Social Media'
            
            # Combine dataframes
            combined_df = pd.concat([news_volume_df, social_volume_df])
            
            # Create comparison chart
            volume_compare_fig = px.line(
                combined_df,
                x='period',
                y='count',
                color='source',
                title="News vs. Social Media Volume",
                markers=True
            )
            
            volume_compare_fig.update_layout(
                xaxis_title="Date",
                yaxis_title="Count",
                legend_title="Source",
                legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5)
            )
            
            st.plotly_chart(volume_compare_fig, use_container_width=True)
        else:
            st.info("Not enough data available for volume comparison")
    
    elif comparison_type == "News vs. Social Media Sentiment":
        # Create comparison of news and social media sentiment
        if 'sentiment' in news_trends and news_trends['sentiment'] and 'sentiment' in social_trends and social_trends['sentiment']:
            # Convert to DataFrames
            news_sentiment_df = pd.DataFrame(news_trends['sentiment'])
            social_sentiment_df = pd.DataFrame(social_trends['sentiment'])
            
            # Ensure 'period' is datetime
            news_sentiment_df['period'] = pd.to_datetime(news_sentiment_df['period'])
            social_sentiment_df['period'] = pd.to_datetime(social_sentiment_df['period'])
            
            # Add source column
            news_sentiment_df['source'] = 'News'
            social_sentiment_df['source'] = 'Social Media'
            
            # Combine dataframes
            combined_df = pd.concat([news_sentiment_df, social_sentiment_df])
            
            # Create comparison chart
            sentiment_compare_fig = px.line(
                combined_df,
                x='period',
                y='count',
                color='sentiment_category',
                facet_col='source',
                title="News vs. Social Media Sentiment",
                markers=True,
                color_discrete_map={'positive': '#4CAF50', 'neutral': '#FF9800', 'negative': '#F44336'}
            )
            
            sentiment_compare_fig.update_layout(
                xaxis_title="Date",
                yaxis_title="Count",
                legend_title="Sentiment",
                height=500
            )
            
            st.plotly_chart(sentiment_compare_fig, use_container_width=True)
        else:
            st.info("Not enough data available for sentiment comparison")
    
    elif comparison_type == "Platform Comparison (Social Media)":
        # Create comparison of different social media platforms
        if 'platform' in social_trends and social_trends['platform']:
            # Convert to DataFrame
            platform_df = pd.DataFrame(social_trends['platform'])
            
            # Ensure 'period' is datetime
            platform_df['period'] = pd.to_datetime(platform_df['period'])
            
            # Create comparison chart
            platform_fig = px.line(
                platform_df,
                x='period',
                y='count',
                color='platform',
                title="Social Media Platform Comparison",
                markers=True
            )
            
            platform_fig.update_layout(
                xaxis_title="Date",
                yaxis_title="Count",
                legend_title="Platform",
                legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5)
            )
            
            st.plotly_chart(platform_fig, use_container_width=True)
        else:
            st.info("Not enough data available for platform comparison")

    # Footer with refresh time
    st.markdown("---")
    st.markdown(f"Data last updated at {st.session_state.last_update.strftime('%Y-%m-%d %H:%M:%S')}")
