import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
import altair as alt

from utils.data_loader import get_data
from utils.analysis import (
    analyze_news_sentiment, 
    analyze_social_sentiment, 
    extract_keywords,
    create_word_cloud_data,
    search_content
)
from utils.visualization import (
    plot_sentiment_distribution,
    plot_sentiment_timeline,
    plot_stacked_bar,
    plot_word_cloud_data,
    plot_radar_chart
)

def show_sentiment(
    start_date,
    end_date,
    news_sources,
    social_sources,
    regions,
    topics
):
    """
    Display sentiment analysis for media content.
    
    Args:
        start_date: Start date for filtering data
        end_date: End date for filtering data
        news_sources: List of selected news sources
        social_sources: List of selected social media platforms
        regions: List of selected regions
        topics: List of selected topics
    """
    st.header("Media Sentiment Analysis")
    
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
    
    # Ensure sentiment data is analyzed
    if not news_df.empty and 'sentiment_category' not in news_df.columns:
        with st.spinner("Analyzing news sentiment..."):
            news_df = analyze_news_sentiment(news_df)
            # Update session state
            st.session_state.news_data = news_df
    
    if not social_df.empty and 'sentiment_category' not in social_df.columns:
        with st.spinner("Analyzing social media sentiment..."):
            social_df = analyze_social_sentiment(social_df)
            # Update session state
            st.session_state.social_data = social_df
    
    # Overall Sentiment Distribution
    st.subheader("Overall Sentiment Distribution")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # News sentiment distribution
        if not news_df.empty and 'sentiment_category' in news_df.columns:
            news_sentiment_fig = plot_sentiment_distribution(news_df, "News Sentiment Distribution")
            st.plotly_chart(news_sentiment_fig, use_container_width=True)
            
            # Show summary metrics
            positive_pct = (news_df['sentiment_category'] == 'positive').mean() * 100
            neutral_pct = (news_df['sentiment_category'] == 'neutral').mean() * 100
            negative_pct = (news_df['sentiment_category'] == 'negative').mean() * 100
            
            # Display metrics in columns
            metrics_cols = st.columns(3)
            metrics_cols[0].metric("Positive", f"{positive_pct:.1f}%")
            metrics_cols[1].metric("Neutral", f"{neutral_pct:.1f}%")
            metrics_cols[2].metric("Negative", f"{negative_pct:.1f}%")
        else:
            st.info("No news sentiment data available")
    
    with col2:
        # Social media sentiment distribution
        if not social_df.empty and 'sentiment_category' in social_df.columns:
            social_sentiment_fig = plot_sentiment_distribution(social_df, "Social Media Sentiment Distribution")
            st.plotly_chart(social_sentiment_fig, use_container_width=True)
            
            # Show summary metrics
            positive_pct = (social_df['sentiment_category'] == 'positive').mean() * 100
            neutral_pct = (social_df['sentiment_category'] == 'neutral').mean() * 100
            negative_pct = (social_df['sentiment_category'] == 'negative').mean() * 100
            
            # Display metrics in columns
            metrics_cols = st.columns(3)
            metrics_cols[0].metric("Positive", f"{positive_pct:.1f}%")
            metrics_cols[1].metric("Neutral", f"{neutral_pct:.1f}%")
            metrics_cols[2].metric("Negative", f"{negative_pct:.1f}%")
        else:
            st.info("No social media sentiment data available")
    
    # Sentiment Over Time
    st.subheader("Sentiment Trends Over Time")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # News sentiment timeline
        if not news_df.empty and 'sentiment_category' in news_df.columns and 'published_at' in news_df.columns:
            news_timeline_fig = plot_sentiment_timeline(news_df, 'published_at', "News Sentiment Over Time")
            st.plotly_chart(news_timeline_fig, use_container_width=True)
        else:
            st.info("No news sentiment timeline data available")
    
    with col2:
        # Social media sentiment timeline
        if not social_df.empty and 'sentiment_category' in social_df.columns and 'posted_at' in social_df.columns:
            social_timeline_fig = plot_sentiment_timeline(social_df, 'posted_at', "Social Media Sentiment Over Time")
            st.plotly_chart(social_timeline_fig, use_container_width=True)
        else:
            st.info("No social media sentiment timeline data available")
    
    # Sentiment by Topic
    st.subheader("Sentiment Analysis by Topic")
    
    if not news_df.empty and 'sentiment_category' in news_df.columns and 'category' in news_df.columns:
        # Group by category and sentiment
        topic_sentiment = news_df.groupby(['category', 'sentiment_category']).size().reset_index(name='count')
        
        # Create stacked bar chart
        if not topic_sentiment.empty:
            topic_sentiment_fig = plot_stacked_bar(
                topic_sentiment,
                'category',
                'count',
                'sentiment_category',
                "Sentiment Distribution by Topic"
            )
            st.plotly_chart(topic_sentiment_fig, use_container_width=True)
        else:
            st.info("No topic sentiment data available")
    else:
        st.info("Topic or sentiment data not available")
    
    # Sentiment by Source
    st.subheader("Sentiment Analysis by Source")
    
    if not news_df.empty and 'sentiment_category' in news_df.columns and 'source' in news_df.columns:
        # Group by source and sentiment
        source_sentiment = news_df.groupby(['source', 'sentiment_category']).size().reset_index(name='count')
        
        # Create stacked bar chart
        if not source_sentiment.empty:
            source_sentiment_fig = plot_stacked_bar(
                source_sentiment,
                'source',
                'count',
                'sentiment_category',
                "Sentiment Distribution by News Source"
            )
            st.plotly_chart(source_sentiment_fig, use_container_width=True)
        else:
            st.info("No source sentiment data available")
    else:
        st.info("Source or sentiment data not available")
    
    # Sentiment by Region
    st.subheader("Sentiment Analysis by Region")
    
    if not news_df.empty and 'sentiment_category' in news_df.columns and 'region' in news_df.columns:
        # Group by region and sentiment
        region_sentiment = news_df.groupby(['region', 'sentiment_category']).size().reset_index(name='count')
        
        # Create stacked bar chart
        if not region_sentiment.empty:
            region_sentiment_fig = plot_stacked_bar(
                region_sentiment,
                'region',
                'count',
                'sentiment_category',
                "Sentiment Distribution by Region"
            )
            st.plotly_chart(region_sentiment_fig, use_container_width=True)
        else:
            st.info("No regional sentiment data available")
    else:
        st.info("Regional or sentiment data not available")
    
    # Sentiment Word Analysis
    st.subheader("Sentiment Word Analysis")
    
    # Create tabs for positive, negative, and neutral content
    sentiment_tabs = st.tabs(["Positive Content", "Negative Content", "Neutral Content"])
    
    # Prepare word cloud data for each sentiment category
    with sentiment_tabs[0]:  # Positive content
        col1, col2 = st.columns(2)
        
        with col1:
            # News positive word cloud
            if not news_df.empty and 'sentiment_category' in news_df.columns:
                positive_news = news_df[news_df['sentiment_category'] == 'positive']
                
                if not positive_news.empty and 'content' in positive_news.columns:
                    # Create word cloud data
                    word_cloud_data = create_word_cloud_data(positive_news, 'content', max_words=50)
                    
                    if word_cloud_data:
                        st.markdown("#### Positive Words in News")
                        word_cloud_chart = plot_word_cloud_data(word_cloud_data)
                        st.altair_chart(word_cloud_chart, use_container_width=True)
                    else:
                        st.info("No positive words found in news content")
                else:
                    st.info("No positive news content available")
            else:
                st.info("No news sentiment data available")
        
        with col2:
            # Social media positive word cloud
            if not social_df.empty and 'sentiment_category' in social_df.columns:
                positive_social = social_df[social_df['sentiment_category'] == 'positive']
                
                if not positive_social.empty and 'content' in positive_social.columns:
                    # Create word cloud data
                    word_cloud_data = create_word_cloud_data(positive_social, 'content', max_words=50)
                    
                    if word_cloud_data:
                        st.markdown("#### Positive Words in Social Media")
                        word_cloud_chart = plot_word_cloud_data(word_cloud_data)
                        st.altair_chart(word_cloud_chart, use_container_width=True)
                    else:
                        st.info("No positive words found in social media content")
                else:
                    st.info("No positive social media content available")
            else:
                st.info("No social media sentiment data available")
    
    with sentiment_tabs[1]:  # Negative content
        col1, col2 = st.columns(2)
        
        with col1:
            # News negative word cloud
            if not news_df.empty and 'sentiment_category' in news_df.columns:
                negative_news = news_df[news_df['sentiment_category'] == 'negative']
                
                if not negative_news.empty and 'content' in negative_news.columns:
                    # Create word cloud data
                    word_cloud_data = create_word_cloud_data(negative_news, 'content', max_words=50)
                    
                    if word_cloud_data:
                        st.markdown("#### Negative Words in News")
                        word_cloud_chart = plot_word_cloud_data(word_cloud_data)
                        st.altair_chart(word_cloud_chart, use_container_width=True)
                    else:
                        st.info("No negative words found in news content")
                else:
                    st.info("No negative news content available")
            else:
                st.info("No news sentiment data available")
        
        with col2:
            # Social media negative word cloud
            if not social_df.empty and 'sentiment_category' in social_df.columns:
                negative_social = social_df[social_df['sentiment_category'] == 'negative']
                
                if not negative_social.empty and 'content' in negative_social.columns:
                    # Create word cloud data
                    word_cloud_data = create_word_cloud_data(negative_social, 'content', max_words=50)
                    
                    if word_cloud_data:
                        st.markdown("#### Negative Words in Social Media")
                        word_cloud_chart = plot_word_cloud_data(word_cloud_data)
                        st.altair_chart(word_cloud_chart, use_container_width=True)
                    else:
                        st.info("No negative words found in social media content")
                else:
                    st.info("No negative social media content available")
            else:
                st.info("No social media sentiment data available")
    
    with sentiment_tabs[2]:  # Neutral content
        col1, col2 = st.columns(2)
        
        with col1:
            # News neutral word cloud
            if not news_df.empty and 'sentiment_category' in news_df.columns:
                neutral_news = news_df[news_df['sentiment_category'] == 'neutral']
                
                if not neutral_news.empty and 'content' in neutral_news.columns:
                    # Create word cloud data
                    word_cloud_data = create_word_cloud_data(neutral_news, 'content', max_words=50)
                    
                    if word_cloud_data:
                        st.markdown("#### Neutral Words in News")
                        word_cloud_chart = plot_word_cloud_data(word_cloud_data)
                        st.altair_chart(word_cloud_chart, use_container_width=True)
                    else:
                        st.info("No neutral words found in news content")
                else:
                    st.info("No neutral news content available")
            else:
                st.info("No news sentiment data available")
        
        with col2:
            # Social media neutral word cloud
            if not social_df.empty and 'sentiment_category' in social_df.columns:
                neutral_social = social_df[social_df['sentiment_category'] == 'neutral']
                
                if not neutral_social.empty and 'content' in neutral_social.columns:
                    # Create word cloud data
                    word_cloud_data = create_word_cloud_data(neutral_social, 'content', max_words=50)
                    
                    if word_cloud_data:
                        st.markdown("#### Neutral Words in Social Media")
                        word_cloud_chart = plot_word_cloud_data(word_cloud_data)
                        st.altair_chart(word_cloud_chart, use_container_width=True)
                    else:
                        st.info("No neutral words found in social media content")
                else:
                    st.info("No neutral social media content available")
            else:
                st.info("No social media sentiment data available")
    
    # Sentiment Explorer - Allow users to search for content with specific sentiment
    st.subheader("Sentiment Content Explorer")
    
    # Create tabs for news and social media
    explorer_tabs = st.tabs(["News Content", "Social Media Content"])
    
    # News content explorer
    with explorer_tabs[0]:
        if not news_df.empty and 'sentiment_category' in news_df.columns:
            # Sentiment filter
            sentiment_filter = st.selectbox(
                "Filter by sentiment",
                ["All", "Positive", "Neutral", "Negative"],
                key="news_sentiment_filter"
            )
            
            # Apply filter
            if sentiment_filter != "All":
                filtered_df = news_df[news_df['sentiment_category'].str.lower() == sentiment_filter.lower()]
            else:
                filtered_df = news_df
            
            # Show number of articles
            st.write(f"Found {len(filtered_df)} news articles with {sentiment_filter.lower() if sentiment_filter != 'All' else 'any'} sentiment")
            
            # Sort by sentiment polarity
            if 'sentiment_polarity' in filtered_df.columns:
                sorted_df = filtered_df.sort_values(
                    'sentiment_polarity', 
                    ascending=(sentiment_filter == "Negative")
                )
            else:
                sorted_df = filtered_df
            
            # Display articles
            for i, (_, article) in enumerate(sorted_df.head(10).iterrows()):
                # Get article details
                title = article.get('title', f"Article {i+1}")
                source = article.get('source', 'Unknown')
                date = article.get('published_at', '')
                sentiment = article.get('sentiment_category', '')
                polarity = article.get('sentiment_polarity', 0)
                content = article.get('content', 'No content available')
                url = article.get('url', '')
                
                # Format date if available
                if date:
                    try:
                        date = pd.to_datetime(date).strftime('%Y-%m-%d %H:%M')
                    except:
                        pass
                
                # Determine sentiment color
                sentiment_color = {
                    'positive': ':green[Positive]',
                    'neutral': ':blue[Neutral]',
                    'negative': ':red[Negative]'
                }.get(sentiment.lower(), '')
                
                # Create expander for the article
                with st.expander(f"{title} ({source}, {date}) - {sentiment_color} ({polarity:.2f})"):
                    st.markdown(f"**Source:** {source}")
                    st.markdown(f"**Published:** {date}")
                    st.markdown(f"**Sentiment:** {sentiment_color} ({polarity:.2f})")
                    st.markdown(f"**Content:**\n{content[:500]}...")
                    if url:
                        st.markdown(f"[Read full article]({url})")
        else:
            st.info("No news sentiment data available")
    
    # Social media content explorer
    with explorer_tabs[1]:
        if not social_df.empty and 'sentiment_category' in social_df.columns:
            # Sentiment filter
            sentiment_filter = st.selectbox(
                "Filter by sentiment",
                ["All", "Positive", "Neutral", "Negative"],
                key="social_sentiment_filter"
            )
            
            # Apply filter
            if sentiment_filter != "All":
                filtered_df = social_df[social_df['sentiment_category'].str.lower() == sentiment_filter.lower()]
            else:
                filtered_df = social_df
            
            # Show number of posts
            st.write(f"Found {len(filtered_df)} social media posts with {sentiment_filter.lower() if sentiment_filter != 'All' else 'any'} sentiment")
            
            # Sort by sentiment polarity
            if 'sentiment_polarity' in filtered_df.columns:
                sorted_df = filtered_df.sort_values(
                    'sentiment_polarity', 
                    ascending=(sentiment_filter == "Negative")
                )
            else:
                sorted_df = filtered_df
            
            # Display posts
            for i, (_, post) in enumerate(sorted_df.head(10).iterrows()):
                # Get post details
                username = post.get('username', f"User {i+1}")
                platform = post.get('platform', 'Unknown')
                date = post.get('posted_at', '')
                sentiment = post.get('sentiment_category', '')
                polarity = post.get('sentiment_polarity', 0)
                content = post.get('content', 'No content available')
                url = post.get('url', '')
                
                # Format date if available
                if date:
                    try:
                        date = pd.to_datetime(date).strftime('%Y-%m-%d %H:%M')
                    except:
                        pass
                
                # Determine sentiment color
                sentiment_color = {
                    'positive': ':green[Positive]',
                    'neutral': ':blue[Neutral]',
                    'negative': ':red[Negative]'
                }.get(sentiment.lower(), '')
                
                # Create expander for the post
                with st.expander(f"{username} on {platform} ({date}) - {sentiment_color} ({polarity:.2f})"):
                    st.markdown(f"**User:** {username}")
                    st.markdown(f"**Platform:** {platform}")
                    st.markdown(f"**Posted:** {date}")
                    st.markdown(f"**Sentiment:** {sentiment_color} ({polarity:.2f})")
                    
                    # Display engagement metrics if available
                    engagement_info = []
                    if 'likes' in post:
                        engagement_info.append(f"Likes: {post['likes']}")
                    if 'retweets' in post:
                        engagement_info.append(f"Retweets: {post['retweets']}")
                    if 'shares' in post:
                        engagement_info.append(f"Shares: {post['shares']}")
                    if 'comments' in post:
                        engagement_info.append(f"Comments: {post['comments']}")
                    
                    if engagement_info:
                        st.markdown(f"**Engagement:** {' | '.join(engagement_info)}")
                    
                    st.markdown(f"**Content:**\n{content}")
                    
                    if url:
                        st.markdown(f"[View original post]({url})")
        else:
            st.info("No social media sentiment data available")
    
    # Footer with refresh time
    st.markdown("---")
    st.markdown(f"Data last updated at {st.session_state.last_update.strftime('%Y-%m-%d %H:%M:%S')}")
