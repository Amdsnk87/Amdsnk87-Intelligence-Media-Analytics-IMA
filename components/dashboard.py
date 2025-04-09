import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time

from utils.data_loader import get_data
from utils.analysis import (
    analyze_news_sentiment, 
    analyze_social_sentiment, 
    extract_keywords,
    analyze_trends,
    identify_influencers,
    calculate_media_share
)
from utils.visualization import (
    plot_sentiment_distribution,
    plot_topic_distribution,
    plot_media_share,
    plot_volume_timeline,
    plot_regional_heatmap,
    plot_top_entities
)

def show_dashboard(
    start_date,
    end_date,
    news_sources,
    social_sources,
    regions,
    topics
):
    """
    Display the main dashboard with real-time media analytics.
    
    Args:
        start_date: Start date for filtering data
        end_date: End date for filtering data
        news_sources: List of selected news sources
        social_sources: List of selected social media platforms
        regions: List of selected regions
        topics: List of selected topics
    """
    st.header("Real-time Media Analytics Dashboard")
    
    # Add loading indicator
    with st.spinner("Fetching and analyzing data..."):
        # Check if refresh button was clicked
        force_refresh = st.session_state.get('refresh_clicked', False)
        
        if force_refresh:
            st.session_state.refresh_clicked = False  # Reset after using
        
        # Use our data loader to get data from database or API
        news_df, social_df = get_data(
            start_date=start_date,
            end_date=end_date,
            news_sources=news_sources,
            social_sources=social_sources,
            regions=regions,
            topics=topics,
            force_refresh=force_refresh
        )
        
        # Update session state
        st.session_state.news_data = news_df
        st.session_state.social_data = social_df
    
    # Display summary metrics
    st.subheader("Summary")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        news_count = len(news_df) if not news_df.empty else 0
        st.metric("News Articles", news_count)
        
        # Display "Show All Articles" button
        if news_count > 0:
            st.session_state.show_all_articles = st.checkbox("Show all articles", value=False)
    
    with col2:
        social_count = len(social_df) if not social_df.empty else 0
        st.metric("Social Media Posts", social_count)
    
    with col3:
        if not news_df.empty and 'sentiment_category' in news_df.columns:
            positive_pct = (news_df['sentiment_category'] == 'positive').mean() * 100
            st.metric("Positive News", f"{positive_pct:.1f}%")
        else:
            st.metric("Positive News", "N/A")
    
    with col4:
        if not news_df.empty and 'region' in news_df.columns:
            regions_count = news_df['region'].nunique()
            st.metric("Regions Covered", regions_count)
        else:
            st.metric("Regions Covered", "N/A")
    
    # Display all articles if checkbox is checked
    if st.session_state.get('show_all_articles', False) and not news_df.empty:
        st.subheader(f"All News Articles ({len(news_df)} articles)")
        
        # Add filter options
        with st.expander("Filter Options"):
            # Filter by source
            filter_sources = st.multiselect(
                "Filter by Source",
                options=sorted(news_df['source'].unique().tolist()),
                default=[]
            )
            
            # Filter by category
            if 'category' in news_df.columns:
                filter_categories = st.multiselect(
                    "Filter by Category",
                    options=sorted(news_df['category'].unique().tolist()),
                    default=[]
                )
            else:
                filter_categories = []
            
            # Filter by region
            if 'region' in news_df.columns:
                filter_regions = st.multiselect(
                    "Filter by Region",
                    options=sorted(news_df['region'].unique().tolist()),
                    default=[]
                )
            else:
                filter_regions = []
            
            # Filter by sentiment
            if 'sentiment_category' in news_df.columns:
                filter_sentiment = st.multiselect(
                    "Filter by Sentiment",
                    options=sorted(news_df['sentiment_category'].unique().tolist()),
                    default=[]
                )
            else:
                filter_sentiment = []
            
            # Search within articles
            search_text = st.text_input("Search within articles", "")
        
        # Apply filters
        filtered_df = news_df.copy()
        
        # Filter by source
        if filter_sources:
            filtered_df = filtered_df[filtered_df['source'].isin(filter_sources)]
        
        # Filter by category
        if filter_categories and 'category' in filtered_df.columns:
            filtered_df = filtered_df[filtered_df['category'].isin(filter_categories)]
        
        # Filter by region
        if filter_regions and 'region' in filtered_df.columns:
            filtered_df = filtered_df[filtered_df['region'].isin(filter_regions)]
        
        # Filter by sentiment
        if filter_sentiment and 'sentiment_category' in filtered_df.columns:
            filtered_df = filtered_df[filtered_df['sentiment_category'].isin(filter_sentiment)]
        
        # Filter by search text
        if search_text:
            # Search in title, content and description
            search_condition = False
            for col in ['title', 'content', 'description']:
                if col in filtered_df.columns:
                    filtered_df[col] = filtered_df[col].fillna('')
                    col_condition = filtered_df[col].str.contains(search_text, case=False)
                    search_condition = search_condition | col_condition if search_condition is not False else col_condition
            
            if search_condition is not False:
                filtered_df = filtered_df[search_condition]
        
        # Sort by date (most recent first)
        if 'published_at' in filtered_df.columns:
            filtered_df = filtered_df.sort_values('published_at', ascending=False)
        
        # Show filtered count
        filtered_count = len(filtered_df)
        st.write(f"Showing {filtered_count} articles based on current filters")
        
        # Display sort options
        sort_options = ["Most Recent First", "Oldest First", "Source (A-Z)", "Category (A-Z)"]
        sort_choice = st.selectbox("Sort by:", sort_options)
        
        # Apply sorting
        if sort_choice == "Most Recent First" and 'published_at' in filtered_df.columns:
            filtered_df = filtered_df.sort_values('published_at', ascending=False)
        elif sort_choice == "Oldest First" and 'published_at' in filtered_df.columns:
            filtered_df = filtered_df.sort_values('published_at', ascending=True)
        elif sort_choice == "Source (A-Z)" and 'source' in filtered_df.columns:
            filtered_df = filtered_df.sort_values('source', ascending=True)
        elif sort_choice == "Category (A-Z)" and 'category' in filtered_df.columns:
            filtered_df = filtered_df.sort_values('category', ascending=True)
        
        # Display articles
        for i, (_, article) in enumerate(filtered_df.iterrows()):
            # Get article details
            title = article.get('title', f"Article {i+1}")
            source = article.get('source', 'Unknown')
            date = article.get('published_at', '')
            content = article.get('content', 'No content available')
            description = article.get('description', '')
            url = article.get('url', '')
            sentiment = article.get('sentiment_category', '')
            category = article.get('category', '')
            region = article.get('region', '')
            
            # Format date if available
            if date:
                try:
                    date = pd.to_datetime(date).strftime('%Y-%m-%d %H:%M')
                except:
                    pass
            
            # Create expander for the article
            sentiment_color = {
                'positive': ':green[Positive]',
                'neutral': ':blue[Neutral]',
                'negative': ':red[Negative]'
            }.get(sentiment.lower() if isinstance(sentiment, str) else '', '')
            
            expander_title = f"{title} ({source})"
            with st.expander(expander_title):
                # Two columns for metadata
                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"**Source:** {source}")
                    st.write(f"**Published:** {date}")
                    if sentiment:
                        st.write(f"**Sentiment:** {sentiment_color}")
                
                with col2:
                    if category:
                        st.write(f"**Category:** {category}")
                    if region:
                        st.write(f"**Region:** {region}")
                    if url:
                        st.write(f"**URL:** [Link to article]({url})")
                
                # Article content
                st.markdown("---")
                st.markdown(f"**Description:** {description}")
                if content:
                    st.markdown(f"**Full Content:** {content[:1000]}{'...' if len(content) > 1000 else ''}")
        
        # Add pagination-like message if there are many articles
        if filtered_count > 50:
            st.markdown("---")
            st.info("Showing all available articles. Scroll up to see more.")
        
        # Add a separator
        st.markdown("---")
        
    # Two columns for charts
    col1, col2 = st.columns(2)
    
    with col1:
        # News Sentiment Distribution
        if not news_df.empty and 'sentiment_category' in news_df.columns:
            sentiment_fig = plot_sentiment_distribution(news_df, "News Sentiment Distribution")
            st.plotly_chart(sentiment_fig, use_container_width=True)
        else:
            st.info("No news sentiment data available")
        
        # Topic Distribution
        if not news_df.empty and 'category' in news_df.columns:
            topic_fig = plot_topic_distribution(news_df, "Topic Distribution")
            st.plotly_chart(topic_fig, use_container_width=True)
        else:
            st.info("No topic distribution data available")
    
    with col2:
        # Media Share (by source)
        if not news_df.empty and 'source' in news_df.columns:
            media_fig = plot_media_share(news_df, "Media Share by Source")
            st.plotly_chart(media_fig, use_container_width=True)
        else:
            st.info("No media share data available")
        
        # Regional Distribution
        if not news_df.empty and 'region' in news_df.columns:
            region_fig = plot_regional_heatmap(news_df, "Regional News Distribution")
            st.plotly_chart(region_fig, use_container_width=True)
        else:
            st.info("No regional distribution data available")
    
    # News Volume Timeline
    st.subheader("News Volume Timeline")
    if not news_df.empty and 'published_at' in news_df.columns:
        volume_fig = plot_volume_timeline(news_df, 'published_at', "News Volume Over Time")
        st.plotly_chart(volume_fig, use_container_width=True)
    else:
        st.info("No news volume timeline data available")
    
    # Social Media Analysis
    st.subheader("Social Media Analysis")
    if not social_df.empty:
        col1, col2 = st.columns(2)
        
        with col1:
            # Social Media Sentiment
            if 'sentiment_category' in social_df.columns:
                social_sentiment_fig = plot_sentiment_distribution(social_df, "Social Media Sentiment")
                st.plotly_chart(social_sentiment_fig, use_container_width=True)
            else:
                st.info("No social media sentiment data available")
        
        with col2:
            # Social Media Platform Distribution
            if 'platform' in social_df.columns:
                platform_counts = social_df['platform'].value_counts().reset_index()
                platform_counts.columns = ['platform', 'count']
                
                # Create simple pie chart for platforms
                import plotly.express as px
                platform_fig = px.pie(
                    platform_counts, 
                    values='count', 
                    names='platform',
                    title="Social Media Platforms"
                )
                st.plotly_chart(platform_fig, use_container_width=True)
            else:
                st.info("No platform distribution data available")
        
        # Top Influencers
        st.subheader("Top Social Media Influencers")
        influencers = identify_influencers(social_df, top_n=10)
        
        if influencers:
            # Display top influencers
            influencer_df = pd.DataFrame(influencers)
            
            # Select columns to display
            display_cols = ['username', 'user_followers', 'total_engagement', 'content']
            display_cols = [col for col in display_cols if col in influencer_df.columns]
            
            if display_cols:
                # Rename columns for better display
                column_map = {
                    'username': 'Username',
                    'user_followers': 'Followers',
                    'total_engagement': 'Total Engagement',
                    'content': 'Post Count'
                }
                
                influencer_df = influencer_df[display_cols].rename(columns=column_map)
                st.dataframe(influencer_df, use_container_width=True)
                
                # Create a bar chart of top influencers by engagement
                if 'Total Engagement' in influencer_df.columns and 'Username' in influencer_df.columns:
                    influencer_fig = plot_top_entities(
                        influencer_df.to_dict('records'),
                        'Total Engagement',
                        'Username',
                        'Top Influencers by Engagement'
                    )
                    st.plotly_chart(influencer_fig, use_container_width=True)
            else:
                st.info("Influencer data available but missing key columns")
        else:
            st.info("No influencer data available")
    else:
        st.info("No social media data available for analysis")
    
    # Top Keywords
    st.subheader("Top Keywords")
    col1, col2 = st.columns(2)
    
    with col1:
        # News Keywords
        if not news_df.empty and 'content' in news_df.columns:
            news_keywords = extract_keywords(news_df, 'content', top_n=15)
            
            if news_keywords:
                # Create DataFrame for display
                keyword_df = pd.DataFrame(news_keywords)
                st.markdown("**Top News Keywords**")
                st.dataframe(keyword_df, use_container_width=True)
            else:
                st.info("No news keywords data available")
        else:
            st.info("No news content data available for keyword extraction")
    
    with col2:
        # Social Media Keywords
        if not social_df.empty and 'content' in social_df.columns:
            social_keywords = extract_keywords(social_df, 'content', top_n=15)
            
            if social_keywords:
                # Create DataFrame for display
                social_keyword_df = pd.DataFrame(social_keywords)
                st.markdown("**Top Social Media Keywords**")
                st.dataframe(social_keyword_df, use_container_width=True)
            else:
                st.info("No social media keywords data available")
        else:
            st.info("No social media content data available for keyword extraction")
    
    # Display a sample of the latest news
    st.subheader("Latest News Articles")
    if not news_df.empty:
        # Sort by published date (most recent first)
        if 'published_at' in news_df.columns:
            latest_news = news_df.sort_values('published_at', ascending=False).head(5)
        else:
            latest_news = news_df.head(5)
        
        # Display each article in an expander
        for i, (_, article) in enumerate(latest_news.iterrows()):
            # Get article details
            title = article.get('title', f"Article {i+1}")
            source = article.get('source', 'Unknown')
            date = article.get('published_at', '')
            content = article.get('content', 'No content available')
            url = article.get('url', '')
            
            # Format date if available
            if date:
                try:
                    date = pd.to_datetime(date).strftime('%Y-%m-%d %H:%M')
                except:
                    pass
            
            # Create expander
            with st.expander(f"{title} ({source}, {date})"):
                st.markdown(f"**Source:** {source}")
                st.markdown(f"**Published:** {date}")
                st.markdown(f"**Content:**\n{content[:500]}...")
                if url:
                    st.markdown(f"[Read full article]({url})")
    else:
        st.info("No news articles available")
    
    # Database Stats
    st.markdown("---")
    st.subheader("Database Statistics")
    
    # Get database stats using SQLAlchemy
    from utils.database import Session
    from sqlalchemy import func, select
    from utils.database import NewsArticle, SocialMediaPost, Keyword, TopicModel, Topic
    
    session = Session()
    
    try:
        # Get counts
        news_count = session.query(func.count(NewsArticle.id)).scalar() or 0
        social_count = session.query(func.count(SocialMediaPost.id)).scalar() or 0
        keyword_count = session.query(func.count(Keyword.id)).scalar() or 0
        topic_model_count = session.query(func.count(TopicModel.id)).scalar() or 0
        
        # Display metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("News Articles in DB", news_count)
        
        with col2:
            st.metric("Social Posts in DB", social_count)
        
        with col3:
            st.metric("Keywords in DB", keyword_count)
        
        with col4:
            st.metric("Topic Models in DB", topic_model_count)
            
        # Show storage status
        st.caption("Database is being used to store all collected data for persistent access and analysis.")
    
    except Exception as e:
        st.error(f"Error getting database statistics: {e}")
    finally:
        session.close()
    
    # Footer with refresh time
    st.markdown("---")
    st.markdown(f"Dashboard last updated at {st.session_state.last_update.strftime('%Y-%m-%d %H:%M:%S')}")
