import streamlit as st
import pandas as pd
from datetime import datetime

from utils.data_loader import get_data

def show_all_articles(
    start_date,
    end_date,
    news_sources,
    social_sources,
    regions,
    topics
):
    """
    Display all collected news articles in a detailed list.
    
    Args:
        start_date: Start date for filtering data
        end_date: End date for filtering data
        news_sources: List of selected news sources
        social_sources: List of selected social media platforms
        regions: List of selected regions
        topics: List of selected topics
    """
    st.header("All Collected Articles")
    
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
    
    # If no news data, show message
    if news_df.empty:
        st.warning("No news articles available in the database. Please refresh data from the dashboard.")
        return
    
    # Count total articles
    total_articles = len(news_df)
    st.subheader(f"Total Articles: {total_articles}")
    
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
            
    # Add pagination if there are many articles
    if filtered_count > 50:
        st.markdown("---")
        st.info("Showing all available articles. Scroll up to see more.")