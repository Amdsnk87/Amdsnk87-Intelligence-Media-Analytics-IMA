import streamlit as st
import pandas as pd
from datetime import datetime, timedelta

from utils.data_loader import get_data
from utils.analysis import search_content, extract_keywords
from utils.visualization import plot_sentiment_distribution, plot_topic_distribution
from utils.database import store_keywords

def show_search(
    start_date,
    end_date,
    news_sources,
    social_sources,
    regions,
    topics
):
    """
    Display the search and analysis page.
    
    Args:
        start_date: Start date for filtering data
        end_date: End date for filtering data
        news_sources: List of selected news sources
        social_sources: List of selected social media platforms
        regions: List of selected regions
        topics: List of selected topics
    """
    st.header("Search & Content Analysis")
    
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
    
    # Search interface
    st.subheader("Search Content")
    
    # Search bar
    search_query = st.text_input("Enter search terms", 
                                key="search_input", 
                                help="Search for keywords across news and social media content")
    
    # Select content type
    content_type = st.radio("Content type", ["All", "News", "Social Media"], horizontal=True)
    
    # Advanced options expander
    with st.expander("Advanced Search Options"):
        # Date range within the global filter
        search_start_date = st.date_input("Search from date", 
                                        value=start_date,
                                        min_value=start_date,
                                        max_value=end_date)
        
        search_end_date = st.date_input("Search to date", 
                                       value=end_date,
                                       min_value=search_start_date,
                                       max_value=end_date)
        
        # Source selection based on global filter
        if content_type in ["All", "News"]:
            search_news_sources = st.multiselect(
                "News sources", 
                news_sources,
                default=news_sources
            )
        else:
            search_news_sources = []
            
        if content_type in ["All", "Social Media"]:
            search_social_sources = st.multiselect(
                "Social media platforms", 
                social_sources,
                default=social_sources
            )
        else:
            search_social_sources = []
        
        # Region and topic selection
        search_regions = st.multiselect(
            "Regions", 
            regions,
            default=regions
        )
        
        search_topics = st.multiselect(
            "Topics", 
            topics,
            default=topics
        )
    
    # Search button
    search_button = st.button("Search", type="primary")
    
    # Add to search history
    if search_button and search_query:
        if 'search_history' not in st.session_state:
            st.session_state.search_history = []
        
        # Add to history if not already present
        if search_query not in st.session_state.search_history:
            st.session_state.search_history.insert(0, search_query)
            # Keep only the 10 most recent searches
            st.session_state.search_history = st.session_state.search_history[:10]
    
    # Display search history
    if 'search_history' in st.session_state and st.session_state.search_history:
        st.markdown("**Recent searches:**")
        history_cols = st.columns(5)
        
        for i, query in enumerate(st.session_state.search_history):
            col_idx = i % 5
            if history_cols[col_idx].button(query, key=f"history_{i}", use_container_width=True):
                # Set the search input to this historical query
                st.session_state.search_input = query
                # Trigger a rerun to update the UI
                st.rerun()
    
    # Perform search when button is clicked
    if search_button or search_query:
        st.markdown("---")
        st.subheader(f"Search Results for: '{search_query}'")
        
        # Filter data based on date range
        search_start_datetime = datetime.combine(search_start_date, datetime.min.time())
        search_end_datetime = datetime.combine(search_end_date, datetime.max.time())
        
        # Filter news data
        filtered_news_df = pd.DataFrame()
        if content_type in ["All", "News"] and not news_df.empty:
            # Apply date filter
            if 'published_at' in news_df.columns:
                news_df['published_at'] = pd.to_datetime(news_df['published_at'], errors='coerce')
                date_filtered_news = news_df[
                    (news_df['published_at'] >= search_start_datetime) & 
                    (news_df['published_at'] <= search_end_datetime)
                ]
            else:
                date_filtered_news = news_df
            
            # Apply source filter
            if search_news_sources and 'source' in date_filtered_news.columns:
                source_filtered_news = date_filtered_news[date_filtered_news['source'].isin(search_news_sources)]
            else:
                source_filtered_news = date_filtered_news
            
            # Apply region filter
            if search_regions and 'region' in source_filtered_news.columns and 'All Indonesia' not in search_regions:
                region_filtered_news = source_filtered_news[source_filtered_news['region'].isin(search_regions)]
            else:
                region_filtered_news = source_filtered_news
            
            # Apply topic filter
            if search_topics and 'category' in region_filtered_news.columns:
                topic_filtered_news = region_filtered_news[region_filtered_news['category'].isin(search_topics)]
            else:
                topic_filtered_news = region_filtered_news
            
            # Search in content and title
            if search_query:
                columns_to_search = []
                for col in ['title', 'content', 'description']:
                    if col in topic_filtered_news.columns:
                        columns_to_search.append(col)
                
                if columns_to_search:
                    filtered_news_df = search_content(topic_filtered_news, search_query, columns_to_search)
                else:
                    filtered_news_df = topic_filtered_news
            else:
                filtered_news_df = topic_filtered_news
        
        # Filter social media data
        filtered_social_df = pd.DataFrame()
        if content_type in ["All", "Social Media"] and not social_df.empty:
            # Apply date filter
            if 'posted_at' in social_df.columns:
                social_df['posted_at'] = pd.to_datetime(social_df['posted_at'], errors='coerce')
                date_filtered_social = social_df[
                    (social_df['posted_at'] >= search_start_datetime) & 
                    (social_df['posted_at'] <= search_end_datetime)
                ]
            else:
                date_filtered_social = social_df
            
            # Apply platform filter
            if search_social_sources and 'platform' in date_filtered_social.columns:
                platform_filtered_social = date_filtered_social[date_filtered_social['platform'].isin(search_social_sources)]
            else:
                platform_filtered_social = date_filtered_social
            
            # Apply region filter
            if search_regions and 'region' in platform_filtered_social.columns and 'All Indonesia' not in search_regions:
                region_filtered_social = platform_filtered_social[platform_filtered_social['region'].isin(search_regions)]
            else:
                region_filtered_social = platform_filtered_social
            
            # Apply topic filter
            if search_topics and 'category' in region_filtered_social.columns:
                topic_filtered_social = region_filtered_social[region_filtered_social['category'].isin(search_topics)]
            else:
                topic_filtered_social = region_filtered_social
            
            # Search in content
            if search_query and 'content' in topic_filtered_social.columns:
                filtered_social_df = search_content(topic_filtered_social, search_query, ['content'])
            else:
                filtered_social_df = topic_filtered_social
        
        # Display results count
        news_count = len(filtered_news_df)
        social_count = len(filtered_social_df)
        
        col1, col2 = st.columns(2)
        col1.metric("News Results", news_count)
        col2.metric("Social Media Results", social_count)
        
        if news_count == 0 and social_count == 0:
            st.info(f"No results found for '{search_query}' with the current filters. Try broadening your search criteria.")
        else:
            # Analysis of search results
            st.markdown("### Analysis of Search Results")
            
            col1, col2 = st.columns(2)
            
            # Sentiment distribution of search results
            with col1:
                # News sentiment
                if not filtered_news_df.empty and 'sentiment_category' in filtered_news_df.columns:
                    news_sentiment_fig = plot_sentiment_distribution(filtered_news_df, "News Sentiment for Search Results")
                    st.plotly_chart(news_sentiment_fig, use_container_width=True)
                else:
                    st.info("No news sentiment data available for this search")
            
            with col2:
                # Social media sentiment
                if not filtered_social_df.empty and 'sentiment_category' in filtered_social_df.columns:
                    social_sentiment_fig = plot_sentiment_distribution(filtered_social_df, "Social Media Sentiment for Search Results")
                    st.plotly_chart(social_sentiment_fig, use_container_width=True)
                else:
                    st.info("No social media sentiment data available for this search")
            
            # Topic distribution of search results
            col1, col2 = st.columns(2)
            
            with col1:
                # News topics
                if not filtered_news_df.empty and 'category' in filtered_news_df.columns:
                    news_topic_fig = plot_topic_distribution(filtered_news_df, "News Topics for Search Results")
                    st.plotly_chart(news_topic_fig, use_container_width=True)
                else:
                    st.info("No news topic data available for this search")
            
            with col2:
                # Social media topics
                if not filtered_social_df.empty and 'category' in filtered_social_df.columns:
                    social_topic_fig = plot_topic_distribution(filtered_social_df, "Social Media Topics for Search Results")
                    st.plotly_chart(social_topic_fig, use_container_width=True)
                else:
                    st.info("No social media topic data available for this search")
            
            # Keywords from search results
            col1, col2 = st.columns(2)
            
            with col1:
                # News keywords
                if not filtered_news_df.empty and any(col in filtered_news_df.columns for col in ['content', 'title']):
                    text_col = 'content' if 'content' in filtered_news_df.columns else 'title'
                    news_keywords = extract_keywords(filtered_news_df, text_col, top_n=10)
                    
                    if news_keywords:
                        st.markdown("**Top Keywords in News Results**")
                        keyword_df = pd.DataFrame(news_keywords)
                        st.dataframe(keyword_df, use_container_width=True)
                    else:
                        st.info("No keywords extracted from news results")
                else:
                    st.info("No news content available for keyword extraction")
            
            with col2:
                # Social media keywords
                if not filtered_social_df.empty and 'content' in filtered_social_df.columns:
                    social_keywords = extract_keywords(filtered_social_df, 'content', top_n=10)
                    
                    if social_keywords:
                        st.markdown("**Top Keywords in Social Media Results**")
                        social_keyword_df = pd.DataFrame(social_keywords)
                        st.dataframe(social_keyword_df, use_container_width=True)
                    else:
                        st.info("No keywords extracted from social media results")
                else:
                    st.info("No social media content available for keyword extraction")
            
            # Display search results
            tabs = st.tabs(["News Results", "Social Media Results"])
            
            # News results
            with tabs[0]:
                if not filtered_news_df.empty:
                    # Sort by date (most recent first)
                    if 'published_at' in filtered_news_df.columns:
                        display_news = filtered_news_df.sort_values('published_at', ascending=False)
                    else:
                        display_news = filtered_news_df
                    
                    # Display a summary of number of articles found
                    st.markdown(f"### Found {len(display_news)} news articles matching your search criteria")
                    st.markdown("Showing up to 100 most recent articles.")
                    
                    # Display each article (up to 100)
                    for i, (_, article) in enumerate(display_news.head(100).iterrows()):
                        # Get article details
                        title = article.get('title', f"Article {i+1}")
                        source = article.get('source', 'Unknown')
                        date = article.get('published_at', '')
                        content = article.get('content', 'No content available')
                        url = article.get('url', '')
                        sentiment = article.get('sentiment_category', '')
                        
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
                        }.get(sentiment, '')
                        
                        expander_title = f"{title} ({source}, {date})"
                        if sentiment:
                            expander_title += f" - {sentiment_color}"
                        
                        with st.expander(expander_title):
                            st.markdown(f"**Source:** {source}")
                            st.markdown(f"**Published:** {date}")
                            
                            # Highlight search query in content if present
                            if search_query and content:
                                # Use markdown to highlight the query
                                # This simple approach just adds emphasis to the query
                                highlighted_content = content.replace(search_query, f"**{search_query}**")
                                st.markdown(f"**Content:**\n{highlighted_content[:1000]}...")
                            else:
                                st.markdown(f"**Content:**\n{content[:1000]}...")
                            
                            if url:
                                st.markdown(f"[Read full article]({url})")
                else:
                    st.info("No news results found matching your search criteria")
            
            # Social media results
            with tabs[1]:
                if not filtered_social_df.empty:
                    # Sort by date (most recent first)
                    if 'posted_at' in filtered_social_df.columns:
                        display_social = filtered_social_df.sort_values('posted_at', ascending=False)
                    else:
                        display_social = filtered_social_df
                    
                    # Display a summary of number of social media posts found
                    st.markdown(f"### Found {len(display_social)} social media posts matching your search criteria")
                    st.markdown("Showing up to 100 most recent posts.")
                    
                    # Display each post (up to 100)
                    for i, (_, post) in enumerate(display_social.head(100).iterrows()):
                        # Get post details
                        username = post.get('username', f"User {i+1}")
                        platform = post.get('platform', 'Unknown')
                        date = post.get('posted_at', '')
                        content = post.get('content', 'No content available')
                        url = post.get('url', '')
                        sentiment = post.get('sentiment_category', '')
                        
                        # Format date if available
                        if date:
                            try:
                                date = pd.to_datetime(date).strftime('%Y-%m-%d %H:%M')
                            except:
                                pass
                        
                        # Create expander for the post
                        sentiment_color = {
                            'positive': ':green[Positive]',
                            'neutral': ':blue[Neutral]',
                            'negative': ':red[Negative]'
                        }.get(sentiment, '')
                        
                        expander_title = f"{username} on {platform} ({date})"
                        if sentiment:
                            expander_title += f" - {sentiment_color}"
                        
                        with st.expander(expander_title):
                            st.markdown(f"**User:** {username}")
                            st.markdown(f"**Platform:** {platform}")
                            st.markdown(f"**Posted:** {date}")
                            
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
                            
                            # Highlight search query in content if present
                            if search_query and content:
                                # Use markdown to highlight the query
                                highlighted_content = content.replace(search_query, f"**{search_query}**")
                                st.markdown(f"**Content:**\n{highlighted_content}")
                            else:
                                st.markdown(f"**Content:**\n{content}")
                            
                            if url:
                                st.markdown(f"[View original post]({url})")
                else:
                    st.info("No social media results found matching your search criteria")
    
    # If no search is performed yet, show sample analysis
    else:
        st.markdown("---")
        st.info("Enter a search query and click 'Search' to analyze content across news and social media.")
        
        # Display sample news and social media content
        st.subheader("Sample Content Preview")
        
        tabs = st.tabs(["News Sample", "Social Media Sample"])
        
        # News sample
        with tabs[0]:
            if not news_df.empty:
                # Display a few sample articles
                sample_news = news_df.head(3)
                
                for i, (_, article) in enumerate(sample_news.iterrows()):
                    # Get article details
                    title = article.get('title', f"Article {i+1}")
                    source = article.get('source', 'Unknown')
                    date = article.get('published_at', '')
                    
                    # Format date if available
                    if date:
                        try:
                            date = pd.to_datetime(date).strftime('%Y-%m-%d %H:%M')
                        except:
                            pass
                    
                    st.markdown(f"**{title}** ({source}, {date})")
            else:
                st.info("No news data available. Please refresh data from the dashboard.")
        
        # Social media sample
        with tabs[1]:
            if not social_df.empty:
                # Display a few sample posts
                sample_social = social_df.head(3)
                
                for i, (_, post) in enumerate(sample_social.iterrows()):
                    # Get post details
                    username = post.get('username', f"User {i+1}")
                    platform = post.get('platform', 'Unknown')
                    date = post.get('posted_at', '')
                    content = post.get('content', 'No content available')
                    
                    # Format date if available
                    if date:
                        try:
                            date = pd.to_datetime(date).strftime('%Y-%m-%d %H:%M')
                        except:
                            pass
                    
                    st.markdown(f"**{username}** on {platform} ({date})")
                    st.markdown(f"{content[:100]}...")
            else:
                st.info("No social media data available. Please refresh data from the dashboard.")
