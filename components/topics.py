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
    topic_modeling,
    create_word_cloud_data
)
from utils.visualization import (
    plot_topic_distribution,
    plot_stacked_bar,
    plot_word_cloud_data,
    plot_radar_chart
)

def show_topics(
    start_date,
    end_date,
    news_sources,
    social_sources,
    regions,
    topics
):
    """
    Display topic classification and analysis for media content.
    
    Args:
        start_date: Start date for filtering data
        end_date: End date for filtering data
        news_sources: List of selected news sources
        social_sources: List of selected social media platforms
        regions: List of selected regions
        topics: List of selected topics
    """
    st.header("Topic Classification & Analysis")
    
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
    
    # Overall Topic Distribution
    st.subheader("Overall Topic Distribution")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # News topic distribution
        if not news_df.empty and 'category' in news_df.columns:
            news_topic_fig = plot_topic_distribution(news_df, "News Topic Distribution")
            st.plotly_chart(news_topic_fig, use_container_width=True)
        else:
            st.info("No news topic data available")
    
    with col2:
        # Social media topic distribution
        if not social_df.empty and 'category' in social_df.columns:
            social_topic_fig = plot_topic_distribution(social_df, "Social Media Topic Distribution")
            st.plotly_chart(social_topic_fig, use_container_width=True)
        else:
            st.info("No social media topic data available")
    
    # Topic by Source
    st.subheader("Topic Distribution by News Source")
    
    if not news_df.empty and 'category' in news_df.columns and 'source' in news_df.columns:
        # Group by source and category
        source_topic = news_df.groupby(['source', 'category']).size().reset_index(name='count')
        
        # Create stacked bar chart
        if not source_topic.empty:
            source_topic_fig = plot_stacked_bar(
                source_topic,
                'source',
                'count',
                'category',
                "Topic Distribution by News Source"
            )
            st.plotly_chart(source_topic_fig, use_container_width=True)
        else:
            st.info("No source topic data available")
    else:
        st.info("Source or topic data not available")
    
    # Topic by Region
    st.subheader("Topic Distribution by Region")
    
    if not news_df.empty and 'category' in news_df.columns and 'region' in news_df.columns:
        # Group by region and category
        region_topic = news_df.groupby(['region', 'category']).size().reset_index(name='count')
        
        # Create stacked bar chart
        if not region_topic.empty:
            region_topic_fig = plot_stacked_bar(
                region_topic,
                'region',
                'count',
                'category',
                "Topic Distribution by Region"
            )
            st.plotly_chart(region_topic_fig, use_container_width=True)
        else:
            st.info("No regional topic data available")
    else:
        st.info("Regional or topic data not available")
    
    # Topic Sentiment Analysis
    st.subheader("Topic Sentiment Analysis")
    
    if not news_df.empty and 'category' in news_df.columns and 'sentiment_category' in news_df.columns:
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
    
    # Automated Topic Modeling
    st.subheader("Automated Topic Discovery")
    
    # Let user select number of topics to extract
    n_topics = st.slider("Number of topics to discover", 3, 10, 5)
    
    # Topics in news content
    if not news_df.empty and 'content' in news_df.columns:
        with st.spinner("Discovering topics in news content..."):
            # Perform topic modeling
            news_topics = topic_modeling(news_df, 'content', n_topics=n_topics)
            
            if news_topics:
                st.markdown("#### Discovered Topics in News Content")
                
                # Display topics in a grid of cards
                # Calculate number of rows needed (2 topics per row)
                n_rows = (len(news_topics) + 1) // 2
                
                for row in range(n_rows):
                    cols = st.columns(2)
                    
                    for col_idx in range(2):
                        topic_idx = row * 2 + col_idx
                        
                        if topic_idx < len(news_topics):
                            topic = news_topics[topic_idx]
                            
                            with cols[col_idx]:
                                # Create topic card
                                st.markdown(f"**Topic {topic['id'] + 1}**")
                                
                                # Create word cloud for topic if available
                                if 'words' in topic and 'weights' in topic:
                                    # Prepare word cloud data
                                    word_data = []
                                    for word, weight in zip(topic['words'], topic['weights']):
                                        word_data.append({
                                            'text': word,
                                            'value': float(weight),
                                            'weight': float(weight)
                                        })
                                    
                                    if word_data:
                                        # Create word cloud
                                        word_cloud_chart = plot_word_cloud_data(word_data)
                                        st.altair_chart(word_cloud_chart, use_container_width=True)
                                
                                # Display top words
                                if 'words' in topic:
                                    st.markdown("**Top words:**")
                                    st.write(", ".join(topic['words'][:7]))
            else:
                st.info("Could not extract topics from news content")
    else:
        st.info("No news content available for topic modeling")
    
    # Topic Explorer - Allow users to explore content by topic
    st.subheader("Topic Content Explorer")
    
    # Create tabs for different categories
    if not news_df.empty and 'category' in news_df.columns:
        # Get unique categories
        categories = sorted(news_df['category'].unique())
        
        if categories:
            # Create tabs
            category_tabs = st.tabs(categories)
            
            # Display content for each category
            for i, category in enumerate(categories):
                with category_tabs[i]:
                    # Filter news by category
                    category_news = news_df[news_df['category'] == category]
                    
                    if not category_news.empty:
                        # Display category metrics
                        num_articles = len(category_news)
                        
                        # Sentiment distribution if available
                        if 'sentiment_category' in category_news.columns:
                            positive_pct = (category_news['sentiment_category'] == 'positive').mean() * 100
                            neutral_pct = (category_news['sentiment_category'] == 'neutral').mean() * 100
                            negative_pct = (category_news['sentiment_category'] == 'negative').mean() * 100
                            
                            # Display metrics in columns
                            metrics_cols = st.columns(4)
                            metrics_cols[0].metric("Articles", num_articles)
                            metrics_cols[1].metric("Positive", f"{positive_pct:.1f}%")
                            metrics_cols[2].metric("Neutral", f"{neutral_pct:.1f}%")
                            metrics_cols[3].metric("Negative", f"{negative_pct:.1f}%")
                        else:
                            # Just show the article count
                            st.metric("Articles", num_articles)
                        
                        # Extract keywords for this topic
                        if 'content' in category_news.columns:
                            topic_keywords = extract_keywords(category_news, 'content', top_n=20)
                            
                            if topic_keywords:
                                # Create word cloud data
                                word_cloud_data = []
                                for kw in topic_keywords:
                                    word_cloud_data.append({
                                        'text': kw['keyword'],
                                        'value': kw['count'],
                                        'weight': kw['count']
                                    })
                                
                                # Display word cloud
                                st.markdown("#### Key Terms")
                                word_cloud_chart = plot_word_cloud_data(word_cloud_data)
                                st.altair_chart(word_cloud_chart, use_container_width=True)
                        
                        # Display articles for this category
                        st.markdown("#### Recent Articles")
                        
                        # Sort by date (most recent first)
                        if 'published_at' in category_news.columns:
                            sorted_news = category_news.sort_values('published_at', ascending=False)
                        else:
                            sorted_news = category_news
                        
                        # Display articles
                        for i, (_, article) in enumerate(sorted_news.head(5).iterrows()):
                            # Get article details
                            title = article.get('title', f"Article {i+1}")
                            source = article.get('source', 'Unknown')
                            date = article.get('published_at', '')
                            sentiment = article.get('sentiment_category', '')
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
                            expander_title = f"{title} ({source}, {date})"
                            if sentiment:
                                expander_title += f" - {sentiment_color}"
                            
                            with st.expander(expander_title):
                                st.markdown(f"**Source:** {source}")
                                st.markdown(f"**Published:** {date}")
                                if sentiment:
                                    st.markdown(f"**Sentiment:** {sentiment_color}")
                                st.markdown(f"**Content:**\n{content[:500]}...")
                                if url:
                                    st.markdown(f"[Read full article]({url})")
                    else:
                        st.info(f"No articles found for the '{category}' category")
        else:
            st.info("No categories available in the news data")
    else:
        st.info("No category data available in the news content")
    
    # Topic Relations Analysis
    st.subheader("Topic Relations Analysis")
    
    if not news_df.empty and 'category' in news_df.columns:
        # Create a co-occurrence matrix of topics
        # For this, we need to find articles that mention multiple topics
        
        # First, create a radar chart showing topic distributions
        topic_counts = news_df['category'].value_counts().to_dict()
        
        if topic_counts:
            # Normalize values for radar chart
            max_count = max(topic_counts.values())
            normalized_counts = {k: (v / max_count) * 100 for k, v in topic_counts.items()}
            
            # Create radar chart
            radar_fig = plot_radar_chart(normalized_counts, "Topic Distribution Radar")
            st.plotly_chart(radar_fig, use_container_width=True)
        
        # Try to find topic co-occurrences in content
        if 'content' in news_df.columns:
            # Get unique topics
            unique_topics = sorted(news_df['category'].unique())
            
            # Initialize co-occurrence matrix
            co_occurrence = np.zeros((len(unique_topics), len(unique_topics)))
            
            # For each article, check which topics are mentioned in the content
            for _, article in news_df.iterrows():
                article_category = article.get('category', '')
                article_content = article.get('content', '').lower()
                
                # Skip if no content or category
                if not article_content or not article_category:
                    continue
                
                # Get index of main category
                main_idx = unique_topics.index(article_category) if article_category in unique_topics else -1
                
                if main_idx >= 0:
                    # Check for mentions of other topics in the content
                    for topic_idx, topic in enumerate(unique_topics):
                        # Skip self
                        if topic_idx == main_idx:
                            continue
                        
                        # Check if topic is mentioned in content
                        if topic.lower() in article_content:
                            co_occurrence[main_idx, topic_idx] += 1
            
            # Check if we have any co-occurrences
            if np.sum(co_occurrence) > 0:
                # Create heatmap for topic co-occurrences
                fig = px.imshow(
                    co_occurrence,
                    labels=dict(x="Mentioned Topic", y="Main Topic", color="Co-occurrences"),
                    x=unique_topics,
                    y=unique_topics,
                    color_continuous_scale="Reds",
                    title="Topic Co-occurrence Matrix"
                )
                
                fig.update_layout(
                    xaxis=dict(tickangle=45),
                    height=600
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                st.markdown("""
                **Note:** This matrix shows how often topics co-occur in articles. 
                The row represents the main topic of an article, while the column shows other topics mentioned in the content.
                """)
            else:
                st.info("No significant topic co-occurrences detected in the content")
        else:
            st.info("No content available for topic co-occurrence analysis")
    else:
        st.info("No topic data available for relations analysis")
    
    # Topic Comparison Between News and Social Media
    st.subheader("Topic Comparison: News vs. Social Media")
    
    if not news_df.empty and not social_df.empty and 'category' in news_df.columns and 'category' in social_df.columns:
        # Get topic counts for news
        news_topic_counts = news_df['category'].value_counts().reset_index()
        news_topic_counts.columns = ['category', 'count']
        news_topic_counts['source'] = 'News'
        
        # Get topic counts for social media
        social_topic_counts = social_df['category'].value_counts().reset_index()
        social_topic_counts.columns = ['category', 'count']
        social_topic_counts['source'] = 'Social Media'
        
        # Combine dataframes
        combined_counts = pd.concat([news_topic_counts, social_topic_counts])
        
        if not combined_counts.empty:
            # Create grouped bar chart
            fig = px.bar(
                combined_counts,
                x='category',
                y='count',
                color='source',
                barmode='group',
                title="Topic Distribution Comparison: News vs. Social Media"
            )
            
            fig.update_layout(
                xaxis_title="Topic",
                yaxis_title="Count",
                legend_title="Source",
                xaxis=dict(tickangle=45)
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No topic comparison data available")
    else:
        st.info("Topic data not available for both news and social media")
    
    # Footer with refresh time
    st.markdown("---")
    st.markdown(f"Data last updated at {st.session_state.last_update.strftime('%Y-%m-%d %H:%M:%S')}")
