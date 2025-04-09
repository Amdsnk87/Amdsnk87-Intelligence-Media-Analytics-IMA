"""
Influencer identification and tracking component.
"""
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
import networkx as nx
from collections import Counter

from utils.data_loader import get_data
from utils.influencer_analytics import (
    calculate_engagement_metrics,
    identify_influencers,
    analyze_influencer_topics,
    calculate_influencer_network,
    track_influencer_growth,
    identify_niche_influencers,
    get_comprehensive_influencer_analysis
)

def show_influencer_tracking(
    start_date,
    end_date,
    news_sources,
    social_sources,
    regions,
    topics
):
    """
    Display influencer identification and tracking analysis.
    
    Args:
        start_date: Start date for filtering data
        end_date: End date for filtering data
        news_sources: List of selected news sources
        social_sources: List of selected social media platforms
        regions: List of selected regions
        topics: List of selected topics
    """
    st.header("Influencer Identification & Tracking")
    
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
    
    # Show error if no social media data
    if social_df.empty:
        st.warning("No social media data available for influencer analysis. Please adjust your filters or fetch more data.")
        return
    
    # Create historical data for comparison
    # Get data from 2 weeks before the current date range
    historical_start = start_date - timedelta(days=14)
    historical_end = start_date - timedelta(days=1)
    
    try:
        # Get historical data
        historical_news_df, historical_social_df = get_data(
            start_date=historical_start,
            end_date=historical_end,
            news_sources=news_sources,
            social_sources=social_sources,
            regions=regions,
            topics=topics,
            force_refresh=False
        )
    except:
        # If historical data retrieval fails, use empty DataFrame
        historical_news_df = pd.DataFrame()
        historical_social_df = pd.DataFrame()
    
    # Top Influencers Section
    st.subheader("Top Social Media Influencers")
    
    # Identify top influencers
    top_influencers = identify_influencers(social_df, min_posts=1, top_n=15)
    
    if top_influencers:
        # Enrich with topic information
        top_influencers = analyze_influencer_topics(social_df, top_influencers)
        
        # Display as table
        influencer_df = pd.DataFrame(top_influencers)
        
        # Select and rename columns
        display_cols = ['username', 'post_count', 'avg_engagement', 'total_engagement', 'user_followers', 'engagement_rate']
        display_cols = [col for col in display_cols if col in influencer_df.columns]
        
        if display_cols:
            # Rename columns for better display
            column_map = {
                'username': 'Username',
                'post_count': 'Posts',
                'avg_engagement': 'Avg. Engagement',
                'total_engagement': 'Total Engagement',
                'user_followers': 'Followers',
                'engagement_rate': 'Engagement Rate'
            }
            
            # Only rename columns that exist
            rename_cols = {k: v for k, v in column_map.items() if k in display_cols}
            display_df = influencer_df[display_cols].rename(columns=rename_cols)
            
            # Format numeric columns
            numeric_cols = ['Avg. Engagement', 'Total Engagement', 'Followers', 'Engagement Rate']
            for col in numeric_cols:
                if col in display_df.columns:
                    if col == 'Engagement Rate':
                        # Format as percentage
                        display_df[col] = display_df[col].apply(lambda x: f"{x*100:.2f}%" if pd.notnull(x) else "N/A")
                    else:
                        # Format with commas for thousands
                        display_df[col] = display_df[col].apply(lambda x: f"{x:,.0f}" if pd.notnull(x) else "N/A")
            
            # Show table
            st.dataframe(display_df, use_container_width=True)
            
            # Show bar chart of top 10 influencers by engagement
            st.subheader("Top Influencers by Engagement")
            
            # Use original dataframe for visualization
            if 'total_engagement' in influencer_df.columns and 'username' in influencer_df.columns:
                top10_df = influencer_df.head(10).sort_values('total_engagement')
                
                # Create bar chart
                fig = px.bar(
                    top10_df,
                    y='username',
                    x='total_engagement',
                    title="Top 10 Influencers by Total Engagement",
                    labels={'username': 'Username', 'total_engagement': 'Total Engagement'},
                    orientation='h'
                )
                
                # Show plot
                st.plotly_chart(fig, use_container_width=True)
            
            # Show topic analysis for top influencers
            st.subheader("Influencer Topic Analysis")
            
            if 'top_topics' in influencer_df.columns:
                # Create table of influencers and their top topics
                topic_df = influencer_df[['username', 'top_topics']].head(5)
                
                # Display each influencer's topics
                for _, row in topic_df.iterrows():
                    username = row['username']
                    topics = row['top_topics']
                    
                    if topics:
                        topics_str = ", ".join(topics)
                        st.markdown(f"**{username}**: {topics_str}")
                
                # Collect all topics for word cloud
                all_topics = []
                for topics_list in influencer_df['top_topics']:
                    if topics_list:
                        all_topics.extend(topics_list)
                
                if all_topics:
                    # Count topic frequencies
                    topic_counts = Counter(all_topics)
                    topic_df = pd.DataFrame({
                        'topic': list(topic_counts.keys()),
                        'count': list(topic_counts.values())
                    })
                    
                    # Create bar chart
                    fig = px.bar(
                        topic_df.sort_values('count', ascending=False).head(10),
                        x='topic',
                        y='count',
                        title="Most Common Topics Among Top Influencers",
                        labels={'topic': 'Topic', 'count': 'Frequency'}
                    )
                    
                    # Show plot
                    st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Limited influencer data available.")
    else:
        st.info("No influencer data available.")
    
    # Influencer Network Analysis
    st.subheader("Influencer Network Analysis")
    
    if not social_df.empty and 'username' in social_df.columns and 'content' in social_df.columns:
        # Calculate influencer network
        nodes, edges = calculate_influencer_network(social_df)
        
        if nodes and edges:
            # Display network statistics
            st.markdown(f"Network analysis of **{len(nodes)}** influencers with **{len(edges)}** connections")
            
            # Get top influencers by centrality
            top_central = sorted(nodes, key=lambda x: x['eigenvector'], reverse=True)[:5]
            
            # Display top central influencers
            st.markdown("### Most Central Influencers")
            st.markdown("These influencers have the most influence in the network based on their connections:")
            
            for i, node in enumerate(top_central):
                st.markdown(f"{i+1}. **{node['name']}** (Influence Score: {node['eigenvector']:.3f}, Connections: {node['connections']})")
            
            # Create network visualization with Plotly
            # (Simple network visualization as Plotly's capabilities are limited)
            
            # Get top edges for visualization (to avoid cluttering)
            top_edges = sorted(edges, key=lambda x: x['value'], reverse=True)[:min(30, len(edges))]
            
            # Create edge trace
            edge_x = []
            edge_y = []
            
            # Create a simple graph with networkx
            G = nx.Graph()
            
            # Add nodes
            node_names = [node['id'] for node in nodes]
            G.add_nodes_from(node_names)
            
            # Add edges
            for edge in top_edges:
                G.add_edge(edge['source'], edge['target'], weight=edge['value'])
            
            # Get positions
            pos = nx.spring_layout(G, seed=42)
            
            # Get traces for plotting
            edge_trace = go.Scatter(
                x=[],
                y=[],
                line=dict(width=0.5, color='#888'),
                hoverinfo='none',
                mode='lines')
            
            for edge in G.edges():
                x0, y0 = pos[edge[0]]
                x1, y1 = pos[edge[1]]
                edge_trace['x'] += (x0, x1, None)
                edge_trace['y'] += (y0, y1, None)
            
            node_trace = go.Scatter(
                x=[],
                y=[],
                text=[],
                mode='markers',
                hoverinfo='text',
                marker=dict(
                    showscale=True,
                    colorscale='YlGnBu',
                    size=[],
                    color=[],  # Initialize color as empty list
                    colorbar=dict(
                        thickness=15,
                        title='Node Connections',
                        xanchor='left',
                        title_side='right'  # Fixed: changed 'titleside' to 'title_side'
                    ),
                    line=dict(width=2)))
            
            for node in G.nodes():
                x, y = pos[node]
                node_trace['x'] += (x,)
                node_trace['y'] += (y,)
            
            # Color nodes by their number of connections
            for node, adjacencies in enumerate(G.adjacency()):
                node_trace['marker']['size'] += (len(adjacencies[1]) * 10 + 10,)
                node_info = next((n for n in nodes if n['id'] == adjacencies[0]), None)
                if node_info:
                    node_trace['marker']['color'] += (node_info['eigenvector'] * 100,)
                    node_trace['text'] += (f"{adjacencies[0]}<br>Connections: {len(adjacencies[1])}<br>Influence: {node_info['eigenvector']:.3f}",)
            
            # Create figure
            fig = go.Figure(data=[edge_trace, node_trace],
                        layout=go.Layout(
                            title=dict(
                                text="Influencer Network Map",
                                font=dict(size=16)
                            ),
                            showlegend=False,
                            hovermode='closest',
                            margin=dict(b=20, l=5, r=5, t=40),
                            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                        )
            
            # Show the network visualization
            st.plotly_chart(fig, use_container_width=True)
            
            # Network interpretation
            st.markdown("### Network Interpretation")
            st.markdown("""
            - **Larger circles** represent influencers with more connections
            - **Brighter colors** show influencers with higher eigenvector centrality (more influential)
            - **Connected nodes** indicate content similarity between influencers
            """)
        else:
            st.info("Not enough data for network analysis. Try expanding your date range or including more social sources.")
    else:
        st.info("Social media content data required for network analysis.")
    
    # Influencer Growth Tracking
    st.subheader("Influencer Growth Tracking")
    
    if not historical_social_df.empty and 'username' in historical_social_df.columns:
        # Track influencer growth
        growth_data = track_influencer_growth(social_df, historical_social_df, min_posts=1)
        
        if growth_data:
            # Create DataFrame
            growth_df = pd.DataFrame(growth_data)
            
            # Select columns to display
            display_cols = [
                'username', 
                'post_count_current', 
                'total_engagement_current', 
                'total_engagement_growth_pct'
            ]
            display_cols = [col for col in display_cols if col in growth_df.columns]
            
            if display_cols:
                # Rename columns
                column_map = {
                    'username': 'Username',
                    'post_count_current': 'Posts',
                    'total_engagement_current': 'Total Engagement',
                    'total_engagement_growth_pct': 'Engagement Growth %'
                }
                rename_cols = {k: v for k, v in column_map.items() if k in display_cols}
                display_growth_df = growth_df[display_cols].rename(columns=rename_cols)
                
                # Format percentage columns
                if 'Engagement Growth %' in display_growth_df.columns:
                    display_growth_df['Engagement Growth %'] = display_growth_df['Engagement Growth %'].apply(
                        lambda x: f"{x:.1f}%" if pd.notnull(x) else "N/A"
                    )
                
                # Display table
                st.dataframe(display_growth_df.head(10), use_container_width=True)
                
                # Create visualization of fastest growing influencers
                if 'total_engagement_growth_pct' in growth_df.columns:
                    # Get top 10 growth influencers
                    top_growth = growth_df.sort_values('total_engagement_growth_pct', ascending=False).head(10)
                    
                    # Create bar chart
                    fig = px.bar(
                        top_growth,
                        y='username',
                        x='total_engagement_growth_pct',
                        title="Fastest Growing Influencers (by Engagement)",
                        labels={'username': 'Username', 'total_engagement_growth_pct': 'Engagement Growth %'},
                        orientation='h',
                        color='total_engagement_growth_pct',
                        color_continuous_scale='Viridis'
                    )
                    
                    # Format x-axis as percentage
                    fig.update_layout(xaxis_tickformat=',.0%')
                    
                    # Show plot
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Show growth insights
                    fast_growing = top_growth.iloc[0]
                    st.success(f"🚀 **{fast_growing['username']}** shows the fastest growth with a {fast_growing['total_engagement_growth_pct']:.1f}% increase in engagement")
            else:
                st.info("Limited growth data available.")
        else:
            st.info("Not enough historical data to track influencer growth.")
    else:
        st.info("Historical data not available for growth tracking.")
    
    # Niche Influencer Identification
    st.subheader("Niche/Topic Influencers")
    
    if not social_df.empty and 'username' in social_df.columns:
        # Identify niche influencers
        niche_influencers = identify_niche_influencers(social_df, min_posts=1)
        
        if niche_influencers:
            # Create tabs for each niche
            tabs = st.tabs(list(niche_influencers.keys()))
            
            # For each niche
            for i, (topic, influencers) in enumerate(niche_influencers.items()):
                with tabs[i]:
                    if influencers:
                        # Create DataFrame
                        niche_df = pd.DataFrame(influencers)
                        
                        # Select columns to display
                        display_cols = ['username', 'post_count', 'total_engagement', 'user_followers']
                        display_cols = [col for col in display_cols if col in niche_df.columns]
                        
                        if display_cols:
                            # Rename columns
                            column_map = {
                                'username': 'Username',
                                'post_count': 'Posts',
                                'total_engagement': 'Total Engagement',
                                'user_followers': 'Followers'
                            }
                            rename_cols = {k: v for k, v in column_map.items() if k in display_cols}
                            display_niche_df = niche_df[display_cols].rename(columns=rename_cols)
                            
                            # Display table
                            st.dataframe(display_niche_df, use_container_width=True)
                            
                            # Create visualization
                            if 'username' in niche_df.columns and 'total_engagement' in niche_df.columns:
                                # Create bar chart
                                fig = px.bar(
                                    niche_df,
                                    x='username',
                                    y='total_engagement',
                                    title=f"Top Influencers for '{topic}'",
                                    labels={'username': 'Username', 'total_engagement': 'Engagement'}
                                )
                                
                                # Show plot
                                st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.info(f"Limited data available for {topic} influencers.")
                    else:
                        st.info(f"No influencers identified for {topic}.")
        else:
            st.info("No niche influencers identified. Try expanding your data range.")
    else:
        st.info("Social media data required for niche influencer identification.")
    
    # Information about methodology
    with st.expander("About Influencer Analysis"):
        st.markdown("""
        ### How Influencer Identification Works
        
        The influencer analysis uses several metrics and techniques:
        
        - **Engagement Metrics**: Calculated based on likes, comments, shares, and other interactions
        - **Network Analysis**: Uses content similarity to identify connections between influencers
        - **Topic Modeling**: Identifies key topics and themes for each influencer
        - **Growth Tracking**: Compares current metrics to historical data to identify rising influencers
        
        ### Definitions
        
        - **Total Engagement**: Sum of all interactions (likes, comments, shares, etc.)
        - **Engagement Rate**: Total engagement divided by follower count
        - **Influence Score**: Measure of an influencer's centrality in the network (eigenvector centrality)
        - **Niche Influencers**: Influencers who specialize in specific topics or categories
        
        ### Limitations
        
        - Analysis is limited by available data in the selected date range
        - Some metrics may be estimated if actual engagement data is unavailable
        - Network connections are based on content similarity, not actual social connections
        
        For best results, include multiple social media sources and a wider date range.
        """)