"""
Advanced influencer identification and tracking functionality.
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Any, Tuple, Optional
import networkx as nx
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from collections import Counter

def calculate_engagement_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate comprehensive engagement metrics for social media users.
    
    Args:
        df: DataFrame with social media data
        
    Returns:
        DataFrame with engagement metrics
    """
    # Check required columns exist
    required_columns = ['username', 'content']
    for col in required_columns:
        if col not in df.columns:
            print(f"Required column '{col}' not found in DataFrame")
            return df  # Return original dataframe if missing columns
    
    # Create new dataframe for metrics or use existing if possible
    result = df.copy()
    
    # Calculate user post counts
    user_post_counts = df['username'].value_counts().reset_index()
    user_post_counts.columns = ['username', 'post_count']
    
    # Calculate average engagement per user if metrics available
    engagement_metrics = ['likes', 'shares', 'comments', 'retweets', 'favorites']
    available_metrics = [col for col in engagement_metrics if col in df.columns]
    
    # Calculate total engagement if any metrics are available
    if available_metrics:
        # Add total engagement column
        df['total_engagement'] = df[available_metrics].sum(axis=1)
        
        # Calculate average engagement per user
        avg_engagement = df.groupby('username')['total_engagement'].mean().reset_index()
        avg_engagement.columns = ['username', 'avg_engagement']
        
        # Calculate total engagement per user
        total_engagement = df.groupby('username')['total_engagement'].sum().reset_index()
        total_engagement.columns = ['username', 'total_engagement']
    else:
        # If no engagement metrics, use post count as a proxy
        total_engagement = user_post_counts.copy()
        total_engagement.columns = ['username', 'total_engagement']
        total_engagement['total_engagement'] = total_engagement['total_engagement'] * 1.0  # Convert to float
        
        avg_engagement = user_post_counts.copy()
        avg_engagement.columns = ['username', 'avg_engagement'] 
        avg_engagement['avg_engagement'] = 1.0  # Default value
    
    # Merge metrics into one dataframe
    metrics_df = user_post_counts.merge(avg_engagement, on='username', how='left')
    metrics_df = metrics_df.merge(total_engagement, on='username', how='left')
    
    # Add followers data if available
    if 'user_followers' in df.columns:
        followers = df.groupby('username')['user_followers'].first().reset_index()
        metrics_df = metrics_df.merge(followers, on='username', how='left')
    else:
        metrics_df['user_followers'] = np.nan
    
    # Calculate engagement rate (engagement / followers)
    if 'user_followers' in metrics_df.columns:
        metrics_df['engagement_rate'] = metrics_df['total_engagement'] / metrics_df['user_followers'].replace(0, 1)
    else:
        metrics_df['engagement_rate'] = np.nan
    
    # Sort by total engagement
    metrics_df = metrics_df.sort_values('total_engagement', ascending=False)
    
    return metrics_df

def identify_influencers(
    df: pd.DataFrame, 
    min_posts: int = 2,
    top_n: int = 20
) -> List[Dict[str, Any]]:
    """
    Identify top social media influencers based on engagement metrics.
    
    Args:
        df: DataFrame with social media data
        min_posts: Minimum number of posts to consider a user
        top_n: Number of top influencers to return
        
    Returns:
        List of dictionaries with influencer data
    """
    # Skip if dataframe is empty or missing username
    if df.empty or 'username' not in df.columns:
        return []
    
    # Calculate comprehensive metrics
    metrics_df = calculate_engagement_metrics(df)
    
    # Filter by minimum post count
    metrics_df = metrics_df[metrics_df['post_count'] >= min_posts]
    
    # If no users meet criteria
    if metrics_df.empty:
        return []
    
    # Select top influencers
    top_influencers = metrics_df.head(top_n)
    
    # Convert to list of dictionaries
    result = top_influencers.to_dict(orient='records')
    
    return result

def analyze_influencer_topics(
    df: pd.DataFrame,
    influencers: List[Dict[str, Any]],
    text_column: str = 'content'
) -> List[Dict[str, Any]]:
    """
    Analyze topics and themes for each influencer based on their content.
    
    Args:
        df: DataFrame with social media data
        influencers: List of influencer dictionaries
        text_column: Column containing text content
        
    Returns:
        Enriched list of influencer dictionaries with topic information
    """
    from utils.analysis import extract_keywords
    
    # Skip if data is missing
    if df.empty or 'username' not in df.columns or text_column not in df.columns:
        return influencers
    
    result = []
    
    for influencer in influencers:
        username = influencer.get('username')
        if not username:
            result.append(influencer)
            continue
        
        # Filter posts by this influencer
        inf_posts = df[df['username'] == username]
        
        if inf_posts.empty:
            result.append(influencer)
            continue
        
        # Extract top keywords for this influencer
        inf_keywords = extract_keywords(inf_posts, text_column, top_n=5)
        
        # Copy the influencer dict and add keywords
        inf_copy = influencer.copy()
        inf_copy['top_topics'] = [k['keyword'] for k in inf_keywords]
        
        result.append(inf_copy)
    
    return result

def calculate_influencer_network(
    df: pd.DataFrame,
    min_connection_strength: int = 2,
    max_connections: int = 100
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Calculate influencer network graph based on content similarity.
    
    Args:
        df: DataFrame with social media data
        min_connection_strength: Minimum number of common terms to consider a connection
        max_connections: Maximum number of connections to return
        
    Returns:
        Tuple of (nodes, edges) for network graph
    """
    # Skip if dataframe is empty or missing required columns
    if df.empty or 'username' not in df.columns or 'content' not in df.columns:
        return [], []
    
    # Get top users
    top_users = df['username'].value_counts().head(30).index.tolist()
    user_df = df[df['username'].isin(top_users)]
    
    if user_df.empty:
        return [], []
    
    # Calculate TF-IDF vectors for each user's content
    user_content = user_df.groupby('username')['content'].apply(' '.join).reset_index()
    
    # Calculate TF-IDF
    vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
    
    try:
        tfidf_matrix = vectorizer.fit_transform(user_content['content'])
        
        # Calculate similarity between users (dot product of TF-IDF vectors)
        similarity = (tfidf_matrix * tfidf_matrix.T).toarray()
        np.fill_diagonal(similarity, 0)  # Remove self-connections
        
        # Create network
        G = nx.Graph()
        
        # Add nodes
        for username in user_content['username']:
            G.add_node(username)
        
        # Add edges based on similarity
        edges = []
        for i, user1 in enumerate(user_content['username']):
            for j, user2 in enumerate(user_content['username']):
                if i < j and similarity[i, j] > 0.1:  # Threshold for connection
                    weight = similarity[i, j]
                    G.add_edge(user1, user2, weight=weight)
                    edges.append((user1, user2, weight))
        
        # Sort edges by weight and keep only the top ones
        edges.sort(key=lambda x: x[2], reverse=True)
        edges = edges[:max_connections]
        
        # Calculate centrality metrics
        betweenness = nx.betweenness_centrality(G)
        closeness = nx.closeness_centrality(G)
        eigenvector = nx.eigenvector_centrality(G, max_iter=1000, tol=1e-06, weight='weight')
        
        # Create nodes list with centrality data
        nodes = []
        for user in user_content['username']:
            nodes.append({
                'id': user,
                'name': user,
                'betweenness': betweenness.get(user, 0),
                'closeness': closeness.get(user, 0),
                'eigenvector': eigenvector.get(user, 0),
                'connections': len(list(G.neighbors(user)))
            })
        
        # Sort nodes by eigenvector centrality
        nodes.sort(key=lambda x: x['eigenvector'], reverse=True)
        
        # Format edges for visualization
        formatted_edges = [
            {'source': source, 'target': target, 'value': weight}
            for source, target, weight in edges
        ]
        
        return nodes, formatted_edges
    except Exception as e:
        print(f"Error in influencer network calculation: {e}")
        return [], []

def track_influencer_growth(
    current_df: pd.DataFrame,
    historical_df: pd.DataFrame,
    min_posts: int = 2
) -> List[Dict[str, Any]]:
    """
    Track influencer growth by comparing current and historical data.
    
    Args:
        current_df: DataFrame with current social media data
        historical_df: DataFrame with historical social media data
        min_posts: Minimum number of posts to consider a user
        
    Returns:
        List of dictionaries with influencer growth data
    """
    # Skip if dataframes are empty
    if current_df.empty or historical_df.empty:
        return []
    
    # Calculate metrics for current and historical data
    current_metrics = calculate_engagement_metrics(current_df)
    historical_metrics = calculate_engagement_metrics(historical_df)
    
    # Filter by minimum post count
    current_metrics = current_metrics[current_metrics['post_count'] >= min_posts]
    historical_metrics = historical_metrics[historical_metrics['post_count'] >= min_posts]
    
    # If no data meets criteria
    if current_metrics.empty or historical_metrics.empty:
        return []
    
    # Merge current and historical metrics
    merged_metrics = current_metrics.merge(
        historical_metrics, 
        on='username', 
        how='left',
        suffixes=('_current', '_historical')
    )
    
    # Calculate growth metrics
    metrics = ['post_count', 'avg_engagement', 'total_engagement', 'user_followers']
    for metric in metrics:
        if f'{metric}_current' in merged_metrics.columns and f'{metric}_historical' in merged_metrics.columns:
            # Fill NA values with current metrics (for new influencers)
            merged_metrics[f'{metric}_historical'] = merged_metrics[f'{metric}_historical'].fillna(
                merged_metrics[f'{metric}_current'] / 2  # Assume some growth for new influencers
            )
            
            # Calculate growth rate
            merged_metrics[f'{metric}_growth'] = (
                merged_metrics[f'{metric}_current'] / merged_metrics[f'{metric}_historical'].replace(0, 1)
            )
            
            # Calculate growth percentage
            merged_metrics[f'{metric}_growth_pct'] = (
                (merged_metrics[f'{metric}_current'] - merged_metrics[f'{metric}_historical']) /
                merged_metrics[f'{metric}_historical'].replace(0, 1) * 100
            )
    
    # Sort by engagement growth
    if 'total_engagement_growth' in merged_metrics.columns:
        merged_metrics = merged_metrics.sort_values('total_engagement_growth', ascending=False)
    
    # Convert to list of dictionaries
    result = merged_metrics.to_dict(orient='records')
    
    return result

def identify_niche_influencers(
    df: pd.DataFrame, 
    topic_column: str = 'category',
    text_column: str = 'content',
    min_posts: int = 2,
    top_n_per_topic: int = 5
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Identify top influencers for specific niches/topics.
    
    Args:
        df: DataFrame with social media data
        topic_column: Column containing topic/category
        text_column: Column containing text content
        min_posts: Minimum number of posts to consider a user
        top_n_per_topic: Number of top influencers to return per topic
        
    Returns:
        Dictionary with topic as key and list of influencers as value
    """
    # Skip if dataframe is empty or missing required columns
    if df.empty or 'username' not in df.columns:
        return {}
    
    result = {}
    
    # If we have a topic column, use it for categorization
    if topic_column in df.columns:
        topics = df[topic_column].unique()
        
        for topic in topics:
            # Filter data for this topic
            topic_df = df[df[topic_column] == topic]
            
            if topic_df.empty:
                continue
            
            # Identify influencers for this topic
            influencers = identify_influencers(
                topic_df, 
                min_posts=min_posts,
                top_n=top_n_per_topic
            )
            
            if influencers:
                result[topic] = influencers
    
    # If no topic column or no results, try clustering
    if not result and text_column in df.columns:
        # Group content by user
        user_content = df.groupby('username')[text_column].apply(' '.join).reset_index()
        
        # Only process if we have enough users
        if len(user_content) >= 5:
            try:
                # Calculate TF-IDF vectors
                vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
                tfidf_matrix = vectorizer.fit_transform(user_content[text_column])
                
                # Determine optimal number of clusters (between 2 and 5)
                num_clusters = min(5, len(user_content) // 2)
                num_clusters = max(2, num_clusters)
                
                # Apply K-means clustering
                kmeans = KMeans(n_clusters=num_clusters, random_state=42)
                user_content['cluster'] = kmeans.fit_predict(tfidf_matrix)
                
                # Get top terms for each cluster
                cluster_terms = {}
                feature_names = vectorizer.get_feature_names_out()
                cluster_centers = kmeans.cluster_centers_
                
                for i in range(num_clusters):
                    # Get top terms for this cluster
                    indices = cluster_centers[i].argsort()[-5:]  # Top 5 terms
                    top_terms = [feature_names[j] for j in indices]
                    cluster_terms[i] = top_terms
                
                # Create a mapping from username to cluster
                username_cluster = dict(zip(user_content['username'], user_content['cluster']))
                
                # Add cluster to original dataframe
                df_copy = df.copy()
                df_copy['cluster'] = df_copy['username'].map(username_cluster)
                
                # Now identify influencers for each cluster
                for cluster_id in range(num_clusters):
                    cluster_df = df_copy[df_copy['cluster'] == cluster_id]
                    
                    if cluster_df.empty:
                        continue
                    
                    # Identify influencers for this cluster
                    influencers = identify_influencers(
                        cluster_df, 
                        min_posts=min_posts,
                        top_n=top_n_per_topic
                    )
                    
                    if influencers:
                        # Use cluster terms as topic name
                        topic_name = f"Topic {cluster_id+1}: {', '.join(cluster_terms[cluster_id])}"
                        result[topic_name] = influencers
            except Exception as e:
                print(f"Error in niche influencer clustering: {e}")
    
    return result

def get_comprehensive_influencer_analysis(
    social_df: pd.DataFrame,
    historical_social_df: pd.DataFrame = None
) -> Dict[str, Any]:
    """
    Get comprehensive influencer analysis including:
    - Top influencers overall
    - Influencer network
    - Influencer growth trends
    - Niche influencers
    
    Args:
        social_df: DataFrame with current social media data
        historical_social_df: Optional DataFrame with historical social media data
        
    Returns:
        Dictionary with comprehensive influencer analysis
    """
    result = {}
    
    # Skip if dataframe is empty
    if social_df.empty:
        return {"error": "No social media data available for analysis"}
    
    # 1. Identify top influencers
    top_influencers = identify_influencers(social_df, min_posts=1, top_n=20)
    
    if top_influencers:
        # Enrich with topic information
        top_influencers = analyze_influencer_topics(social_df, top_influencers)
        result['top_influencers'] = top_influencers
    
    # 2. Calculate influencer network
    nodes, edges = calculate_influencer_network(social_df)
    
    if nodes and edges:
        result['influencer_network'] = {
            'nodes': nodes,
            'edges': edges
        }
    
    # 3. Track influencer growth if historical data is available
    if historical_social_df is not None and not historical_social_df.empty:
        growth_data = track_influencer_growth(social_df, historical_social_df, min_posts=1)
        
        if growth_data:
            result['influencer_growth'] = growth_data
    
    # 4. Identify niche influencers
    niche_influencers = identify_niche_influencers(social_df, min_posts=1)
    
    if niche_influencers:
        result['niche_influencers'] = niche_influencers
    
    return result