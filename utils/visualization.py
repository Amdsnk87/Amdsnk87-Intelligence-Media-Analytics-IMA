import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import altair as alt
from datetime import datetime, timedelta
from typing import List, Dict, Any, Tuple, Optional

def plot_sentiment_distribution(df: pd.DataFrame, title: str = "Sentiment Distribution") -> go.Figure:
    """
    Create a pie chart showing distribution of sentiment categories.
    
    Args:
        df: DataFrame with sentiment_category column
        title: Chart title
        
    Returns:
        Plotly figure
    """
    if df.empty or 'sentiment_category' not in df.columns:
        # Return empty chart with message
        fig = go.Figure()
        fig.add_annotation(
            text="No sentiment data available",
            x=0.5, y=0.5,
            xref="paper", yref="paper",
            showarrow=False,
            font=dict(size=16)
        )
        fig.update_layout(title=title)
        return fig
    
    # Count sentiment categories
    sentiment_counts = df['sentiment_category'].value_counts().reset_index()
    sentiment_counts.columns = ['sentiment', 'count']
    
    # Color map
    colors = {'positive': '#4CAF50', 'neutral': '#FF9800', 'negative': '#F44336'}
    
    # Create pie chart
    fig = px.pie(
        sentiment_counts, 
        values='count', 
        names='sentiment',
        title=title,
        color='sentiment',
        color_discrete_map=colors
    )
    
    fig.update_layout(
        legend_title="Sentiment",
        legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5)
    )
    
    return fig

def plot_sentiment_timeline(df: pd.DataFrame, date_col: str, 
                           title: str = "Sentiment Over Time") -> go.Figure:
    """
    Create a line chart showing sentiment trends over time.
    
    Args:
        df: DataFrame with sentiment and date columns
        date_col: Name of date column
        title: Chart title
        
    Returns:
        Plotly figure
    """
    if df.empty or 'sentiment_category' not in df.columns or date_col not in df.columns:
        # Return empty chart with message
        fig = go.Figure()
        fig.add_annotation(
            text="No sentiment timeline data available",
            x=0.5, y=0.5,
            xref="paper", yref="paper",
            showarrow=False,
            font=dict(size=16)
        )
        fig.update_layout(title=title)
        return fig
    
    # Create a copy of the DataFrame
    df_copy = df.copy()
    
    # Convert date column to datetime
    df_copy[date_col] = pd.to_datetime(df_copy[date_col], errors='coerce')
    
    # Group by date and sentiment
    df_grouped = df_copy.groupby([pd.Grouper(key=date_col, freq='D'), 'sentiment_category']).size().reset_index(name='count')
    
    # Create line chart
    fig = px.line(
        df_grouped, 
        x=date_col, 
        y='count', 
        color='sentiment_category',
        title=title,
        color_discrete_map={'positive': '#4CAF50', 'neutral': '#FF9800', 'negative': '#F44336'}
    )
    
    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Count",
        legend_title="Sentiment",
        legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5)
    )
    
    return fig

def plot_topic_distribution(df: pd.DataFrame, title: str = "Topic Distribution") -> go.Figure:
    """
    Create a bar chart showing distribution of topics/categories.
    
    Args:
        df: DataFrame with category column
        title: Chart title
        
    Returns:
        Plotly figure
    """
    if df.empty or 'category' not in df.columns:
        # Return empty chart with message
        fig = go.Figure()
        fig.add_annotation(
            text="No topic data available",
            x=0.5, y=0.5,
            xref="paper", yref="paper",
            showarrow=False,
            font=dict(size=16)
        )
        fig.update_layout(title=title)
        return fig
    
    # Count categories
    category_counts = df['category'].value_counts().reset_index()
    category_counts.columns = ['category', 'count']
    
    # Sort by count
    category_counts = category_counts.sort_values('count', ascending=False)
    
    # Create horizontal bar chart
    fig = px.bar(
        category_counts, 
        y='category', 
        x='count',
        title=title,
        orientation='h',
        color='count',
        color_continuous_scale=px.colors.sequential.Blues
    )
    
    fig.update_layout(
        xaxis_title="Count",
        yaxis_title="Topic",
        coloraxis_showscale=False
    )
    
    return fig

def plot_media_share(df: pd.DataFrame, title: str = "Media Share") -> go.Figure:
    """
    Create a pie chart showing media share by source.
    
    Args:
        df: DataFrame with source column
        title: Chart title
        
    Returns:
        Plotly figure
    """
    if df.empty or 'source' not in df.columns:
        # Return empty chart with message
        fig = go.Figure()
        fig.add_annotation(
            text="No media source data available",
            x=0.5, y=0.5,
            xref="paper", yref="paper",
            showarrow=False,
            font=dict(size=16)
        )
        fig.update_layout(title=title)
        return fig
    
    # Count sources
    source_counts = df['source'].value_counts().reset_index()
    source_counts.columns = ['source', 'count']
    
    # Calculate percentage
    total = source_counts['count'].sum()
    source_counts['percentage'] = (source_counts['count'] / total * 100).round(1)
    
    # Create labels with percentages
    source_counts['label'] = source_counts['source'] + ' (' + source_counts['percentage'].astype(str) + '%)'
    
    # Create pie chart
    fig = px.pie(
        source_counts, 
        values='count', 
        names='label',
        title=title
    )
    
    fig.update_layout(
        legend_title="Source",
        legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5)
    )
    
    return fig

def plot_volume_timeline(df: pd.DataFrame, date_col: str, 
                        title: str = "Volume Over Time") -> go.Figure:
    """
    Create a line chart showing volume trends over time.
    
    Args:
        df: DataFrame with date column
        date_col: Name of date column
        title: Chart title
        
    Returns:
        Plotly figure
    """
    if df.empty or date_col not in df.columns:
        # Return empty chart with message
        fig = go.Figure()
        fig.add_annotation(
            text="No timeline data available",
            x=0.5, y=0.5,
            xref="paper", yref="paper",
            showarrow=False,
            font=dict(size=16)
        )
        fig.update_layout(title=title)
        return fig
    
    # Create a copy of the DataFrame
    df_copy = df.copy()
    
    # Convert date column to datetime
    df_copy[date_col] = pd.to_datetime(df_copy[date_col], errors='coerce')
    
    # Group by date
    df_grouped = df_copy.groupby(pd.Grouper(key=date_col, freq='D')).size().reset_index(name='count')
    
    # Create line chart
    fig = px.line(
        df_grouped, 
        x=date_col, 
        y='count',
        title=title,
        markers=True
    )
    
    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Count"
    )
    
    return fig

def plot_regional_heatmap(df: pd.DataFrame, title: str = "Regional Distribution") -> go.Figure:
    """
    Create a choropleth map for Indonesia showing regional distribution.
    
    Args:
        df: DataFrame with region column
        title: Chart title
        
    Returns:
        Plotly figure
    """
    if df.empty or 'region' not in df.columns:
        # Return empty chart with message
        fig = go.Figure()
        fig.add_annotation(
            text="No regional data available",
            x=0.5, y=0.5,
            xref="paper", yref="paper",
            showarrow=False,
            font=dict(size=16)
        )
        fig.update_layout(title=title)
        return fig
    
    # Count by region
    region_counts = df['region'].value_counts().reset_index()
    region_counts.columns = ['region', 'count']
    
    # Define Indonesia regions (provinces) with their geolocation center
    indonesia_regions = {
        "DKI Jakarta": {'lat': -6.2088, 'lon': 106.8456},
        "West Java": {'lat': -6.9175, 'lon': 107.6191},
        "Central Java": {'lat': -7.1510, 'lon': 110.1403},
        "East Java": {'lat': -7.5360, 'lon': 112.2384},
        "Bali": {'lat': -8.4095, 'lon': 115.1889},
        "North Sumatra": {'lat': 2.1154, 'lon': 99.5451},
        "South Sulawesi": {'lat': -5.1477, 'lon': 119.4327},
        "Yogyakarta": {'lat': -7.7956, 'lon': 110.3695},
        "Aceh": {'lat': 4.6951, 'lon': 96.7494},
        "Riau": {'lat': 0.2933, 'lon': 101.7068},
        "South Sumatra": {'lat': -2.9761, 'lon': 104.7754},
        "West Kalimantan": {'lat': 0.0000, 'lon': 109.3333},
        "East Kalimantan": {'lat': 0.4387, 'lon': 116.9740},
        "South Kalimantan": {'lat': -3.0926, 'lon': 115.2838},
        "North Sulawesi": {'lat': 0.6246, 'lon': 123.9750},
        "Papua": {'lat': -4.2699, 'lon': 138.0804},
        "All Indonesia": {'lat': -2.5489, 'lon': 118.0149},  # Center of Indonesia
        "Unspecified": {'lat': -2.5489, 'lon': 118.0149}  # Center of Indonesia
    }
    
    # Prepare data for the map
    map_data = []
    for _, row in region_counts.iterrows():
        region = row['region']
        count = row['count']
        
        # Skip if region not in our defined list
        if region not in indonesia_regions:
            continue
        
        # Get coordinates
        lat = indonesia_regions[region]['lat']
        lon = indonesia_regions[region]['lon']
        
        map_data.append({
            'region': region,
            'count': count,
            'lat': lat,
            'lon': lon
        })
    
    # Create DataFrame for plotting
    map_df = pd.DataFrame(map_data)
    
    if map_df.empty:
        # Return empty chart with message
        fig = go.Figure()
        fig.add_annotation(
            text="No mappable regional data available",
            x=0.5, y=0.5,
            xref="paper", yref="paper",
            showarrow=False,
            font=dict(size=16)
        )
        fig.update_layout(title=title)
        return fig
    
    # Create bubble map
    fig = px.scatter_geo(
        map_df,
        lat='lat',
        lon='lon',
        size='count',
        hover_name='region',
        hover_data=['count'],
        title=title,
        size_max=30,
        color='count',
        color_continuous_scale=px.colors.sequential.Reds
    )
    
    # Update to focus on Indonesia
    fig.update_geos(
        visible=False,
        showcountries=True,
        countrycolor="gray",
        showcoastlines=True,
        coastlinecolor="gray",
        projection_type="mercator",
        lonaxis_range=[95, 141],
        lataxis_range=[-11, 6]
    )
    
    fig.update_layout(
        height=500,
        geo=dict(
            scope='asia',
            showland=True,
            landcolor='rgb(243, 243, 243)',
            countrycolor='rgb(204, 204, 204)'
        )
    )
    
    return fig

def plot_word_cloud_data(word_data: List[Dict[str, Any]]) -> alt.Chart:
    """
    Create a word cloud visualization using Altair.
    
    Args:
        word_data: List of dictionaries with text and value keys
        
    Returns:
        Altair chart
    """
    if not word_data:
        # Return empty chart with message
        return alt.Chart().mark_text(
            text="No word data available",
            align="center",
            baseline="middle",
            fontSize=16
        ).properties(
            width=600,
            height=400
        )
    
    # Create DataFrame from word data
    df = pd.DataFrame(word_data)
    
    # Create word cloud chart with Altair
    chart = alt.Chart(df).mark_text(
        align='center',
        baseline='middle'
    ).encode(
        text=alt.Text('text:N'),
        size=alt.Size('value:Q', scale=alt.Scale(range=[10, 80])),
        color=alt.Color('value:Q', scale=alt.Scale(scheme='blues')),
        tooltip=['text:N', 'value:Q']
    ).transform_window(
        rank='rank()',
        sort=[alt.SortField('value', order='descending')]
    ).transform_filter(
        alt.datum.rank <= 100  # Show top 100 words
    ).properties(
        width=600,
        height=400
    )
    
    return chart

def plot_comparison_chart(data: Dict[str, List], x_field: str, y_field: str, 
                         category_field: str, title: str) -> go.Figure:
    """
    Create a comparison chart for multiple series.
    
    Args:
        data: Dictionary with category names as keys and lists of data points as values
        x_field: Name of x-axis field in data points
        y_field: Name of y-axis field in data points
        category_field: Name of category field in data points
        title: Chart title
        
    Returns:
        Plotly figure
    """
    if not data:
        # Return empty chart with message
        fig = go.Figure()
        fig.add_annotation(
            text="No comparison data available",
            x=0.5, y=0.5,
            xref="paper", yref="paper",
            showarrow=False,
            font=dict(size=16)
        )
        fig.update_layout(title=title)
        return fig
    
    # Flatten data for Plotly
    flat_data = []
    for category, items in data.items():
        for item in items:
            item_copy = item.copy()
            item_copy[category_field] = category
            flat_data.append(item_copy)
    
    # Create DataFrame
    df = pd.DataFrame(flat_data)
    
    if x_field not in df.columns or y_field not in df.columns or category_field not in df.columns:
        # Return empty chart with message
        fig = go.Figure()
        fig.add_annotation(
            text=f"Missing required fields in data",
            x=0.5, y=0.5,
            xref="paper", yref="paper",
            showarrow=False,
            font=dict(size=16)
        )
        fig.update_layout(title=title)
        return fig
    
    # Create line chart
    fig = px.line(
        df, 
        x=x_field, 
        y=y_field,
        color=category_field,
        title=title,
        markers=True
    )
    
    fig.update_layout(
        xaxis_title=x_field.capitalize(),
        yaxis_title=y_field.capitalize(),
        legend_title=category_field.capitalize()
    )
    
    return fig

def plot_stacked_bar(df: pd.DataFrame, x_col: str, y_col: str, stack_col: str, 
                    title: str = "Stacked Distribution") -> go.Figure:
    """
    Create a stacked bar chart.
    
    Args:
        df: DataFrame with data
        x_col: Column for x-axis categories
        y_col: Column for y-axis values
        stack_col: Column for stacking/color categories
        title: Chart title
        
    Returns:
        Plotly figure
    """
    if df.empty or x_col not in df.columns or y_col not in df.columns or stack_col not in df.columns:
        # Return empty chart with message
        fig = go.Figure()
        fig.add_annotation(
            text="No data available for stacked chart",
            x=0.5, y=0.5,
            xref="paper", yref="paper",
            showarrow=False,
            font=dict(size=16)
        )
        fig.update_layout(title=title)
        return fig
    
    # Create stacked bar chart
    fig = px.bar(
        df,
        x=x_col,
        y=y_col,
        color=stack_col,
        title=title,
        barmode='stack'
    )
    
    fig.update_layout(
        xaxis_title=x_col.capitalize(),
        yaxis_title=y_col.capitalize(),
        legend_title=stack_col.capitalize()
    )
    
    return fig

def plot_top_entities(data: List[Dict[str, Any]], value_col: str, name_col: str,
                     title: str = "Top Entities") -> go.Figure:
    """
    Create a horizontal bar chart showing top entities.
    
    Args:
        data: List of dictionaries with entity data
        value_col: Key for value field in dictionaries
        name_col: Key for name field in dictionaries
        title: Chart title
        
    Returns:
        Plotly figure
    """
    if not data:
        # Return empty chart with message
        fig = go.Figure()
        fig.add_annotation(
            text="No entity data available",
            x=0.5, y=0.5,
            xref="paper", yref="paper",
            showarrow=False,
            font=dict(size=16)
        )
        fig.update_layout(title=title)
        return fig
    
    # Create DataFrame from data
    df = pd.DataFrame(data)
    
    if value_col not in df.columns or name_col not in df.columns:
        # Return empty chart with message
        fig = go.Figure()
        fig.add_annotation(
            text=f"Missing required fields in data",
            x=0.5, y=0.5,
            xref="paper", yref="paper",
            showarrow=False,
            font=dict(size=16)
        )
        fig.update_layout(title=title)
        return fig
    
    # Sort by value
    df = df.sort_values(value_col, ascending=True)
    
    # Create horizontal bar chart
    fig = px.bar(
        df,
        y=name_col,
        x=value_col,
        title=title,
        orientation='h',
        color=value_col,
        color_continuous_scale=px.colors.sequential.Blues
    )
    
    fig.update_layout(
        xaxis_title=value_col.capitalize(),
        yaxis_title="",
        coloraxis_showscale=False
    )
    
    return fig

def plot_radar_chart(data: Dict[str, float], title: str = "Radar Analysis") -> go.Figure:
    """
    Create a radar chart (polar area chart).
    
    Args:
        data: Dictionary with categories as keys and values as values
        title: Chart title
        
    Returns:
        Plotly figure
    """
    if not data:
        # Return empty chart with message
        fig = go.Figure()
        fig.add_annotation(
            text="No radar data available",
            x=0.5, y=0.5,
            xref="paper", yref="paper",
            showarrow=False,
            font=dict(size=16)
        )
        fig.update_layout(title=title)
        return fig
    
    # Prepare data for radar chart
    categories = list(data.keys())
    values = list(data.values())
    
    # Create radar chart
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        name='Analysis'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, max(values) * 1.1]
            )
        ),
        title=title
    )
    
    return fig
