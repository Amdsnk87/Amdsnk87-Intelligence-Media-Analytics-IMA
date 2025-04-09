"""
Predictive trends analysis component.
"""
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go

from utils.data_loader import get_data
from utils.predictive_analytics import (
    aggregate_time_series,
    predict_trend_linear,
    predict_trend_arima,
    detect_emerging_keywords,
    predict_topic_trends,
    predict_seasonal_patterns,
    get_trending_predictions
)

def show_predictive_trends(
    start_date,
    end_date,
    news_sources,
    social_sources,
    regions,
    topics
):
    """
    Display predictive trend analysis for media content.
    
    Args:
        start_date: Start date for filtering data
        end_date: End date for filtering data
        news_sources: List of selected news sources
        social_sources: List of selected social media platforms
        regions: List of selected regions
        topics: List of selected topics
    """
    st.header("Predictive Trend Analysis")
    
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
    
    # Show error if no data
    if news_df.empty and social_df.empty:
        st.warning("No data available for predictive analysis. Please adjust your filters or fetch more data.")
        return
    
    # Sidebar for prediction settings
    with st.sidebar:
        st.header("Prediction Settings")
        prediction_days = st.slider("Days to predict", min_value=3, max_value=14, value=7)
        
        prediction_model = st.selectbox(
            "Prediction model", 
            options=["Linear Regression", "ARIMA (time series)"],
            index=0
        )
        
        st.caption("Note: Predictions are estimates based on historical patterns and may not reflect future events accurately.")
    
    # Overall volume prediction
    st.subheader("Content Volume Prediction")
    
    # Create 2 columns
    col1, col2 = st.columns(2)
    
    with col1:
        # News volume prediction
        if not news_df.empty and 'published_at' in news_df.columns:
            # Aggregate data
            news_ts = aggregate_time_series(news_df, 'published_at')
            
            # Make prediction based on selected model
            if prediction_model == "Linear Regression":
                prediction, rmse, r2 = predict_trend_linear(news_ts, days_to_predict=prediction_days)
                model_metrics = f"RMSE: {rmse:.2f}, R²: {r2:.2f}"
            else:
                prediction, rmse = predict_trend_arima(news_ts, days_to_predict=prediction_days)
                model_metrics = f"RMSE: {rmse:.2f}"
            
            # Create prediction plot
            fig = go.Figure()
            
            # Add historical data
            historical = prediction[prediction['type'] == 'historical']
            fig.add_trace(go.Scatter(
                x=historical['date'],
                y=historical['count'],
                mode='lines+markers',
                name='Historical',
                line=dict(color='blue')
            ))
            
            # Add prediction
            forecast = prediction[prediction['type'] == 'prediction']
            fig.add_trace(go.Scatter(
                x=forecast['date'],
                y=forecast['count'],
                mode='lines+markers',
                name='Prediction',
                line=dict(color='red', dash='dash')
            ))
            
            # Update layout
            fig.update_layout(
                title=f"News Volume Prediction (Next {prediction_days} days)",
                xaxis_title="Date",
                yaxis_title="Article Count",
                legend_title="Data Type",
                hovermode="x unified"
            )
            
            # Add confidence interval
            if prediction_model == "Linear Regression":
                # Calculate simple confidence interval for linear model
                margin = rmse * 1.96  # 95% confidence
                
                fig.add_trace(go.Scatter(
                    x=forecast['date'].tolist() + forecast['date'].tolist()[::-1],
                    y=(forecast['count'] + margin).tolist() + (forecast['count'] - margin).tolist()[::-1],
                    fill='toself',
                    fillcolor='rgba(255,0,0,0.2)',
                    line=dict(color='rgba(255,0,0,0)'),
                    hoverinfo="skip",
                    showlegend=False
                ))
            
            # Show plot
            st.plotly_chart(fig, use_container_width=True)
            
            # Show metrics
            st.caption(f"Model: {prediction_model}, {model_metrics}")
            
            # Prediction insights
            last_value = historical['count'].iloc[-1] if not historical.empty else 0
            pred_value = forecast['count'].iloc[-1] if not forecast.empty else 0
            
            if pred_value > last_value:
                st.info(f"📈 Predicted increase of {pred_value - last_value:.1f} articles ({(pred_value - last_value) / last_value * 100:.1f}%) in {prediction_days} days")
            elif pred_value < last_value:
                st.info(f"📉 Predicted decrease of {last_value - pred_value:.1f} articles ({(last_value - pred_value) / last_value * 100:.1f}%) in {prediction_days} days")
            else:
                st.info("⏸️ Volume predicted to remain stable")
        else:
            st.info("No news data available for volume prediction.")
        
    with col2:
        # Social media volume prediction
        if not social_df.empty and 'posted_at' in social_df.columns:
            # Aggregate data
            social_ts = aggregate_time_series(social_df, 'posted_at')
            
            # Make prediction based on selected model
            if prediction_model == "Linear Regression":
                prediction, rmse, r2 = predict_trend_linear(social_ts, days_to_predict=prediction_days)
                model_metrics = f"RMSE: {rmse:.2f}, R²: {r2:.2f}"
            else:
                prediction, rmse = predict_trend_arima(social_ts, days_to_predict=prediction_days)
                model_metrics = f"RMSE: {rmse:.2f}"
            
            # Create prediction plot
            fig = go.Figure()
            
            # Add historical data
            historical = prediction[prediction['type'] == 'historical']
            fig.add_trace(go.Scatter(
                x=historical['date'],
                y=historical['count'],
                mode='lines+markers',
                name='Historical',
                line=dict(color='blue')
            ))
            
            # Add prediction
            forecast = prediction[prediction['type'] == 'prediction']
            fig.add_trace(go.Scatter(
                x=forecast['date'],
                y=forecast['count'],
                mode='lines+markers',
                name='Prediction',
                line=dict(color='red', dash='dash')
            ))
            
            # Update layout
            fig.update_layout(
                title=f"Social Media Volume Prediction (Next {prediction_days} days)",
                xaxis_title="Date",
                yaxis_title="Post Count",
                legend_title="Data Type",
                hovermode="x unified"
            )
            
            # Add confidence interval
            if prediction_model == "Linear Regression":
                # Calculate simple confidence interval for linear model
                margin = rmse * 1.96  # 95% confidence
                
                fig.add_trace(go.Scatter(
                    x=forecast['date'].tolist() + forecast['date'].tolist()[::-1],
                    y=(forecast['count'] + margin).tolist() + (forecast['count'] - margin).tolist()[::-1],
                    fill='toself',
                    fillcolor='rgba(255,0,0,0.2)',
                    line=dict(color='rgba(255,0,0,0)'),
                    hoverinfo="skip",
                    showlegend=False
                ))
            
            # Show plot
            st.plotly_chart(fig, use_container_width=True)
            
            # Show metrics
            st.caption(f"Model: {prediction_model}, {model_metrics}")
            
            # Prediction insights
            last_value = historical['count'].iloc[-1] if not historical.empty else 0
            pred_value = forecast['count'].iloc[-1] if not forecast.empty else 0
            
            if pred_value > last_value:
                st.info(f"📈 Predicted increase of {pred_value - last_value:.1f} posts ({(pred_value - last_value) / max(1, last_value) * 100:.1f}%) in {prediction_days} days")
            elif pred_value < last_value:
                st.info(f"📉 Predicted decrease of {last_value - pred_value:.1f} posts ({(last_value - pred_value) / max(1, last_value) * 100:.1f}%) in {prediction_days} days")
            else:
                st.info("⏸️ Volume predicted to remain stable")
        else:
            st.info("No social media data available for volume prediction.")
    
    # Topic Trend Prediction
    st.subheader("Topic Trend Prediction")
    
    if not news_df.empty and 'published_at' in news_df.columns and 'category' in news_df.columns:
        # Predict topic trends
        topic_predictions, topic_rmse = predict_topic_trends(
            news_df, 'published_at', 'category', prediction_days
        )
        
        if not topic_predictions.empty:
            # Show topic prediction
            unique_topics = topic_predictions['category'].unique()
            
            # Allow user to select which topics to view
            selected_topics = st.multiselect(
                "Select topics to display",
                options=unique_topics,
                default=list(unique_topics)[:min(5, len(unique_topics))]
            )
            
            if selected_topics:
                # Filter data for selected topics
                filtered_predictions = topic_predictions[topic_predictions['category'].isin(selected_topics)]
                
                # Create figure
                fig = go.Figure()
                
                # Add each topic
                for topic in selected_topics:
                    topic_data = filtered_predictions[filtered_predictions['category'] == topic]
                    
                    # Historical data
                    historical = topic_data[topic_data['type'] == 'historical']
                    fig.add_trace(go.Scatter(
                        x=historical['date'],
                        y=historical[topic],
                        mode='lines+markers',
                        name=f"{topic} (Historical)",
                        line=dict(width=2)
                    ))
                    
                    # Prediction data
                    forecast = topic_data[topic_data['type'] == 'prediction']
                    fig.add_trace(go.Scatter(
                        x=forecast['date'],
                        y=forecast[topic],
                        mode='lines+markers',
                        name=f"{topic} (Prediction)",
                        line=dict(dash='dash', width=2)
                    ))
                
                # Update layout
                fig.update_layout(
                    title=f"Topic Trend Prediction (Next {prediction_days} days)",
                    xaxis_title="Date",
                    yaxis_title="Article Count",
                    legend_title="Topic",
                    hovermode="x unified"
                )
                
                # Show plot
                st.plotly_chart(fig, use_container_width=True)
                
                # Show prediction insights
                st.subheader("Topic Prediction Insights")
                
                # Calculate growth for each topic
                topic_insights = []
                
                for topic in selected_topics:
                    topic_data = filtered_predictions[filtered_predictions['category'] == topic]
                    
                    historical = topic_data[topic_data['type'] == 'historical']
                    forecast = topic_data[topic_data['type'] == 'prediction']
                    
                    if not historical.empty and not forecast.empty:
                        last_value = historical[topic].iloc[-1]
                        pred_value = forecast[topic].iloc[-1]
                        
                        growth = pred_value - last_value
                        growth_pct = growth / max(1, last_value) * 100
                        
                        topic_insights.append({
                            'topic': topic,
                            'current': last_value,
                            'predicted': pred_value,
                            'growth': growth,
                            'growth_pct': growth_pct,
                            'rmse': topic_rmse.get(topic, 0)
                        })
                
                # Sort by growth percentage
                topic_insights = sorted(topic_insights, key=lambda x: x['growth_pct'], reverse=True)
                
                # Display as table
                if topic_insights:
                    # Create DataFrame
                    insight_df = pd.DataFrame(topic_insights)
                    insight_df.columns = ['Topic', 'Current', 'Predicted', 'Growth', 'Growth %', 'RMSE']
                    
                    # Format columns
                    insight_df['Current'] = insight_df['Current'].round(1)
                    insight_df['Predicted'] = insight_df['Predicted'].round(1)
                    insight_df['Growth'] = insight_df['Growth'].round(1)
                    insight_df['Growth %'] = insight_df['Growth %'].round(1)
                    insight_df['RMSE'] = insight_df['RMSE'].round(2)
                    
                    # Show table
                    st.dataframe(insight_df, use_container_width=True)
                    
                    # Identify fastest growing and declining topics
                    growing = [t for t in topic_insights if t['growth_pct'] > 10]
                    declining = [t for t in topic_insights if t['growth_pct'] < -10]
                    
                    if growing:
                        top_growing = growing[0]
                        st.success(f"🚀 Fastest growing topic: **{top_growing['topic']}** (+{top_growing['growth_pct']:.1f}% predicted)")
                    
                    if declining:
                        top_declining = declining[0]
                        st.warning(f"📉 Fastest declining topic: **{top_declining['topic']}** ({top_declining['growth_pct']:.1f}% predicted)")
            else:
                st.info("Please select at least one topic to display predictions.")
        else:
            st.info("Not enough data for topic trend prediction.")
    else:
        st.info("Topic categorization not available. Make sure articles have categories assigned.")
    
    # Emerging Keywords Analysis
    st.subheader("Emerging Keywords Detection")
    
    if not news_df.empty and 'content' in news_df.columns and 'published_at' in news_df.columns:
        # Detect emerging keywords
        emerging_keywords = detect_emerging_keywords(
            news_df, 'content', 'published_at', min_count=2, growth_threshold=1.2
        )
        
        if emerging_keywords:
            # Create two columns
            col1, col2 = st.columns(2)
            
            with col1:
                # Display as table
                emerging_df = pd.DataFrame(emerging_keywords)
                
                # Rename columns
                column_mapping = {
                    'keyword': 'Keyword',
                    'recent_count': 'Recent Count',
                    'previous_count': 'Previous Count',
                    'growth': 'Growth',
                    'growth_factor': 'Growth Factor'
                }
                emerging_df = emerging_df.rename(columns=column_mapping)
                
                # Show only certain columns
                display_cols = ['Keyword', 'Recent Count', 'Previous Count', 'Growth']
                st.dataframe(emerging_df[display_cols], use_container_width=True)
                
                # Explain the table
                st.caption("This table shows keywords that are growing in popularity. 'New' means the keyword wasn't present in the previous period.")
            
            with col2:
                # Create bar chart of top emerging keywords
                top_emerging = emerging_df.head(10)
                
                # Create figure
                fig = px.bar(
                    top_emerging,
                    x='Keyword',
                    y='Recent Count',
                    text='Growth',
                    title="Top Emerging Keywords",
                    color='Growth Factor',
                    color_continuous_scale='Viridis'
                )
                
                # Show plot
                st.plotly_chart(fig, use_container_width=True)
            
            # Insights about emerging keywords
            st.markdown("### Emerging Topics Insights")
            
            # Group keywords into potential topics
            if len(emerging_keywords) >= 3:
                st.markdown("**Potential Emerging Stories/Topics:**")
                
                # Simple approach - just display top 3 emerging keywords together
                top3 = emerging_keywords[:3]
                st.markdown(f"- {', '.join([k['keyword'] for k in top3])}")
                
                # If we have more, show another group
                if len(emerging_keywords) >= 6:
                    next3 = emerging_keywords[3:6]
                    st.markdown(f"- {', '.join([k['keyword'] for k in next3])}")
        else:
            st.info("No emerging keywords detected in the current data. Try widening your date range.")
    else:
        st.info("Content data not available for keyword trend analysis.")
    
    # Seasonal Pattern Analysis
    st.subheader("Seasonal Pattern Analysis")
    
    if not news_df.empty and 'published_at' in news_df.columns:
        # Predict seasonal patterns
        try:
            seasonal_patterns = predict_seasonal_patterns(news_df, 'published_at')
            
            if 'error' not in seasonal_patterns:
                # Create two columns
                col1, col2 = st.columns(2)
                
                with col1:
                    # Display seasonal patterns
                    st.markdown("**Weekly Content Patterns:**")
                    
                    if 'highest_activity_day' in seasonal_patterns:
                        st.markdown(f"- Highest activity day: **{seasonal_patterns['highest_activity_day']}**")
                    
                    if 'lowest_activity_day' in seasonal_patterns:
                        st.markdown(f"- Lowest activity day: **{seasonal_patterns['lowest_activity_day']}**")
                    
                    if 'has_seasonal_pattern' in seasonal_patterns:
                        if seasonal_patterns['has_seasonal_pattern']:
                            st.markdown("- **Strong seasonal pattern detected** in content publishing")
                        else:
                            st.markdown("- No strong seasonal pattern detected in content publishing")
                
                with col2:
                    # Create bar chart of weekday patterns
                    if 'weekday_patterns' in seasonal_patterns:
                        weekday_patterns = seasonal_patterns['weekday_patterns']
                        
                        # Convert to DataFrame
                        weekday_df = pd.DataFrame({
                            'Day': list(weekday_patterns.keys()),
                            'Activity': list(weekday_patterns.values())
                        })
                        
                        # Order by days of week
                        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
                        weekday_df['Day'] = pd.Categorical(weekday_df['Day'], categories=day_order, ordered=True)
                        weekday_df = weekday_df.sort_values('Day')
                        
                        # Create figure
                        fig = px.bar(
                            weekday_df,
                            x='Day',
                            y='Activity',
                            title="Weekly Content Patterns",
                            color='Activity',
                            color_continuous_scale='Viridis'
                        )
                        
                        # Show plot
                        st.plotly_chart(fig, use_container_width=True)
            else:
                st.info(f"Seasonal analysis error: {seasonal_patterns['error']}")
        except Exception as e:
            st.info(f"Could not analyze seasonal patterns: {e}")
    else:
        st.info("Content timestamp data not available for seasonal analysis.")
    
    # Overall prediction insights
    st.subheader("Predictive Insights Summary")
    
    # Get comprehensive prediction data
    predictions = get_trending_predictions(news_df, social_df, prediction_days)
    
    if 'error' not in predictions:
        # Display summary of predictions
        st.markdown("### Key Insights")
        
        insights = []
        
        # Volume predictions
        if 'news_volume_prediction' in predictions:
            news_pred = predictions['news_volume_prediction']
            news_data = pd.DataFrame(news_pred['data'])
            
            if not news_data.empty:
                hist = news_data[news_data['type'] == 'historical']
                pred = news_data[news_data['type'] == 'prediction']
                
                if not hist.empty and not pred.empty:
                    current = hist['count'].iloc[-1]
                    future = pred['count'].iloc[-1]
                    change = (future - current) / max(1, current) * 100
                    
                    if abs(change) > 10:
                        direction = "increase" if change > 0 else "decrease"
                        insights.append(f"News volume expected to {direction} by {abs(change):.1f}% in the next {prediction_days} days")
        
        # Emerging keywords
        if 'emerging_keywords' in predictions and predictions['emerging_keywords']:
            top_keyword = predictions['emerging_keywords'][0]['keyword']
            insights.append(f"'{top_keyword}' is the fastest growing keyword, watch this trend closely")
        
        # Topic predictions
        if 'topic_predictions' in predictions and 'data' in predictions['topic_predictions']:
            topic_data = pd.DataFrame(predictions['topic_predictions']['data'])
            
            if not topic_data.empty:
                # Get unique topics
                topics = topic_data['category'].unique()
                
                # Find fastest growing topic
                fastest_growth = None
                fastest_topic = None
                
                for topic in topics:
                    topic_rows = topic_data[topic_data['category'] == topic]
                    
                    hist = topic_rows[topic_rows['type'] == 'historical']
                    pred = topic_rows[topic_rows['type'] == 'prediction']
                    
                    if not hist.empty and not pred.empty:
                        current = hist[topic].iloc[-1]
                        future = pred[topic].iloc[-1]
                        
                        if current > 0:  # Avoid division by zero
                            growth = (future - current) / current
                            
                            if fastest_growth is None or growth > fastest_growth:
                                fastest_growth = growth
                                fastest_topic = topic
                
                if fastest_topic and fastest_growth and fastest_growth > 0.1:
                    insights.append(f"'{fastest_topic}' topic is trending up, with {fastest_growth*100:.1f}% predicted growth")
        
        # Seasonal patterns
        if 'seasonal_patterns' in predictions and 'highest_activity_day' in predictions['seasonal_patterns']:
            insights.append(f"{predictions['seasonal_patterns']['highest_activity_day']} shows the highest content activity")
        
        # Display insights
        if insights:
            for i, insight in enumerate(insights):
                st.markdown(f"{i+1}. {insight}")
        else:
            st.info("No significant predictive insights detected with the current data.")
    else:
        st.info(f"Prediction error: {predictions['error']}")
    
    # Prediction methodology explanation
    with st.expander("About Predictive Analytics"):
        st.markdown("""
        ### How Predictions Work
        
        The predictions shown on this page use a combination of machine learning and statistical techniques:
        
        - **Linear Regression** predicts future values based on the trend of historical data
        - **ARIMA (AutoRegressive Integrated Moving Average)** captures temporal patterns in the data
        - **Emerging Keywords Detection** identifies terms that are growing in frequency
        - **Seasonal Decomposition** identifies recurring patterns in publishing activity
        
        ### Limitations
        
        - Predictions are based on historical patterns and may not account for unexpected events
        - Limited data history reduces prediction accuracy
        - Social and political events may cause sudden changes that models cannot anticipate
        - The confidence intervals show the range of likely outcomes
        
        Always use predictions as guidance, not certainty. Regular refreshing of data improves prediction quality.
        """)