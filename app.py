import os
import nltk
import streamlit as st
from datetime import datetime, timedelta
import pandas as pd
import sqlalchemy
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

# Access the database URL from secrets
db_url = st.secrets["DATABASE_URL"]

# Create a database connection
engine = sqlalchemy.create_engine(db_url)

# Set a writable directory for NLTK data on Streamlit Cloud
nltk_data_dir = "/tmp/nltk_data"
os.makedirs(nltk_data_dir, exist_ok=True)
os.environ["NLTK_DATA"] = nltk_data_dir
nltk.data.path.append(nltk_data_dir)

# Download essential resources
nltk.download('punkt', download_dir=nltk_data_dir)
nltk.download('stopwords', download_dir=nltk_data_dir)
nltk.download('wordnet', download_dir=nltk_data_dir)
nltk.download('punkt_tab', download_dir=nltk_data_dir)
nltk.download('averaged_perceptron_tagger', download_dir=nltk_data_dir)

# Import components
from components.dashboard import show_dashboard
from components.search import show_search
from components.trends import show_trends
from components.sentiment import show_sentiment
from components.topics import show_topics
from components.all_articles import show_all_articles
from components.predictive_trends import show_predictive_trends
from components.influencer_tracking import show_influencer_tracking

# Set page config
st.set_page_config(
    page_title="Intelligence Media Analytics (IMA)",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state for data caching
if 'last_update' not in st.session_state:
    st.session_state.last_update = datetime.now() - timedelta(hours=1)
    
if 'news_data' not in st.session_state:
    st.session_state.news_data = None

if 'social_data' not in st.session_state:
    st.session_state.social_data = None

if 'search_history' not in st.session_state:
    st.session_state.search_history = []

# App title
st.title("Intelligence Media Analytics (IMA)")
st.markdown("### Real-time monitoring and analysis of Indonesian media")

# Sidebar
st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/9/9f/Flag_of_Indonesia.svg", width=100)
st.sidebar.title("Navigation")

# Navigation
page = st.sidebar.radio(
    "Select a page",
    ["Dashboard", "All Articles", "Search & Analysis", "Trend Analysis", 
     "Sentiment Analysis", "Topic Classification", "Predictive Trends", "Influencer Tracking"]
)

# Date range filter
st.sidebar.header("Date Range")
today = datetime.now().date()

# Predefined date ranges to improve historical data access
date_range_options = [
    "Last 7 days",
    "Last 30 days",
    "Last 3 months",
    "Last 6 months",
    "Last year",
    "Last 2 years",
    "Last 3 years",
    "Last 5 years",
    "Custom range"
]
selected_range = st.sidebar.selectbox("Select time period", date_range_options)

# Set start and end dates based on selection
if selected_range == "Last 7 days":
    start_date = today - timedelta(days=7)
    end_date = today
elif selected_range == "Last 30 days":
    start_date = today - timedelta(days=30)
    end_date = today
elif selected_range == "Last 3 months":
    start_date = today - timedelta(days=90)
    end_date = today
elif selected_range == "Last 6 months":
    start_date = today - timedelta(days=180)
    end_date = today
elif selected_range == "Last year":
    start_date = today - timedelta(days=365)
    end_date = today
elif selected_range == "Last 2 years":
    start_date = today - timedelta(days=365*2)
    end_date = today
elif selected_range == "Last 3 years":
    start_date = today - timedelta(days=365*3)
    end_date = today
elif selected_range == "Last 5 years":
    start_date = today - timedelta(days=365*5)
    end_date = today
else:  # Custom range
    start_date = st.sidebar.date_input("Start date", today - timedelta(days=30))
    end_date = st.sidebar.date_input("End date", today)

# Show the selected date range
st.sidebar.info(f"Data from: {start_date} to {end_date}")

# Media source filter
st.sidebar.header("Media Sources")
news_sources = st.sidebar.multiselect(
    "Select news sources",
    ["Kompas", "Detik", "Tempo", "CNN Indonesia", "Tribun News", "Republika", "Antara News", 
     "Berita Satu", "Jakarta Post", "Suara", "Liputan6", "Merdeka", "Okezone", "Sindo News",
     "JPNN", "Media Indonesia", "Jakarta Globe", "Bisnis Indonesia", "Kumparan", "Tirto",
     "BBC Indonesia", "VOA Indonesia", "Vice Indonesia", "IDN Times", "Grid", "Kontan", 
     "Katadata", "CNBC Indonesia", "Indonesia Investments"],
    default=["Kompas", "Detik", "CNN Indonesia", "Republika", "Antara News", "Liputan6"]
)

social_sources = st.sidebar.multiselect(
    "Select social media",
    ["Twitter", "Facebook Pages"],
    default=["Twitter"]
)

# Region filter
regions = [
    "All Indonesia", "DKI Jakarta", "West Java", "Central Java", "East Java",
    "Bali", "North Sumatra", "South Sulawesi", "Yogyakarta", "Other Provinces"
]
selected_regions = st.sidebar.multiselect("Region", regions, default=["All Indonesia"])

# Topic filter
topics = [
    "Politics", "Economy", "Social", "Culture", "Security & Defense",
    "Health", "Education", "Environment", "Technology", "International Relations"
]
selected_topics = st.sidebar.multiselect("Topics", topics, default=["Politics", "Economy", "Social"])

# Apply filters button
filter_button = st.sidebar.button("Apply Filters")

# Last updated info
st.sidebar.markdown(f"**Last updated:** {st.session_state.last_update.strftime('%Y-%m-%d %H:%M:%S')}")

# Refresh data button
if st.sidebar.button("Refresh Data Now"):
    # Set flag to force data refresh in components
    st.session_state.refresh_clicked = True
    st.session_state.last_update = datetime.now()
    
    # Clear cached data to force reload
    if 'news_data' in st.session_state:
        st.session_state.news_data = None
    if 'social_data' in st.session_state:
        st.session_state.social_data = None
    
    st.rerun()

# Display appropriate page based on selection
if page == "Dashboard":
    show_dashboard(
        start_date,
        end_date,
        news_sources,
        social_sources,
        selected_regions,
        selected_topics
    )
elif page == "All Articles":
    show_all_articles(
        start_date,
        end_date,
        news_sources,
        social_sources,
        selected_regions,
        selected_topics
    )
elif page == "Search & Analysis":
    show_search(
        start_date,
        end_date,
        news_sources,
        social_sources,
        selected_regions,
        selected_topics
    )
elif page == "Trend Analysis":
    show_trends(
        start_date,
        end_date,
        news_sources,
        social_sources,
        selected_regions,
        selected_topics
    )
elif page == "Sentiment Analysis":
    show_sentiment(
        start_date,
        end_date,
        news_sources,
        social_sources,
        selected_regions,
        selected_topics
    )
elif page == "Topic Classification":
    show_topics(
        start_date,
        end_date,
        news_sources,
        social_sources,
        selected_regions,
        selected_topics
    )
elif page == "Predictive Trends":
    show_predictive_trends(
        start_date,
        end_date,
        news_sources,
        social_sources,
        selected_regions,
        selected_topics
    )
elif page == "Influencer Tracking":
    show_influencer_tracking(
        start_date,
        end_date,
        news_sources,
        social_sources,
        selected_regions,
        selected_topics
    )

# Footer
st.markdown("---")
st.markdown(
    """
    <div style="text-align: center;">
        <small>
            Intelligence Media Analytics (IMA) - Monitoring over 2,000 national and international online media outlets, 
            200+ local and national print newspapers, TV networks, and social media.
        </small>
    </div>
    """,
    unsafe_allow_html=True
)
