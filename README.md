# Intelligence Media Analytics (IMA)

Real-time monitoring and analysis platform for Indonesian media content across news outlets and social media platforms.

## Features

- **Real-time media monitoring**: Track news articles and social media posts from major Indonesian sources
- **Advanced analytics**: Perform sentiment analysis, trend detection, and topic modeling
- **Historical data access**: Analyze content from various time periods with flexible date range selection
- **Interactive visualizations**: Explore data through charts, word clouds, and interactive graphs
- **Search functionality**: Find specific content across multiple sources
- **Filter capabilities**: Narrow results by source, region, topic, and date

## Deployment Instructions

### Railway Deployment

1. Create a new project on [Railway](https://railway.app/)
2. Connect your GitHub repository or use Railway's GitHub integration
3. Railway will automatically detect the configuration and build the application
4. Add the following environment variables in the Railway dashboard:
   - `NEWS_API_KEY`: Your NewsAPI key
   - `TWITTER_API_KEY`: Your Twitter API key
   - `TWITTER_API_SECRET`: Your Twitter API secret
   - `TWITTER_ACCESS_TOKEN`: Your Twitter access token
   - `TWITTER_ACCESS_SECRET`: Your Twitter access token secret
   - Any PostgreSQL database credentials (if not using Railway's PostgreSQL plugin)

5. If you want to use Railway's PostgreSQL plugin:
   - Add the PostgreSQL plugin from the Railway dashboard
   - Railway will automatically set up the database and environment variables

6. Deploy the application - Railway will automatically build and deploy your application

### Local Development

1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Install NLTK data: `python download_nltk_data.py`
4. Set up environment variables for API keys
5. Run the application: `streamlit run app.py`

## API Keys Required

- [NewsAPI](https://newsapi.org/) for news article collection
- [Twitter Developer API](https://developer.twitter.com/en/docs/twitter-api) for social media data

## Technology Stack

- **Frontend**: Streamlit
- **Backend**: Python
- **Database**: PostgreSQL
- **NLP Processing**: NLTK, spaCy, TextBlob
- **Visualization**: Plotly, Altair
- **Data Scraping**: Trafilatura, BeautifulSoup
- **API Integration**: NewsAPI, Twitter API