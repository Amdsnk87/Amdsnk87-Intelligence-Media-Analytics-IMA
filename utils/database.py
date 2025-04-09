import os
import pandas as pd
from sqlalchemy import create_engine, Column, Integer, String, Text, Float, DateTime, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from datetime import datetime

# Database connection
DATABASE_URL = os.getenv("DATABASE_URL")

# Fix for Railway's PostgreSQL - Railway uses 'postgres://' but SQLAlchemy requires 'postgresql://'
if DATABASE_URL and DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://")

# Handle database connection with better error reporting
try:
    if DATABASE_URL:
        engine = create_engine(DATABASE_URL, pool_pre_ping=True, pool_recycle=3600)
        Session = sessionmaker(bind=engine)
        Base = declarative_base()
        print("Database connection established")
    else:
        raise ValueError("DATABASE_URL environment variable not set")
except Exception as e:
    print(f"Database connection error: {str(e)}")
    # Create a fallback engine that will be replaced once connection is available
    Base = declarative_base()
    engine = None
    Session = None

# Define models
class NewsArticle(Base):
    """Model for storing news articles"""
    __tablename__ = "news_articles"
    
    id = Column(Integer, primary_key=True)
    source = Column(String(100))
    title = Column(String(500))
    description = Column(Text, nullable=True)
    content = Column(Text, nullable=True)
    url = Column(String(500), nullable=True)
    published_at = Column(DateTime)
    author = Column(String(200), nullable=True)
    category = Column(String(100), nullable=True)
    region = Column(String(100), nullable=True)
    sentiment_polarity = Column(Float, nullable=True)
    sentiment_subjectivity = Column(Float, nullable=True)
    sentiment_category = Column(String(20), nullable=True)
    created_at = Column(DateTime, default=datetime.now)
    
    def __repr__(self):
        return f"<NewsArticle(id={self.id}, title='{self.title[:30]}...', source='{self.source}')>"


class SocialMediaPost(Base):
    """Model for storing social media posts"""
    __tablename__ = "social_media_posts"
    
    id = Column(Integer, primary_key=True)
    platform = Column(String(50))
    content = Column(Text)
    posted_at = Column(DateTime)
    username = Column(String(100), nullable=True)
    user_id = Column(String(100), nullable=True)
    user_followers = Column(Integer, nullable=True)
    likes = Column(Integer, nullable=True)
    retweets = Column(Integer, nullable=True)
    comments = Column(Integer, nullable=True)
    total_engagement = Column(Integer, nullable=True)
    category = Column(String(100), nullable=True)
    region = Column(String(100), nullable=True)
    sentiment_polarity = Column(Float, nullable=True)
    sentiment_subjectivity = Column(Float, nullable=True)
    sentiment_category = Column(String(20), nullable=True)
    created_at = Column(DateTime, default=datetime.now)
    
    def __repr__(self):
        return f"<SocialMediaPost(id={self.id}, platform='{self.platform}', username='{self.username}')>"


class Keyword(Base):
    """Model for storing keywords and their frequency"""
    __tablename__ = "keywords"
    
    id = Column(Integer, primary_key=True)
    keyword = Column(String(100))
    count = Column(Integer)
    source_type = Column(String(20))  # 'news' or 'social'
    created_at = Column(DateTime, default=datetime.now)
    
    def __repr__(self):
        return f"<Keyword(id={self.id}, keyword='{self.keyword}', count={self.count})>"


class TopicModel(Base):
    """Model for storing topic modeling results"""
    __tablename__ = "topic_models"
    
    id = Column(Integer, primary_key=True)
    model_date = Column(DateTime, default=datetime.now)
    model_type = Column(String(50))
    n_topics = Column(Integer)
    source_type = Column(String(20))  # 'news' or 'social'
    
    topics = relationship("Topic", back_populates="model", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<TopicModel(id={self.id}, date='{self.model_date}', n_topics={self.n_topics})>"


class Topic(Base):
    """Model for storing individual topics from topic modeling"""
    __tablename__ = "topics"
    
    id = Column(Integer, primary_key=True)
    model_id = Column(Integer, ForeignKey('topic_models.id'))
    topic_id = Column(Integer)
    words = Column(Text)  # Stored as JSON string
    weights = Column(Text)  # Stored as JSON string
    
    model = relationship("TopicModel", back_populates="topics")
    
    def __repr__(self):
        return f"<Topic(id={self.id}, topic_id={self.topic_id})>"


# Initialize the database
def init_db():
    """Initialize the database and create all tables if they don't exist."""
    if engine is None:
        print("Cannot initialize database: No connection available")
        return False
    
    try:
        # Create tables
        Base.metadata.create_all(engine)
        print("Database initialized successfully. Tables created.")
        return True
    except Exception as e:
        print(f"Error initializing database: {str(e)}")
        return False


# Functions to store and retrieve data
def store_news_articles(news_df):
    """
    Store news articles from DataFrame to database
    
    Args:
        news_df: DataFrame containing news articles
    """
    if news_df.empty:
        return
    
    session = Session()
    
    try:
        # Convert DataFrame to list of dictionaries
        news_records = news_df.to_dict('records')
        
        for record in news_records:
            # Check if article already exists
            existing = session.query(NewsArticle).filter_by(
                title=record.get('title', ''),
                source=record.get('source', ''),
                published_at=record.get('published_at')
            ).first()
            
            if not existing:
                # Create new article
                article = NewsArticle(
                    source=record.get('source', ''),
                    title=record.get('title', ''),
                    description=record.get('description', ''),
                    content=record.get('content', ''),
                    url=record.get('url', ''),
                    published_at=record.get('published_at'),
                    author=record.get('author', ''),
                    category=record.get('category', ''),
                    region=record.get('region', ''),
                    sentiment_polarity=record.get('sentiment_polarity'),
                    sentiment_subjectivity=record.get('sentiment_subjectivity'),
                    sentiment_category=record.get('sentiment_category')
                )
                session.add(article)
        
        session.commit()
    except Exception as e:
        session.rollback()
        print(f"Error storing news articles: {e}")
    finally:
        session.close()


def store_social_media_posts(social_df):
    """
    Store social media posts from DataFrame to database
    
    Args:
        social_df: DataFrame containing social media posts
    """
    if social_df.empty:
        return
    
    session = Session()
    
    try:
        # Convert DataFrame to list of dictionaries
        social_records = social_df.to_dict('records')
        
        for record in social_records:
            # Check if post already exists
            existing = session.query(SocialMediaPost).filter_by(
                platform=record.get('platform', ''),
                user_id=record.get('user_id', ''),
                posted_at=record.get('posted_at')
            ).first()
            
            if not existing:
                # Create new post
                post = SocialMediaPost(
                    platform=record.get('platform', ''),
                    content=record.get('content', ''),
                    posted_at=record.get('posted_at'),
                    username=record.get('username', ''),
                    user_id=record.get('user_id', ''),
                    user_followers=record.get('user_followers'),
                    likes=record.get('likes'),
                    retweets=record.get('retweets'),
                    comments=record.get('comments'),
                    total_engagement=record.get('total_engagement'),
                    category=record.get('category', ''),
                    region=record.get('region', ''),
                    sentiment_polarity=record.get('sentiment_polarity'),
                    sentiment_subjectivity=record.get('sentiment_subjectivity'),
                    sentiment_category=record.get('sentiment_category')
                )
                session.add(post)
        
        session.commit()
    except Exception as e:
        session.rollback()
        print(f"Error storing social media posts: {e}")
    finally:
        session.close()


def store_keywords(keywords, source_type='news'):
    """
    Store keywords and their frequencies
    
    Args:
        keywords: List of dictionaries with keyword and count
        source_type: 'news' or 'social'
    """
    if not keywords:
        return
    
    session = Session()
    
    try:
        for kw in keywords:
            # Check if keyword already exists
            existing = session.query(Keyword).filter_by(
                keyword=kw['keyword'],
                source_type=source_type
            ).first()
            
            if existing:
                # Update count
                existing.count = kw['count']
                existing.created_at = datetime.now()
            else:
                # Create new keyword
                keyword = Keyword(
                    keyword=kw['keyword'],
                    count=kw['count'],
                    source_type=source_type
                )
                session.add(keyword)
        
        session.commit()
    except Exception as e:
        session.rollback()
        print(f"Error storing keywords: {e}")
    finally:
        session.close()


def store_topic_model(topics, model_type='lda', n_topics=5, source_type='news'):
    """
    Store topic modeling results
    
    Args:
        topics: List of dictionaries with topic words and weights
        model_type: Type of model used ('lda', 'nmf', etc.)
        n_topics: Number of topics
        source_type: 'news' or 'social'
    """
    if not topics:
        return
    
    session = Session()
    
    try:
        # Create topic model
        model = TopicModel(
            model_type=model_type,
            n_topics=n_topics,
            source_type=source_type
        )
        session.add(model)
        session.flush()  # To get the model ID
        
        # Create topics
        for topic in topics:
            import json
            
            topic_obj = Topic(
                model_id=model.id,
                topic_id=topic['id'],
                words=json.dumps(topic['words']),
                weights=json.dumps(topic['weights'])
            )
            session.add(topic_obj)
        
        session.commit()
    except Exception as e:
        session.rollback()
        print(f"Error storing topic model: {e}")
    finally:
        session.close()


def get_news_articles(
    start_date=None, 
    end_date=None, 
    sources=None, 
    categories=None, 
    regions=None,
    limit=1000
):
    """
    Retrieve news articles from database with filtering
    
    Args:
        start_date: Start date for filtering
        end_date: End date for filtering
        sources: List of sources to filter by
        categories: List of categories to filter by
        regions: List of regions to filter by
        limit: Maximum number of records to return
        
    Returns:
        DataFrame with news articles
    """
    session = Session()
    
    try:
        query = session.query(NewsArticle)
        
        # Apply filters
        if start_date:
            query = query.filter(NewsArticle.published_at >= start_date)
        
        if end_date:
            query = query.filter(NewsArticle.published_at <= end_date)
        
        if sources and 'All Sources' not in sources:
            query = query.filter(NewsArticle.source.in_(sources))
        
        if categories and 'All Categories' not in categories:
            query = query.filter(NewsArticle.category.in_(categories))
        
        if regions and 'All Indonesia' not in regions:
            query = query.filter(NewsArticle.region.in_(regions))
        
        # Order by published date (newest first)
        query = query.order_by(NewsArticle.published_at.desc())
        
        # Limit results
        query = query.limit(limit)
        
        # Execute query and get results
        news_articles = query.all()
        
        # Convert to DataFrame
        if news_articles:
            data = []
            for article in news_articles:
                item = {
                    'id': article.id,
                    'source': article.source,
                    'title': article.title,
                    'description': article.description,
                    'content': article.content,
                    'url': article.url,
                    'published_at': article.published_at,
                    'author': article.author,
                    'category': article.category,
                    'region': article.region,
                    'sentiment_polarity': article.sentiment_polarity,
                    'sentiment_subjectivity': article.sentiment_subjectivity,
                    'sentiment_category': article.sentiment_category
                }
                data.append(item)
            
            return pd.DataFrame(data)
        else:
            return pd.DataFrame()
    
    except Exception as e:
        print(f"Error getting news articles: {e}")
        return pd.DataFrame()
    finally:
        session.close()


def get_social_media_posts(
    start_date=None, 
    end_date=None, 
    platforms=None, 
    categories=None, 
    regions=None,
    limit=1000
):
    """
    Retrieve social media posts from database with filtering
    
    Args:
        start_date: Start date for filtering
        end_date: End date for filtering
        platforms: List of platforms to filter by
        categories: List of categories to filter by
        regions: List of regions to filter by
        limit: Maximum number of records to return
        
    Returns:
        DataFrame with social media posts
    """
    session = Session()
    
    try:
        query = session.query(SocialMediaPost)
        
        # Apply filters
        if start_date:
            query = query.filter(SocialMediaPost.posted_at >= start_date)
        
        if end_date:
            query = query.filter(SocialMediaPost.posted_at <= end_date)
        
        if platforms and 'All Platforms' not in platforms:
            query = query.filter(SocialMediaPost.platform.in_(platforms))
        
        if categories and 'All Categories' not in categories:
            query = query.filter(SocialMediaPost.category.in_(categories))
        
        if regions and 'All Indonesia' not in regions:
            query = query.filter(SocialMediaPost.region.in_(regions))
        
        # Order by posted date (newest first)
        query = query.order_by(SocialMediaPost.posted_at.desc())
        
        # Limit results
        query = query.limit(limit)
        
        # Execute query and get results
        social_posts = query.all()
        
        # Convert to DataFrame
        if social_posts:
            data = []
            for post in social_posts:
                item = {
                    'id': post.id,
                    'platform': post.platform,
                    'content': post.content,
                    'posted_at': post.posted_at,
                    'username': post.username,
                    'user_id': post.user_id,
                    'user_followers': post.user_followers,
                    'likes': post.likes,
                    'retweets': post.retweets,
                    'comments': post.comments,
                    'total_engagement': post.total_engagement,
                    'category': post.category,
                    'region': post.region,
                    'sentiment_polarity': post.sentiment_polarity,
                    'sentiment_subjectivity': post.sentiment_subjectivity,
                    'sentiment_category': post.sentiment_category
                }
                data.append(item)
            
            return pd.DataFrame(data)
        else:
            return pd.DataFrame()
    
    except Exception as e:
        print(f"Error getting social media posts: {e}")
        return pd.DataFrame()
    finally:
        session.close()


def get_top_keywords(source_type='news', limit=20):
    """
    Get top keywords from database
    
    Args:
        source_type: 'news' or 'social'
        limit: Maximum number of keywords to return
        
    Returns:
        List of dictionaries with keyword and count
    """
    session = Session()
    
    try:
        keywords = session.query(Keyword).\
            filter_by(source_type=source_type).\
            order_by(Keyword.count.desc()).\
            limit(limit).all()
        
        result = []
        for kw in keywords:
            result.append({
                'keyword': kw.keyword,
                'count': kw.count
            })
        
        return result
    
    except Exception as e:
        print(f"Error getting top keywords: {e}")
        return []
    finally:
        session.close()


def get_latest_topic_model(source_type='news'):
    """
    Get the latest topic model results
    
    Args:
        source_type: 'news' or 'social'
        
    Returns:
        List of dictionaries with topic words and weights
    """
    session = Session()
    
    try:
        # Get the latest model
        model = session.query(TopicModel).\
            filter_by(source_type=source_type).\
            order_by(TopicModel.model_date.desc()).\
            first()
        
        if not model:
            return []
        
        # Get topics for this model
        topics = session.query(Topic).\
            filter_by(model_id=model.id).\
            order_by(Topic.topic_id).all()
        
        result = []
        for topic in topics:
            import json
            
            result.append({
                'id': topic.topic_id,
                'words': json.loads(topic.words),
                'weights': json.loads(topic.weights)
            })
        
        return result
    
    except Exception as e:
        print(f"Error getting latest topic model: {e}")
        return []
    finally:
        session.close()


# Initialize database on import
init_db()