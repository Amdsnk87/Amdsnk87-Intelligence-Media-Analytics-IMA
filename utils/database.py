import os
import pandas as pd
from sqlalchemy import create_engine, Column, Integer, String, Text, Float, DateTime, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from datetime import datetime
from urllib.parse import urlparse
import json

# Get and validate PostgreSQL URL
db_url = st.secrets["DATABASE_URL"]
if not DATABASE_URL:
    raise ValueError("Missing DATABASE_URL or PGDATABASE_URL environment variable")
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

# Safe log of connection
parsed = urlparse(DATABASE_URL)
safe_url = f"{parsed.scheme}://{parsed.username}:***@{parsed.hostname}:{parsed.port}/{parsed.path.lstrip('/')}"
print("Connecting to DB:", safe_url)

# Create SQLAlchemy engine and session
try:
    engine = create_engine(DATABASE_URL, pool_pre_ping=True, pool_recycle=3600, echo=True)
    Session = sessionmaker(bind=engine)
    Base = declarative_base()
    print("Database connection established")
except Exception as e:
    print(f"Database connection error: {str(e)}")
    engine = None
    Session = None
    Base = declarative_base()

# Define models
class NewsArticle(Base):
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

class SocialMediaPost(Base):
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

class Keyword(Base):
    __tablename__ = "keywords"
    id = Column(Integer, primary_key=True)
    keyword = Column(String(100))
    count = Column(Integer)
    source_type = Column(String(20))
    created_at = Column(DateTime, default=datetime.now)

class TopicModel(Base):
    __tablename__ = "topic_models"
    id = Column(Integer, primary_key=True)
    model_date = Column(DateTime, default=datetime.now)
    model_type = Column(String(50))
    n_topics = Column(Integer)
    source_type = Column(String(20))
    topics = relationship("Topic", back_populates="model", cascade="all, delete-orphan")

class Topic(Base):
    __tablename__ = "topics"
    id = Column(Integer, primary_key=True)
    model_id = Column(Integer, ForeignKey('topic_models.id'))
    topic_id = Column(Integer)
    words = Column(Text)
    weights = Column(Text)
    model = relationship("TopicModel", back_populates="topics")

# Initialize DB
def init_db():
    if engine is None:
        print("Cannot initialize database: No connection available")
        return False
    try:
        Base.metadata.create_all(engine)
        print("Database initialized successfully. Tables created.")
        return True
    except Exception as e:
        print(f"Error initializing database: {str(e)}")
        return False

# Store and retrieve functions

def store_news_article(session, article_data):
    article = NewsArticle(**article_data)
    session.add(article)
    session.commit()
    return article

def get_news_articles(session, limit=100):
    return session.query(NewsArticle).order_by(NewsArticle.published_at.desc()).limit(limit).all()

def store_social_media_post(session, post_data):
    post = SocialMediaPost(**post_data)
    session.add(post)
    session.commit()
    return post

def get_social_media_posts(session, limit=100):
    return session.query(SocialMediaPost).order_by(SocialMediaPost.posted_at.desc()).limit(limit).all()

def store_keyword(session, keyword_data):
    keyword = Keyword(**keyword_data)
    session.add(keyword)
    session.commit()
    return keyword

def get_keywords(session, source_type=None):
    query = session.query(Keyword)
    if source_type:
        query = query.filter(Keyword.source_type == source_type)
    return query.order_by(Keyword.count.desc()).all()

def store_topic_model(session, model_data):
    model = TopicModel(**model_data)
    session.add(model)
    session.commit()
    return model

def store_topic(session, topic_data):
    topic = Topic(**topic_data)
    session.add(topic)
    session.commit()
    return topic

def get_topic_models(session, source_type=None):
    query = session.query(TopicModel)
    if source_type:
        query = query.filter(TopicModel.source_type == source_type)
    return query.order_by(TopicModel.model_date.desc()).all()

def get_topics_by_model(session, model_id):
    return session.query(Topic).filter(Topic.model_id == model_id).all()
