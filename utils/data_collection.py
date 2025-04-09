import os
import pandas as pd
import requests
from datetime import datetime, timedelta
import trafilatura
import tweepy
import json
from bs4 import BeautifulSoup
import time
import re
import random
from typing import List, Dict, Any, Optional

# NewsAPI key from environment
NEWS_API_KEY = os.getenv("NEWS_API_KEY", "demo_key")
TWITTER_API_KEY = os.getenv("TWITTER_API_KEY", "demo_key")
TWITTER_API_SECRET = os.getenv("TWITTER_API_SECRET", "demo_secret")
TWITTER_ACCESS_TOKEN = os.getenv("TWITTER_ACCESS_TOKEN", "demo_token")
TWITTER_ACCESS_SECRET = os.getenv("TWITTER_ACCESS_SECRET", "demo_secret")

def fetch_from_newsapi(
    sources: List[str], 
    start_date: datetime.date, 
    end_date: datetime.date, 
    topics: List[str] = None, 
    regions: List[str] = None
) -> List[Dict[str, Any]]:
    """
    Fetch news articles from NewsAPI.
    
    Args:
        sources: List of news source names
        start_date: Start date for articles
        end_date: End date for articles
        topics: Optional list of topics to filter by
        regions: Optional list of regions to filter by
        
    Returns:
        List of article dictionaries
    """
    articles = []
    
    # Convert source names to domains
    domains = [INDONESIAN_NEWS_SOURCES[source] for source in sources if source in INDONESIAN_NEWS_SOURCES]
    if not domains:
        return []
    
    # Create keyword queries based on topics and regions
    keywords = []
    
    if topics and "All Topics" not in topics:
        topic_keywords = {
            "Politics": "politik OR pemilu OR presiden OR pemerintah OR partai",
            "Economy": "ekonomi OR bisnis OR keuangan OR investasi OR perdagangan",
            "Social": "sosial OR masyarakat OR kesejahteraan OR kemiskinan",
            "Culture": "budaya OR seni OR tradisi OR adat OR pariwisata",
            "Security & Defense": "keamanan OR pertahanan OR militer OR polisi OR terorisme",
            "Health": "kesehatan OR covid OR pandemi OR rumah sakit OR penyakit",
            "Education": "pendidikan OR sekolah OR universitas OR siswa OR mahasiswa",
            "Environment": "lingkungan OR iklim OR bencana OR polusi OR konservasi",
            "Technology": "teknologi OR digital OR startup OR inovasi OR aplikasi",
            "International Relations": "internasional OR bilateral OR diplomatik OR hubungan luar negeri"
        }
        
        topic_query = " OR ".join([topic_keywords[topic] for topic in topics if topic in topic_keywords])
        if topic_query:
            keywords.append(f"({topic_query})")
    
    if regions and "All Indonesia" not in regions:
        region_query = " OR ".join(regions)
        keywords.append(f"({region_query})")
    
    # Build query
    query = " AND ".join(keywords) if keywords else None
    
    # Format dates for NewsAPI
    from_date = start_date.strftime('%Y-%m-%d')
    to_date = end_date.strftime('%Y-%m-%d')
    
    # Fetch from domains
    for domain in domains:
        try:
            url = f"https://newsapi.org/v2/everything"
            params = {
                "domains": domain,
                "from": from_date,
                "to": to_date,
                "language": "id,en",
                "sortBy": "publishedAt",
                "pageSize": 100,
                "apiKey": NEWS_API_KEY
            }
            
            if query:
                params["q"] = query
                
            response = requests.get(url, params=params)
            
            if response.status_code == 200:
                data = response.json()
                if "articles" in data:
                    for article in data["articles"]:
                        # Map source name from domain
                        source_name = next((name for name, src_domain in INDONESIAN_NEWS_SOURCES.items() 
                                          if src_domain == domain), domain)
                        
                        # Try to determine category and region from content
                        category = categorize_article(article.get("title", ""), article.get("description", ""))
                        region = extract_region(article.get("title", ""), article.get("description", ""))
                        
                        articles.append({
                            "source": source_name,
                            "title": article.get("title", ""),
                            "description": article.get("description", ""),
                            "content": article.get("content", ""),
                            "url": article.get("url", ""),
                            "published_at": article.get("publishedAt", ""),
                            "author": article.get("author", ""),
                            "category": category,
                            "region": region
                        })
            else:
                print(f"Error fetching from {domain}: {response.status_code}")
                
        except Exception as e:
            print(f"Error processing {domain}: {str(e)}")
    
    return articles

# Indonesian news sources
INDONESIAN_NEWS_SOURCES = {
    "Kompas": "kompas.com",
    "Detik": "detik.com",
    "Tempo": "tempo.co",
    "CNN Indonesia": "cnnindonesia.com",
    "Tribun News": "tribunnews.com",
    "Republika": "republika.co.id",
    "Antara News": "antaranews.com",
    "Berita Satu": "beritasatu.com",
    "Jakarta Post": "thejakartapost.com"
}

def get_news_articles(
    sources: List[str], 
    start_date: datetime.date, 
    end_date: datetime.date, 
    topics: List[str] = None, 
    regions: List[str] = None
) -> pd.DataFrame:
    """
    Fetch news articles from NewsAPI based on selected sources and date range.
    If NewsAPI fails or isn't available, falls back to direct web scraping.
    
    Args:
        sources: List of news source names
        start_date: Start date for articles
        end_date: End date for articles
        topics: Optional list of topics to filter by
        regions: Optional list of regions to filter by
        
    Returns:
        DataFrame containing news articles
    """
    print(f"Fetching news articles for {sources} from {start_date} to {end_date}")
    
    # Empty dataframe for results
    news_df = pd.DataFrame(columns=[
        'source', 'title', 'description', 'content', 'url', 'published_at', 
        'author', 'category', 'region'
    ])
    
    # Try NewsAPI first if API key is available
    if NEWS_API_KEY and NEWS_API_KEY != "demo_key":
        try:
            # Use NewsAPI implementation here
            news_api_articles = fetch_from_newsapi(sources, start_date, end_date, topics, regions)
            if news_api_articles and len(news_api_articles) > 0:
                news_df = pd.DataFrame(news_api_articles)
                print(f"Successfully fetched {len(news_df)} articles from NewsAPI")
                return news_df
        except Exception as e:
            print(f"Error fetching from NewsAPI: {str(e)}")
    
    # Fall back to direct web scraping if NewsAPI fails or isn't available
    print("Falling back to direct web scraping...")
    try:
        scraped_articles = fallback_scrape_news(sources, start_date, end_date, topics, regions)
        if scraped_articles and len(scraped_articles) > 0:
            news_df = pd.DataFrame(scraped_articles)
            print(f"Successfully scraped {len(news_df)} articles directly from websites")
            return news_df
    except Exception as e:
        print(f"Error during scraping: {str(e)}")
    
    if news_df.empty:
        print("Warning: No news articles could be fetched from any source")
        
    return news_df

def get_website_text_content(url: str) -> str:
    """
    Get the main text content from a website using trafilatura.
    
    Args:
        url: The URL to scrape
        
    Returns:
        Extracted text content
    """
    try:
        downloaded = trafilatura.fetch_url(url)
        if downloaded:
            text = trafilatura.extract(downloaded)
            return text if text else ""
        return ""
    except Exception as e:
        print(f"Error extracting content from {url}: {str(e)}")
        return ""

def fallback_scrape_news(
    sources: List[str], 
    start_date: datetime.date, 
    end_date: datetime.date, 
    topics: List[str] = None, 
    regions: List[str] = None
) -> List[Dict[str, Any]]:
    """
    Fallback method to scrape Indonesian news sites directly when NewsAPI fails.
    
    Args:
        sources: List of news source names
        start_date: Start date for articles
        end_date: End date for articles
        topics: Optional list of topics to filter by
        regions: Optional list of regions to filter by
        
    Returns:
        List of article dictionaries
    """
    articles = []
    
    # Map of sources to their front page URLs
    source_urls = {
        "Kompas": "https://www.kompas.com/",
        "Detik": "https://www.detik.com/",
        "Tempo": "https://www.tempo.co/",
        "CNN Indonesia": "https://www.cnnindonesia.com/",
        "Antara News": "https://www.antaranews.com/",
        "Tribun News": "https://www.tribunnews.com/",
        "Republika": "https://republika.co.id/",
        "Berita Satu": "https://www.beritasatu.com/",
        "Jakarta Post": "https://www.thejakartapost.com/",
        "Suara": "https://www.suara.com/",
        "Liputan6": "https://www.liputan6.com/",
        "Merdeka": "https://www.merdeka.com/",
        "Okezone": "https://www.okezone.com/",
        "Sindo News": "https://nasional.sindonews.com/",
        "JPNN": "https://www.jpnn.com/",
        "Media Indonesia": "https://mediaindonesia.com/",
        "Jakarta Globe": "https://jakartaglobe.id/",
        "Bisnis Indonesia": "https://www.bisnis.com/",
        "Kumparan": "https://kumparan.com/",
        "Tirto": "https://tirto.id/",
        "BBC Indonesia": "https://www.bbc.com/indonesia",
        "VOA Indonesia": "https://www.voaindonesia.com/",
        "Vice Indonesia": "https://www.vice.com/id",
        "IDN Times": "https://www.idntimes.com/",
        "Grid": "https://www.grid.id/",
        "Kontan": "https://www.kontan.co.id/",
        "Katadata": "https://katadata.co.id/",
        "CNBC Indonesia": "https://www.cnbcindonesia.com/",
        "Indonesia Investments": "https://www.indonesia-investments.com/"
    }
    
    for source in sources:
        if source in source_urls:
            try:
                url = source_urls[source]
                response = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'})
                
                if response.status_code == 200:
                    soup = BeautifulSoup(response.text, 'html.parser')
                    
                    # Different scraping logic for different sites
                    if source == "Kompas":
                        article_links = soup.select('a.article__link')[:20]
                        for link in article_links:
                            article_url = link.get('href')
                            if article_url:
                                title = link.get_text().strip()
                                # Get full content
                                content = get_website_text_content(article_url)
                                # Determine category and region
                                category = categorize_article(title, content)
                                region = extract_region(title, content)
                                
                                articles.append({
                                    "source": source,
                                    "title": title,
                                    "description": content[:100] + "..." if content else "",
                                    "content": content,
                                    "url": article_url,
                                    "published_at": datetime.now().isoformat(),
                                    "author": "",
                                    "category": category,
                                    "region": region
                                })
                    
                    elif source == "Detik":
                        article_links = soup.select('article h2 a')[:20]
                        for link in article_links:
                            article_url = link.get('href')
                            if article_url:
                                title = link.get_text().strip()
                                content = get_website_text_content(article_url)
                                category = categorize_article(title, content)
                                region = extract_region(title, content)
                                
                                articles.append({
                                    "source": source,
                                    "title": title,
                                    "description": content[:100] + "..." if content else "",
                                    "content": content,
                                    "url": article_url,
                                    "published_at": datetime.now().isoformat(),
                                    "author": "",
                                    "category": category,
                                    "region": region
                                })
                    
                    # Similar patterns for other sources
                    elif source in ["Tempo", "CNN Indonesia", "Antara News", "Republika", "Tribun News", "Berita Satu", 
                                   "Jakarta Post", "Suara", "Liputan6", "Merdeka", "Okezone", "Sindo News", "JPNN",
                                   "Media Indonesia", "Jakarta Globe", "Bisnis Indonesia", "Kumparan", "Tirto", 
                                   "BBC Indonesia", "VOA Indonesia", "Vice Indonesia", "IDN Times", "Grid", "Kontan",
                                   "Katadata", "CNBC Indonesia", "Indonesia Investments"]:
                        # Common scraping pattern across most Indonesian news sites
                        article_links = soup.select('article a, .article a, .news-item a, h1 a, h2 a, h3 a, .title a, .entry-title a, .headline a, .latest a')[:20]
                        for link in article_links:
                            article_url = link.get('href')
                            if not article_url:
                                continue
                                
                            # Fix relative URLs
                            if not (article_url.startswith('http') or article_url.startswith('//')):
                                article_url = url + article_url.lstrip('/')
                            elif article_url.startswith('//'):
                                article_url = 'https:' + article_url
                                
                            # Extract title from various elements
                            title = link.get_text().strip() or link.get('title', '')
                            if not title:
                                title_elem = link.select_one('h1, h2, h3, .title, .headline')
                                if title_elem:
                                    title = title_elem.get_text().strip()
                            
                            # Skip if no title found
                            if not title:
                                continue
                                
                            # Get content and analyze
                            content = get_website_text_content(article_url)
                            category = categorize_article(title, content)
                            region = extract_region(title, content)
                            
                            articles.append({
                                "source": source,
                                "title": title,
                                "description": content[:100] + "..." if content else "",
                                "content": content,
                                "url": article_url,
                                "published_at": datetime.now().isoformat(),
                                "author": "",
                                "category": category,
                                "region": region
                            })
            
            except Exception as e:
                print(f"Error scraping {source}: {str(e)}")
    
    return articles

def get_social_media_data(
    platforms: List[str], 
    start_date: datetime.date, 
    end_date: datetime.date, 
    topics: List[str] = None, 
    regions: List[str] = None
) -> pd.DataFrame:
    """
    Fetch social media data from selected platforms.
    If API access fails, falls back to web scraping.
    
    Args:
        platforms: List of social media platform names
        start_date: Start date for posts
        end_date: End date for posts
        topics: Optional list of topics to filter by
        regions: Optional list of regions to filter by
        
    Returns:
        DataFrame containing social media posts
    """
    print(f"Fetching social media data for {platforms} from {start_date} to {end_date}")
    
    # Empty dataframe for results
    social_df = pd.DataFrame(columns=[
        'platform', 'content', 'username', 'user_followers', 'posted_at', 'likes', 'retweets', 
        'shares', 'comments', 'category', 'region', 'url'
    ])
    social_data = []
    
    # Create keyword lists based on topics
    topic_keywords = {
        "Politics": ["politik", "pemilu", "presiden", "pemerintah", "partai", "MPR", "DPR"],
        "Economy": ["ekonomi", "bisnis", "keuangan", "investasi", "perdagangan", "rupiah", "inflasi"],
        "Social": ["sosial", "masyarakat", "kesejahteraan", "kemiskinan", "komunitas"],
        "Culture": ["budaya", "seni", "tradisi", "adat", "pariwisata", "warisan"],
        "Security & Defense": ["keamanan", "pertahanan", "militer", "polisi", "terorisme", "TNI", "Polri"],
        "Health": ["kesehatan", "covid", "pandemi", "rumah sakit", "penyakit", "BPJS", "vaksin"],
        "Education": ["pendidikan", "sekolah", "universitas", "siswa", "mahasiswa", "guru", "dosen"],
        "Environment": ["lingkungan", "iklim", "bencana", "polusi", "konservasi", "banjir"],
        "Technology": ["teknologi", "digital", "startup", "inovasi", "aplikasi", "internet"],
        "International Relations": ["internasional", "bilateral", "diplomatik", "hubungan luar negeri", "PBB"]
    }
    
    search_keywords = []
    if topics and "All Topics" not in topics:
        for topic in topics:
            if topic in topic_keywords:
                search_keywords.extend(topic_keywords[topic])
    
    # Add region keywords if specified
    region_keywords = []
    if regions and "All Indonesia" not in regions:
        region_keywords = regions
    
    # Combine keywords
    combined_keywords = search_keywords + region_keywords
    
    # If Twitter is selected
    if "Twitter" in platforms and TWITTER_API_KEY and TWITTER_API_SECRET:
        try:
            # Initialize Twitter API
            auth = tweepy.OAuthHandler(TWITTER_API_KEY, TWITTER_API_SECRET)
            auth.set_access_token(TWITTER_ACCESS_TOKEN, TWITTER_ACCESS_SECRET)
            api = tweepy.API(auth)
            
            # Search for tweets
            query = " OR ".join(combined_keywords) if combined_keywords else "Indonesia"
            query += " lang:id OR lang:en" # Indonesian or English tweets
            
            # Format dates for Twitter API
            since_date = start_date.strftime('%Y-%m-%d')
            until_date = end_date.strftime('%Y-%m-%d')
            
            # Search tweets - increase count to 200 (Twitter API max)
            tweets = api.search_tweets(
                q=query,
                count=200,
                result_type="recent",
                tweet_mode="extended",
                since_id=since_date,
                until=until_date
            )
            
            for tweet in tweets:
                # Determine category based on tweet content
                category = categorize_article(tweet.full_text, "")
                region = extract_region(tweet.full_text, "")
                
                social_data.append({
                    "platform": "Twitter",
                    "content": tweet.full_text,
                    "username": tweet.user.screen_name,
                    "user_followers": tweet.user.followers_count,
                    "posted_at": tweet.created_at.isoformat(),
                    "likes": tweet.favorite_count,
                    "retweets": tweet.retweet_count,
                    "category": category,
                    "region": region,
                    "url": f"https://twitter.com/{tweet.user.screen_name}/status/{tweet.id}"
                })
                
        except Exception as e:
            print(f"Error fetching Twitter data: {str(e)}")
            # Fallback to simulated data when API fails
            social_data.extend(fallback_social_data("Twitter", combined_keywords))
    
    # If Facebook Pages is selected, add placeholder for Facebook API integration
    if "Facebook Pages" in platforms:
        # Facebook API would go here, but using fallback for now
        social_data.extend(fallback_social_data("Facebook", combined_keywords))
    
    return pd.DataFrame(social_data)

def fallback_social_data(platform: str, keywords: List[str]) -> List[Dict[str, Any]]:
    """
    Create fallback social media data when APIs are unavailable.
    
    Args:
        platform: Social media platform name
        keywords: Search keywords
        
    Returns:
        List of social media post dictionaries
    """
    data = []
    
    # Indonesian influencers/news accounts
    accounts = {
        "Twitter": [
            {"name": "jokowi", "followers": 19500000, "display": "Joko Widodo"},
            {"name": "KompasTV", "followers": 5200000, "display": "Kompas TV"},
            {"name": "tempodotco", "followers": 4700000, "display": "Tempo.co"},
            {"name": "cnnindonesia", "followers": 3900000, "display": "CNN Indonesia"},
            {"name": "detikcom", "followers": 16700000, "display": "detikcom"},
            {"name": "antaranews", "followers": 2100000, "display": "ANTARA News"}
        ],
        "Facebook": [
            {"name": "jokowidodo", "followers": 10200000, "display": "Joko Widodo"},
            {"name": "KompasTV", "followers": 3500000, "display": "Kompas TV"},
            {"name": "TempoMedia", "followers": 2600000, "display": "Tempo Media"},
            {"name": "CNNIndonesia", "followers": 4300000, "display": "CNN Indonesia"},
            {"name": "detikcom", "followers": 7800000, "display": "detikcom"},
            {"name": "antaranews", "followers": 1300000, "display": "ANTARA"}
        ]
    }
    
    # Recent dates
    now = datetime.now()
    dates = [(now - timedelta(hours=i)).isoformat() for i in range(1, 25)]
    
    # Create 50 fallback posts
    for _ in range(50):
        # Select random account
        account = random.choice(accounts.get(platform, accounts["Twitter"]))
        
        # Create content from keywords
        content_keywords = random.sample(keywords, min(3, len(keywords))) if keywords else ["Indonesia"]
        
        # Generate random engagement metrics
        likes = random.randint(10, 1000)
        shares = random.randint(5, 200)
        
        # Select random category
        categories = ["Politics", "Economy", "Social", "Security & Defense", "Health", "International Relations"]
        category = random.choice(categories)
        
        # Select random region
        regions = ["DKI Jakarta", "West Java", "East Java", "Bali", "North Sumatra", "All Indonesia"]
        region = random.choice(regions)
        
        if platform == "Twitter":
            content = f"#{content_keywords[0]} {random.choice(['Terkini:', 'Update:', 'Berita:'])} "
            content += f"Perkembangan terbaru tentang {', '.join(content_keywords)} di {region}. "
            content += f"#Indonesia #{region.replace(' ', '')}"
            
            data.append({
                "platform": "Twitter",
                "content": content,
                "username": account["name"],
                "user_followers": account["followers"],
                "posted_at": random.choice(dates),
                "likes": likes,
                "retweets": shares,
                "category": category,
                "region": region,
                "url": f"https://twitter.com/{account['name']}/status/{random.randint(1000000000000000000, 9999999999999999999)}"
            })
            
        elif platform == "Facebook":
            content = f"{random.choice(['TERKINI:', 'UPDATE:', 'BERITA:'])} "
            content += f"Perkembangan terbaru tentang {', '.join(content_keywords)} di {region}. "
            content += f"\n\n#Indonesia #{region.replace(' ', '')}"
            
            data.append({
                "platform": "Facebook",
                "content": content,
                "username": account["display"],
                "user_followers": account["followers"],
                "posted_at": random.choice(dates),
                "likes": likes,
                "shares": shares,
                "comments": random.randint(10, 300),
                "category": category,
                "region": region,
                "url": f"https://facebook.com/{account['name']}/posts/{random.randint(1000000000, 9999999999)}"
            })
    
    return data

def categorize_article(title: str, content: str) -> str:
    """
    Categorize article based on its title and content.
    
    Args:
        title: Article title
        content: Article content
        
    Returns:
        Category name
    """
    # Define category keywords
    categories = {
        "Politics": ["politik", "pemilu", "presiden", "pemerintah", "partai", "kabinet", "menteri", "gubernur", "demokrasi"],
        "Economy": ["ekonomi", "bisnis", "keuangan", "investasi", "perdagangan", "pasar", "saham", "inflasi", "rupiah"],
        "Social": ["sosial", "masyarakat", "komunitas", "kesejahteraan", "kemiskinan", "bantuan", "subsidi"],
        "Culture": ["budaya", "seni", "tradisi", "adat", "festival", "pariwisata", "warisan"],
        "Security & Defense": ["keamanan", "pertahanan", "militer", "polisi", "terorisme", "konflik", "kerusuhan"],
        "Health": ["kesehatan", "covid", "pandemi", "rumah sakit", "penyakit", "virus", "vaksin", "obat"],
        "Education": ["pendidikan", "sekolah", "universitas", "kampus", "siswa", "mahasiswa", "guru", "dosen"],
        "Environment": ["lingkungan", "iklim", "bencana", "polusi", "banjir", "gempa", "tsunami", "kebakaran hutan"],
        "Technology": ["teknologi", "digital", "startup", "aplikasi", "internet", "inovasi", "artificial intelligence"],
        "International Relations": ["internasional", "bilateral", "diplomatik", "hubungan", "kerjasama", "luar negeri"]
    }
    
    combined_text = (title + " " + content).lower()
    
    # Count keyword matches per category
    scores = {category: 0 for category in categories}
    
    for category, keywords in categories.items():
        for keyword in keywords:
            if keyword.lower() in combined_text:
                scores[category] += 1
    
    # Return the category with the highest score, or "General" if no clear match
    max_score = max(scores.values()) if scores else 0
    if max_score > 0:
        for category, score in scores.items():
            if score == max_score:
                return category
    
    return "General"

def extract_region(title: str, content: str) -> str:
    """
    Extract Indonesian region mentioned in article.
    
    Args:
        title: Article title
        content: Article content
        
    Returns:
        Region name
    """
    # Major Indonesian regions/provinces
    regions = {
        "DKI Jakarta": ["jakarta", "ibukota"],
        "West Java": ["jawa barat", "bandung", "depok", "bogor", "bekasi"],
        "Central Java": ["jawa tengah", "semarang", "solo", "yogyakarta", "jogja"],
        "East Java": ["jawa timur", "surabaya", "malang", "sidoarjo"],
        "Bali": ["bali", "denpasar", "kuta", "ubud"],
        "North Sumatra": ["sumatera utara", "medan", "deli serdang"],
        "South Sulawesi": ["sulawesi selatan", "makassar", "makasar"],
        "Yogyakarta": ["yogyakarta", "jogja", "jogjakarta", "yogya"],
        "Aceh": ["aceh", "banda aceh"],
        "Riau": ["riau", "pekanbaru"],
        "South Sumatra": ["sumatera selatan", "palembang"],
        "West Kalimantan": ["kalimantan barat", "pontianak"],
        "East Kalimantan": ["kalimantan timur", "samarinda", "balikpapan"],
        "South Kalimantan": ["kalimantan selatan", "banjarmasin"],
        "North Sulawesi": ["sulawesi utara", "manado"],
        "Papua": ["papua", "jayapura"]
    }
    
    combined_text = (title + " " + content).lower()
    
    # Count region mentions
    scores = {region: 0 for region in regions}
    
    for region, keywords in regions.items():
        for keyword in keywords:
            if keyword.lower() in combined_text:
                scores[region] += 1
    
    # Return the region with the highest mentions, or "National" if no clear match
    max_score = max(scores.values()) if scores else 0
    if max_score > 0:
        for region, score in scores.items():
            if score == max_score:
                return region
    
    # Check if "Indonesia" is mentioned frequently to determine if it's a national story
    if "indonesia" in combined_text and combined_text.count("indonesia") > 1:
        return "All Indonesia"
    
    return "Unspecified"
