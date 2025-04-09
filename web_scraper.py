import trafilatura


def get_website_text_content(url: str) -> str:
    """
    This function takes a url and returns the main text content of the website.
    The text content is extracted using trafilatura and easier to understand.
    The results is not directly readable, better to be summarized by LLM before consume
    by the user.

    Some common website to crawl information from:
    MLB scores: https://www.mlb.com/scores/YYYY-MM-DD
    
    Args:
        url: The URL to scrape
        
    Returns:
        Extracted text content
    """
    try:
        # Send a request to the website
        downloaded = trafilatura.fetch_url(url)
        if downloaded:
            text = trafilatura.extract(downloaded)
            return text if text else ""
        return ""
    except Exception as e:
        print(f"Error extracting content from {url}: {str(e)}")
        return ""


if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) > 1:
        url = sys.argv[1]
        print(f"Scraping content from: {url}")
        content = get_website_text_content(url)
        print("\nExtracted content:")
        print("-" * 50)
        print(content[:1000] + "..." if len(content) > 1000 else content)
        print("-" * 50)
        print(f"Total content length: {len(content)} characters")
    else:
        print("Please provide a URL to scrape.")
        print("Example: python web_scraper.py https://example.com")