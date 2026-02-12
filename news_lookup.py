# news_lookup.py

import requests

# -----------------------------
# 1️⃣ Set your API key here
# -----------------------------
API_KEY = "9ffd0edc8b4e46a6a8adc5333b13a6bd"  # Replace with your own API key
NEWSAPI_URL = "https://newsapi.org/v2/everything"

# -----------------------------
# 2️⃣ Function to search news
# -----------------------------
def search_news_online(query, max_results=3):
    """
    Search for news articles online using NewsAPI.
    Returns a list of sources or an empty list if nothing found.
    """
    params = {
        "qInTitle": query,
        "apiKey": API_KEY,
        "language": "en",
        "sortBy": "relevancy",
        "pageSize": max_results
    }
    
    response = requests.get(NEWSAPI_URL, params=params)
    
    if response.status_code != 200:
        print(f"Error fetching news: {response.status_code}")
        return []
    
    data = response.json()
    articles = data.get("articles", [])
    
    sources = []
    for article in articles:
        title = article.get("title", "")
        source_name = article.get("source", {}).get("name", "")
        url = article.get("url", "")
        sources.append({
            "title": title,
            "description": article.get("description", ""),
            "source": source_name,
            "url": url
    })
    
    return sources

# -----------------------------
# 3️⃣ Optional test
# -----------------------------
if __name__ == "__main__":
    query = "india wins asia cup 2025"
    results = search_news_online(query)
    
    if results:
        print(f"Found {len(results)} articles:")
        for r in results:
            print("-", r)
    else:
        print("No trusted sources found.")
