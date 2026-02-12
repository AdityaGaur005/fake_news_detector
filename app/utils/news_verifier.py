from app.utils.verify import classify_news
from news_lookup import search_news_online
from app.utils.claim_support import claim_support_score

def verify_news(user_text: str):
    articles = search_news_online(user_text)

    if articles:
        score, best_article = claim_support_score(user_text, articles)

        if score >= 0.72:
            confidence = min(int(score * 100) + 10, 95)

            return {
                "status": "verified_online",
                "verdict": "Real (verified online)",
                "confidence": confidence,
                "sources": [
                    {
                        "title": best_article["title"],
                        "source": best_article["source"],
                        "url": best_article["url"]
                    }
                ]

            }

    # Internet failed → ML fallback
    label, confidence = classify_news(user_text)

    return {
        "status": "ml_prediction",
        "verdict": f"{label} (ML-based)",
        "confidence": confidence,
        "sources": []
    }
