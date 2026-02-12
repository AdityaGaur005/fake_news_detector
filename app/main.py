from fastapi import FastAPI
from pydantic import BaseModel
from app.utils.news_verifier import verify_news

# FastAPI app instance (THIS NAME IS CRITICAL)
app = FastAPI(
    title="Fake News Detector API",
    description="Hybrid fake news detection using online verification + ML",
    version="1.0"
)

# Request schema
class NewsRequest(BaseModel):
    news: str

# Health check
@app.get("/")
def root():
    return {"message": "Fake News Detector API is running"}

# Main API endpoint
@app.post("/verify-news")
def verify_news_api(request: NewsRequest):
    return verify_news(request.news)
