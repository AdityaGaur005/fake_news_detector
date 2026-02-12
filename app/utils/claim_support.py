from sentence_transformers import SentenceTransformer, util

# Load embedding model ONCE
model = SentenceTransformer("all-MiniLM-L6-v2")

def claim_support_score(claim: str, articles: list):
    """
    Returns:
    - best semantic similarity score
    - best matching article (dict)
    """

    claim_emb = model.encode(claim, convert_to_tensor=True)

    best_score = 0.0
    best_article = None

    for art in articles:
        # Internet articles are structured dicts
        content = f"{art.get('title', '')} {art.get('description', '')}"

        art_emb = model.encode(content, convert_to_tensor=True)
        score = util.cos_sim(claim_emb, art_emb).item()

        if score > best_score:
            best_score = score
            best_article = art

    return best_score, best_article
