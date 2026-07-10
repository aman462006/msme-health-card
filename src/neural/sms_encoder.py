"""
Frozen multilingual sentence encoder for SMS/bank narration text.

PDF: "Frozen multilingual sentence encoder classifies SMS narrations as
salary credit, EMI debit, GST payment, utility, or other."

Uses paraphrase-multilingual-MiniLM-L12-v2 (sentence-transformers) -- same
384-dim space for Hindi, Marathi, Tamil, English SMS narrations.
Frozen = weights never updated, used purely for feature extraction.
"""
import numpy as np
from functools import lru_cache

SMS_CATEGORIES = ["salary_credit", "emi_debit", "gst_payment", "utility", "other"]

# Anchor phrases per category — cosine-similarity classifier, no fine-tuning needed
ANCHORS = {
    "salary_credit": [
        "salary credited", "salary deposited", "basic pay", "net salary",
        "vaetan", "maahevar", "maasik vetan",
    ],
    "emi_debit": [
        "EMI deducted", "loan installment", "auto debit EMI",
        "kist katai", "masik haft",
    ],
    "gst_payment": [
        "GST payment", "GSTN remittance", "integrated tax",
        "goods and services tax payment",
    ],
    "utility": [
        "electricity bill", "mobile recharge", "broadband bill",
        "bijli bill", "phone top-up",
    ],
    "other": ["transfer", "NEFT", "IMPS", "UPI debit", "cash withdrawal"],
}


@lru_cache(maxsize=1)
def _get_model():
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    model.eval()
    return model


@lru_cache(maxsize=1)
def _get_anchor_embeddings() -> dict:
    model = _get_model()
    return {
        cat: model.encode(phrases, convert_to_numpy=True, show_progress_bar=False).mean(axis=0)
        for cat, phrases in ANCHORS.items()
    }


def classify_sms(text: str) -> str:
    """Returns the most likely SMS category for a single narration string."""
    if not text or not text.strip():
        return "other"
    model = _get_model()
    anchors = _get_anchor_embeddings()
    emb = model.encode([text], convert_to_numpy=True, show_progress_bar=False)[0]
    sims = {cat: float(np.dot(emb, a) / (np.linalg.norm(emb) * np.linalg.norm(a) + 1e-9))
            for cat, a in anchors.items()}
    return max(sims, key=sims.get)


def encode_narration_batch(texts: list[str]) -> np.ndarray:
    """Returns (n, 384) embeddings for downstream feature use."""
    if not texts:
        return np.zeros((0, 384), dtype=np.float32)
    model = _get_model()
    return model.encode(texts, convert_to_numpy=True, show_progress_bar=False)


def tag_transactions(transactions: list[dict]) -> list[dict]:
    """
    Adds a 'tag' field to each transaction dict.
    Each dict should have a 'narration' key.
    """
    narrations = [t.get("narration", "") for t in transactions]
    model = _get_model()
    if not any(narrations):
        return [{**t, "tag": "other"} for t in transactions]
    anchors = _get_anchor_embeddings()
    embs = model.encode(narrations, convert_to_numpy=True, show_progress_bar=False)
    tagged = []
    for t, emb in zip(transactions, embs):
        sims = {cat: float(np.dot(emb, a) / (np.linalg.norm(emb) * np.linalg.norm(a) + 1e-9))
                for cat, a in anchors.items()}
        tag = max(sims, key=sims.get)
        tagged.append({**t, "tag": tag})
    return tagged
