import os
import re
import math
import numpy as np
import docx
from pypdf import PdfReader
import nltk
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def _ensure_nltk_resources():
    """Lazily downloads necessary NLTK corpora packages only if missing."""
    resources = {
        'tokenizers/punkt': 'punkt',
        'corpora/wordnet': 'wordnet',
        'corpora/omw-1.4': 'omw-1.4'
    }
    for path, package in resources.items():
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(package, quiet=True)

class DocumentSummarizerService:
    def __init__(self):
        _ensure_nltk_resources()
        self.lemmatizer = WordNetLemmatizer()
        self.sent_boundary_regex = re.compile(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=[.?!])\s+')

    def extract_text_from_file(self, file_path: str) -> str:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Target document not found at path: {file_path}")
            
        ext = os.path.splitext(file_path)[1].lower()
        if ext == ".txt":
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                return f.read().strip()
        elif ext == ".docx":
            doc = docx.Document(file_path)
            return "\n\n".join([p.text for p in doc.paragraphs if p.text.strip()]).strip()
        elif ext == ".pdf":
            reader = PdfReader(file_path)
            pages = [page.extract_text() for page in reader.pages if page.extract_text()]
            return "\n\n".join(pages).strip()
        else:
            raise ValueError("Unsupported format profile. Use .txt, .docx, or .pdf modules.")

    def chunk_text(self, text: str, chunk_size: int = 150) -> list:
        paragraphs = [p.strip() for p in text.split("\n\n") if len(p.strip()) > 30]
        if len(paragraphs) <= 1:
            sentences = self.sent_boundary_regex.split(text)
            paragraphs = []
            temp_chunk = []
            word_count = 0
            for sent in sentences:
                temp_chunk.append(sent)
                word_count += len(sent.split())
                if word_count >= chunk_size:
                    paragraphs.append(" ".join(temp_chunk))
                    temp_chunk = []
                    word_count = 0
            if temp_chunk:
                paragraphs.append(" ".join(temp_chunk))
        return [p for p in paragraphs if len(p.split()) > 10]

    def run_sentiment_analysis(self, text: str) -> dict:
        pos_words = {'good', 'great', 'excellent', 'amazing', 'positive', 'advantage', 'benefit', 'innovative', 'valuable', 'efficient', 'faster', 'easier', 'success'}
        neg_words = {'bad', 'poor', 'expensive', 'slow', 'difficult', 'negative', 'limitation', 'fail', 'error', 'scramble', 'bottleneck', 'risk', 'delay'}
        
        words = re.findall(r'\b\w+\b', text.lower())
        pos_count = sum(1 for w in words if w in pos_words)
        neg_count = sum(1 for w in words if w in neg_words)
        
        total = pos_count + neg_count
        if total == 0:
            return {"label": "Neutral 😐", "score": 50}
            
        pos_ratio = (pos_count / total) * 100
        if pos_ratio > 55:
            return {"label": "Positive 😊", "score": int(pos_ratio)}
        elif pos_ratio < 45:
            return {"label": "Analytical/Critical 😟", "score": int(100 - pos_ratio)}
        return {"label": "Balanced/Objective 😐", "score": 50}

    def extract_keywords(self, chunks: list) -> list:
        if not chunks:
            return []
        try:
            # sublinear_tf scales terms defensively to prevent long-chunk biases
            vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2), sublinear_tf=True)
            tfidf_matrix = vectorizer.fit_transform(chunks)
            scores = np.asarray(tfidf_matrix.sum(axis=0)).flatten()
            features = vectorizer.get_feature_names_out()
            
            top_indices = np.argsort(scores)[::-1]
            keywords = []
            for idx in top_indices:
                term = features[idx]
                if len(term) > 4 and not term.isdigit() and " " in term:
                    keywords.append(term.title())
                if len(keywords) >= 6:
                    break
            return keywords
        except Exception:
            return []

    def generate_extractive_summary(self, chunks: list, ratio: float) -> str:
        if not chunks:
            return ""
        try:
            vectorizer = TfidfVectorizer(stop_words='english', lowercase=True)
            tfidf_matrix = vectorizer.fit_transform(chunks)
            
            similarity_matrix = cosine_similarity(tfidf_matrix)
            centrality_scores = similarity_matrix.mean(axis=1)
            
            num_chunks = max(1, int(len(chunks) * (ratio / 100.0)))
            top_indices = sorted(np.argsort(centrality_scores)[-num_chunks:])
            return "\n\n".join([chunks[idx] for idx in top_indices])
        except ValueError:
            # Handles edge case where chunk only contains words excluded by stop_word filters
            return chunks[0] if chunks else ""
        except Exception:
            return chunks[0] if chunks else ""

    def analyze_document_text(self, raw_text: str, ratio: float = 30.0) -> dict:
        chunks = self.chunk_text(raw_text)
        summary = self.generate_extractive_summary(chunks, ratio)
        keywords = self.extract_keywords(chunks)
        sentiment = self.run_sentiment_analysis(raw_text)
        
        total_words = len(raw_text.split())
        return {
            "metrics": {
                "word_count": total_words,
                "character_count": len(raw_text),
                "segments": len(chunks),
                "estimated_reading_time_mins": max(1, math.ceil(total_words / 220))
            },
            "sentiment": sentiment,
            "keywords": keywords,
            "summary": summary
        }