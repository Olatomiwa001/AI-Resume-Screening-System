"""
Preprocessing Module

Handles text cleaning, normalization, and Named Entity Recognition using spaCy.
"""

import re
import logging
from typing import List, Dict, Any
import spacy
from spacy.language import Language

logger = logging.getLogger(__name__)

# Load spaCy model (cached)
_nlp_model = None


def get_nlp_model() -> Language:
    """Get or load spaCy model (singleton pattern)."""
    global _nlp_model
    if _nlp_model is None:
        try:
            _nlp_model = spacy.load("en_core_web_sm")
            logger.info("Loaded spaCy model: en_core_web_sm")
        except OSError:
            logger.error("spaCy model not found. Run: python -m spacy download en_core_web_sm")
            raise
    return _nlp_model


def preprocess_text(text: str, lowercase: bool = False) -> str:
    """
    Clean and normalize text.
    
    Args:
        text: Raw text
        lowercase: Whether to convert to lowercase
        
    Returns:
        Cleaned text
    """
    if not text:
        return ""
    
    # Remove URLs
    text = re.sub(r'http\S+|www\.\S+', '', text)
    
    # Remove email addresses (but keep for entity extraction before this)
    text = re.sub(r'\S+@\S+', '', text)
    
    # Remove special characters but keep punctuation that might matter
    text = re.sub(r'[^\w\s.,!?;:()\-/]', ' ', text)
    
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Optional lowercase
    if lowercase:
        text = text.lower()
    
    return text.strip()


def extract_entities(text: str) -> Dict[str, List[str]]:
    """
    Extract named entities using spaCy.
    
    Args:
        text: Text to analyze
        
    Returns:
        Dictionary of entity types to entity lists
    """
    nlp = get_nlp_model()
    
    # Process text
    doc = nlp(text[:1000000])  # Limit to 1M chars for performance
    
    entities = {
        'PERSON': [],
        'ORG': [],
        'GPE': [],  # Geopolitical entity (countries, cities)
        'DATE': [],
        'PRODUCT': [],
        'SKILL': [],  # Custom extraction
    }
    
    for ent in doc.ents:
        if ent.label_ in entities:
            entities[ent.label_].append(ent.text)
    
    # Custom skill extraction (common technical skills)
    skill_patterns = [
        r'\b(Python|Java|JavaScript|C\+\+|Ruby|Go|Rust|Swift|Kotlin)\b',
        r'\b(React|Angular|Vue|Node\.js|Django|Flask|Spring|Express)\b',
        r'\b(SQL|MongoDB|PostgreSQL|MySQL|Redis|Elasticsearch)\b',
        r'\b(AWS|Azure|GCP|Docker|Kubernetes|Jenkins|Git)\b',
        r'\b(TensorFlow|PyTorch|scikit-learn|Keras|Pandas|NumPy)\b',
        r'\b(Machine Learning|Deep Learning|NLP|Computer Vision|Data Science)\b',
    ]
    
    for pattern in skill_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        entities['SKILL'].extend(matches)
    
    # Deduplicate
    for key in entities:
        entities[key] = list(set(entities[key]))
    
    return entities


def extract_years_of_experience(text: str) -> float:
    """
    Extract total years of experience from resume text.
    
    Args:
        text: Resume text
        
    Returns:
        Estimated years of experience
    """
    # Patterns for explicit year mentions
    patterns = [
        r'(\d+)\+?\s*years?\s+of\s+experience',
        r'experience\s*:\s*(\d+)\+?\s*years?',
        r'(\d+)\+?\s*years?\s+in\s+',
    ]
    
    years = []
    for pattern in patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        years.extend([int(m) for m in matches])
    
    if years:
        return max(years)
    
    # Fallback: try to extract date ranges and calculate
    date_pattern = r'(19|20)\d{2}'
    dates = re.findall(date_pattern, text)
    
    if dates:
        dates = [int(d) for d in dates]
        year_range = max(dates) - min(dates)
        return min(year_range, 40)  # Cap at 40 years
    
    return 0.0


def extract_education_level(text: str) -> str:
    """
    Extract highest education level.
    
    Args:
        text: Resume text
        
    Returns:
        Education level (PhD, Masters, Bachelors, Associate, High School, Unknown)
    """
    text_lower = text.lower()
    
    if any(term in text_lower for term in ['ph.d', 'phd', 'doctorate', 'doctoral']):
        return 'PhD'
    elif any(term in text_lower for term in ['master', 'msc', 'm.s.', 'mba', 'm.b.a']):
        return 'Masters'
    elif any(term in text_lower for term in ['bachelor', 'bsc', 'b.s.', 'b.a.', 'undergraduate']):
        return 'Bachelors'
    elif any(term in text_lower for term in ['associate', 'a.s.', 'a.a.']):
        return 'Associate'
    elif any(term in text_lower for term in ['high school', 'secondary school']):
        return 'High School'
    else:
        return 'Unknown'


def tokenize_for_tfidf(text: str) -> List[str]:
    """
    Tokenize text for TF-IDF computation.
    
    Args:
        text: Text to tokenize
        
    Returns:
        List of tokens
    """
    nlp = get_nlp_model()
    
    # Process text
    doc = nlp(text.lower())
    
    # Extract lemmatized tokens, excluding stop words and punctuation
    tokens = [
        token.lemma_ for token in doc
        if not token.is_stop and not token.is_punct and len(token.text) > 2
    ]
    
    return tokens