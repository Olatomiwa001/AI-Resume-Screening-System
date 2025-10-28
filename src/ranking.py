"""
Ranking Module

Computes relevance scores for resumes against job descriptions.
Uses multiple scoring components:
- Keyword matching (TF-IDF)
- Semantic similarity (embeddings)
- Experience matching
- Skills bonus
"""

import logging
from typing import List, Dict, Any, Optional
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re

from .preprocessing import (
    extract_years_of_experience,
    extract_entities,
    extract_education_level,
    tokenize_for_tfidf
)
from .embedding import EmbeddingManager

logger = logging.getLogger(__name__)


class ResumeRanker:
    """Ranks resumes against job descriptions."""
    
    def __init__(
        self,
        weight_keyword: float = 0.3,
        weight_semantic: float = 0.4,
        weight_experience: float = 0.2,
        weight_skills: float = 0.1
    ):
        """
        Initialize ranker with scoring weights.
        
        Args:
            weight_keyword: Weight for keyword matching
            weight_semantic: Weight for semantic similarity
            weight_experience: Weight for experience matching
            weight_skills: Weight for skills bonus
        """
        self.weight_keyword = weight_keyword
        self.weight_semantic = weight_semantic
        self.weight_experience = weight_experience
        self.weight_skills = weight_skills
        
        # Ensure weights sum to 1
        total = sum([weight_keyword, weight_semantic, weight_experience, weight_skills])
        if abs(total - 1.0) > 0.01:
            logger.warning(f"Weights sum to {total}, normalizing...")
            self.weight_keyword /= total
            self.weight_semantic /= total
            self.weight_experience /= total
            self.weight_skills /= total
        
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=500,
            ngram_range=(1, 2),
            stop_words='english'
        )
    
    def set_weights(
        self,
        keyword: float,
        semantic: float,
        experience: float,
        skills: float
    ):
        """Update scoring weights."""
        total = keyword + semantic + experience + skills
        self.weight_keyword = keyword / total
        self.weight_semantic = semantic / total
        self.weight_experience = experience / total
        self.weight_skills = skills / total
        
        logger.info(f"Updated weights: K={self.weight_keyword:.2f}, "
                   f"S={self.weight_semantic:.2f}, E={self.weight_experience:.2f}, "
                   f"Sk={self.weight_skills:.2f}")
    
    def rank_candidates(
        self,
        resumes: List[Dict[str, Any]],
        job_description: str,
        embedding_manager: EmbeddingManager
    ) -> List[Dict[str, Any]]:
        """
        Rank candidates against job description.
        
        Args:
            resumes: List of resume dictionaries with 'cleaned_text' key
            job_description: Job description text
            embedding_manager: EmbeddingManager instance
            
        Returns:
            List of ranked candidates with scores
        """
        if not resumes:
            return []
        
        logger.info(f"Ranking {len(resumes)} candidates")
        
        # Extract texts
        resume_texts = [r['cleaned_text'] for r in resumes]
        
        # Compute keyword scores
        keyword_scores = self._compute_keyword_scores(resume_texts, job_description)
        
        # Compute semantic scores
        semantic_scores = self._compute_semantic_scores(
            resume_texts, job_description, embedding_manager
        )
        
        # Compute experience scores
        experience_scores = self._compute_experience_scores(resume_texts, job_description)
        
        # Compute skills scores
        skills_scores = self._compute_skills_scores(resume_texts, job_description)
        
        # Combine scores
        results = []
        for idx, resume in enumerate(resumes):
            total_score = (
                self.weight_keyword * keyword_scores[idx] +
                self.weight_semantic * semantic_scores[idx] +
                self.weight_experience * experience_scores[idx] +
                self.weight_skills * skills_scores[idx]
            )
            
            # Generate explanation
            explanation = self._generate_explanation(
                resume=resume,
                keyword_score=keyword_scores[idx],
                semantic_score=semantic_scores[idx],
                experience_score=experience_scores[idx],
                skills_score=skills_scores[idx],
                job_description=job_description
            )
            
            result = {
                **resume,
                'total_score': total_score,
                'keyword_score': keyword_scores[idx],
                'semantic_score': semantic_scores[idx],
                'experience_score': experience_scores[idx],
                'skills_score': skills_scores[idx],
                'explanation': explanation
            }
            results.append(result)
        
        # Sort by total score
        results.sort(key=lambda x: x['total_score'], reverse=True)
        
        logger.info(f"Ranking complete. Top score: {results[0]['total_score']:.3f}")
        return results
    
    def _compute_keyword_scores(
        self,
        resume_texts: List[str],
        job_description: str
    ) -> np.ndarray:
        """Compute TF-IDF based keyword matching scores."""
        try:
            # Fit on all documents
            all_texts = resume_texts + [job_description]
            tfidf_matrix = self.tfidf_vectorizer.fit_transform(all_texts)
            
            # Job description is last
            job_vector = tfidf_matrix[-1]
            resume_vectors = tfidf_matrix[:-1]
            
            # Compute cosine similarity
            similarities = cosine_similarity(resume_vectors, job_vector).flatten()
            
            return similarities
        
        except Exception as e:
            logger.error(f"Error computing keyword scores: {e}")
            return np.zeros(len(resume_texts))
    
    def _compute_semantic_scores(
        self,
        resume_texts: List[str],
        job_description: str,
        embedding_manager: EmbeddingManager
    ) -> np.ndarray:
        """Compute semantic similarity scores using embeddings."""
        try:
            # Encode all texts
            all_texts = resume_texts + [job_description]
            embeddings = embedding_manager.encode(all_texts)
            
            # Job description embedding is last
            job_embedding = embeddings[-1].reshape(1, -1)
            resume_embeddings = embeddings[:-1]
            
            # Compute cosine similarity
            similarities = cosine_similarity(resume_embeddings, job_embedding).flatten()
            
            return similarities
        
        except Exception as e:
            logger.error(f"Error computing semantic scores: {e}")
            return np.zeros(len(resume_texts))
    
    def _compute_experience_scores(
        self,
        resume_texts: List[str],
        job_description: str
    ) -> np.ndarray:
        """Compute experience matching scores."""
        # Extract required years from job description
        required_years = self._extract_required_years(job_description)
        
        scores = []
        for resume_text in resume_texts:
            candidate_years = extract_years_of_experience(resume_text)
            
            if required_years == 0:
                # No specific requirement
                score = min(candidate_years / 10.0, 1.0)  # Cap at 10 years
            else:
                # Score based on how close to requirement
                if candidate_years >= required_years:
                    score = 1.0
                else:
                    score = candidate_years / required_years
            
            scores.append(score)
        
        return np.array(scores)
    
    def _extract_required_years(self, job_description: str) -> float:
        """Extract required years of experience from job description."""
        patterns = [
            r'(\d+)\+?\s*years?\s+of\s+experience',
            r'minimum\s+(\d+)\s+years?',
            r'at least\s+(\d+)\s+years?',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, job_description, re.IGNORECASE)
            if match:
                return float(match.group(1))
        
        return 0.0
    
    def _compute_skills_scores(
        self,
        resume_texts: List[str],
        job_description: str
    ) -> np.ndarray:
        """Compute skills bonus scores."""
        # Extract required skills from job description
        job_entities = extract_entities(job_description)
        required_skills = set(s.lower() for s in job_entities.get('SKILL', []))
        
        # Also look for explicit skill mentions
        skill_keywords = self._extract_skill_keywords(job_description)
        required_skills.update(skill_keywords)
        
        if not required_skills:
            return np.ones(len(resume_texts))  # No specific requirements
        
        scores = []
        for resume_text in resume_texts:
            resume_entities = extract_entities(resume_text)
            candidate_skills = set(s.lower() for s in resume_entities.get('SKILL', []))
            candidate_skills.update(self._extract_skill_keywords(resume_text))
            
            # Calculate overlap
            matched_skills = required_skills.intersection(candidate_skills)
            score = len(matched_skills) / len(required_skills) if required_skills else 0.0
            scores.append(min(score, 1.0))
        
        return np.array(scores)
    
    def _extract_skill_keywords(self, text: str) -> set:
        """Extract skill keywords from text."""
        text_lower = text.lower()
        skills = set()
        
        # Common technical skills
        skill_list = [
            'python', 'java', 'javascript', 'c++', 'c#', 'ruby', 'go', 'rust',
            'react', 'angular', 'vue', 'node.js', 'django', 'flask', 'spring',
            'sql', 'mongodb', 'postgresql', 'mysql', 'redis',
            'aws', 'azure', 'gcp', 'docker', 'kubernetes', 'jenkins',
            'machine learning', 'deep learning', 'nlp', 'computer vision',
            'tensorflow', 'pytorch', 'scikit-learn', 'pandas', 'numpy',
            'rest api', 'graphql', 'microservices', 'agile', 'scrum',
        ]
        
        for skill in skill_list:
            if skill in text_lower:
                skills.add(skill)
        
        return skills
    
    def _generate_explanation(
        self,
        resume: Dict[str, Any],
        keyword_score: float,
        semantic_score: float,
        experience_score: float,
        skills_score: float,
        job_description: str
    ) -> Dict[str, Any]:
        """Generate explanation for candidate score."""
        explanation = {
            'strengths': [],
            'weaknesses': [],
            'summary': ''
        }
        
        # Analyze strengths
        if keyword_score > 0.7:
            explanation['strengths'].append(
                f"Strong keyword match ({keyword_score:.1%}) - resume contains many relevant terms"
            )
        
        if semantic_score > 0.7:
            explanation['strengths'].append(
                f"Excellent semantic fit ({semantic_score:.1%}) - experience aligns well with role"
            )
        
        if experience_score > 0.8:
            explanation['strengths'].append(
                f"Meets or exceeds experience requirements ({experience_score:.1%})"
            )
        
        if skills_score > 0.7:
            explanation['strengths'].append(
                f"Has most required skills ({skills_score:.1%})"
            )
        
        # Analyze weaknesses
        if keyword_score < 0.4:
            explanation['weaknesses'].append(
                f"Limited keyword match ({keyword_score:.1%}) - fewer relevant terms found"
            )
        
        if semantic_score < 0.4:
            explanation['weaknesses'].append(
                f"Lower semantic alignment ({semantic_score:.1%}) - experience may not directly match"
            )
        
        if experience_score < 0.5:
            explanation['weaknesses'].append(
                f"May lack sufficient experience ({experience_score:.1%})"
            )
        
        if skills_score < 0.5:
            explanation['weaknesses'].append(
                f"Missing some required skills ({skills_score:.1%})"
            )
        
        # Generate summary
        total = (
            self.weight_keyword * keyword_score +
            self.weight_semantic * semantic_score +
            self.weight_experience * experience_score +
            self.weight_skills * skills_score
        )
        
        if total >= 0.7:
            explanation['summary'] = "Excellent candidate with strong qualifications"
        elif total >= 0.5:
            explanation['summary'] = "Good candidate with relevant experience"
        elif total >= 0.3:
            explanation['summary'] = "Potential candidate with some matching qualifications"
        else:
            explanation['summary'] = "Limited match with job requirements"
        
        return explanation