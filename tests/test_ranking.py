"""
Tests for ranking module.
"""

import pytest
import numpy as np

from src.ranking import ResumeRanker
from src.embedding import EmbeddingManager


@pytest.fixture
def sample_resumes():
    """Sample resume data for testing."""
    return [
        {
            'filename': 'resume1.txt',
            'cleaned_text': 'Experienced Python developer with 5 years of experience. '
                          'Skills: Python, Django, SQL, AWS. Worked at TechCorp.',
            'entities': {'SKILL': ['Python', 'Django', 'SQL', 'AWS']}
        },
        {
            'filename': 'resume2.txt',
            'cleaned_text': 'Junior JavaScript developer with 1 year of experience. '
                          'Skills: JavaScript, React, HTML, CSS.',
            'entities': {'SKILL': ['JavaScript', 'React', 'HTML', 'CSS']}
        },
        {
            'filename': 'resume3.txt',
            'cleaned_text': 'Senior software engineer with 10 years of experience. '
                          'Expert in Python, Java, microservices, AWS, Docker.',
            'entities': {'SKILL': ['Python', 'Java', 'AWS', 'Docker']}
        }
    ]


@pytest.fixture
def sample_job_description():
    """Sample job description."""
    return """
Senior Python Developer needed with 5+ years of experience.
Required skills: Python, Django, AWS, Docker, SQL.
Experience with microservices is a plus.
"""


def test_ranker_initialization():
    """Test ResumeRanker initialization."""
    ranker = ResumeRanker()
    assert ranker is not None
    assert abs(ranker.weight_keyword + ranker.weight_semantic + 
              ranker.weight_experience + ranker.weight_skills - 1.0) < 0.01


def test_set_weights():
    """Test setting custom weights."""
    ranker = ResumeRanker()
    ranker.set_weights(keyword=0.4, semantic=0.3, experience=0.2, skills=0.1)
    
    assert abs(ranker.weight_keyword - 0.4) < 0.01
    assert abs(ranker.weight_semantic - 0.3) < 0.01


def test_rank_candidates(sample_resumes, sample_job_description):
    """Test ranking candidates."""
    ranker = ResumeRanker()
    embedding_manager = EmbeddingManager(enable_cache=False)
    
    results = ranker.rank_candidates(
        resumes=sample_resumes,
        job_description=sample_job_description,
        embedding_manager=embedding_manager
    )
    
    assert len(results) == 3
    assert all('total_score' in r for r in results)
    assert all('keyword_score' in r for r in results)
    assert all('semantic_score' in r for r in results)
    
    # Results should be sorted by score
    scores = [r['total_score'] for r in results]
    assert scores == sorted(scores, reverse=True)


def test_rank_empty_resumes(sample_job_description):
    """Test ranking with empty resume list."""
    ranker = ResumeRanker()
    embedding_manager = EmbeddingManager(enable_cache=False)
    
    results = ranker.rank_candidates(
        resumes=[],
        job_description=sample_job_description,
        embedding_manager=embedding_manager
    )
    
    assert results == []


def test_score_components(sample_resumes, sample_job_description):
    """Test that all score components are between 0 and 1."""
    ranker = ResumeRanker()
    embedding_manager = EmbeddingManager(enable_cache=False)
    
    results = ranker.rank_candidates(
        resumes=sample_resumes,
        job_description=sample_job_description,
        embedding_manager=embedding_manager
    )
    
    for result in results:
        assert 0 <= result['keyword_score'] <= 1
        assert 0 <= result['semantic_score'] <= 1
        assert 0 <= result['experience_score'] <= 1
        assert 0 <= result['skills_score'] <= 1
        assert 0 <= result['total_score'] <= 1


def test_explanation_generation(sample_resumes, sample_job_description):
    """Test that explanations are generated."""
    ranker = ResumeRanker()
    embedding_manager = EmbeddingManager(enable_cache=False)
    
    results = ranker.rank_candidates(
        resumes=sample_resumes,
        job_description=sample_job_description,
        embedding_manager=embedding_manager
    )
    
    for result in results:
        assert 'explanation' in result
        assert 'strengths' in result['explanation']
        assert 'weaknesses' in result['explanation']
        assert 'summary' in result['explanation']