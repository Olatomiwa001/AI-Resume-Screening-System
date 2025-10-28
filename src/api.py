"""
API Module

Optional FastAPI wrapper for programmatic resume scoring.
"""

import logging
from typing import Dict, Any, List, Optional
from pathlib import Path
import tempfile
import os

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from .text_extraction import extract_text_from_pdf, extract_text_from_docx
from .preprocessing import preprocess_text, extract_entities
from .embedding import EmbeddingManager
from .ranking import ResumeRanker

logger = logging.getLogger(__name__)

# Initialize app
app = FastAPI(
    title="AI Resume Screening API",
    description="API for scoring and ranking resumes against job descriptions",
    version="1.0.0"
)

# Initialize components (singletons)
embedding_manager = EmbeddingManager()
ranker = ResumeRanker()


# Request/Response models
class ScoreRequest(BaseModel):
    """Request model for scoring a resume."""
    resume_text: str
    job_description: str
    

class ScoreResponse(BaseModel):
    """Response model for resume score."""
    total_score: float
    keyword_score: float
    semantic_score: float
    experience_score: float
    skills_score: float
    explanation: Dict[str, Any]


class BatchScoreRequest(BaseModel):
    """Request model for batch scoring."""
    job_description: str
    resumes: List[Dict[str, str]]  # List of {"filename": "...", "text": "..."}


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "AI Resume Screening API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "score": "/score",
            "score_file": "/score-file",
            "batch_score": "/batch-score"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}


@app.post("/score", response_model=ScoreResponse)
async def score_resume(request: ScoreRequest):
    """
    Score a single resume against a job description.
    
    Args:
        request: ScoreRequest with resume text and job description
        
    Returns:
        ScoreResponse with scores and explanation
    """
    try:
        # Preprocess
        cleaned_resume = preprocess_text(request.resume_text)
        entities = extract_entities(cleaned_resume)
        
        resume_data = {
            'filename': 'api_resume',
            'raw_text': request.resume_text,
            'cleaned_text': cleaned_resume,
            'entities': entities
        }
        
        # Rank (single resume)
        results = ranker.rank_candidates(
            resumes=[resume_data],
            job_description=request.job_description,
            embedding_manager=embedding_manager
        )
        
        if not results:
            raise HTTPException(status_code=500, detail="Scoring failed")
        
        candidate = results[0]
        
        return ScoreResponse(
            total_score=candidate['total_score'],
            keyword_score=candidate['keyword_score'],
            semantic_score=candidate['semantic_score'],
            experience_score=candidate['experience_score'],
            skills_score=candidate['skills_score'],
            explanation=candidate['explanation']
        )
    
    except Exception as e:
        logger.error(f"Error scoring resume: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/score-file")
async def score_resume_file(
    file: UploadFile = File(...),
    job_description: str = Form(...)
):
    """
    Score an uploaded resume file.
    
    Args:
        file: Resume file (PDF or DOCX)
        job_description: Job description text
        
    Returns:
        Score response
    """
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix) as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_path = tmp_file.name
        
        try:
            # Extract text
            if file.filename.lower().endswith('.pdf'):
                text = extract_text_from_pdf(tmp_path)
            elif file.filename.lower().endswith(('.docx', '.doc')):
                text = extract_text_from_docx(tmp_path)
            else:
                raise HTTPException(status_code=400, detail="Unsupported file format")
            
            # Clean up temp file
            os.unlink(tmp_path)
            
            # Score
            cleaned_text = preprocess_text(text)
            entities = extract_entities(cleaned_text)
            
            resume_data = {
                'filename': file.filename,
                'raw_text': text,
                'cleaned_text': cleaned_text,
                'entities': entities
            }
            
            results = ranker.rank_candidates(
                resumes=[resume_data],
                job_description=job_description,
                embedding_manager=embedding_manager
            )
            
            candidate = results[0]
            
            return {
                "filename": file.filename,
                "total_score": candidate['total_score'],
                "keyword_score": candidate['keyword_score'],
                "semantic_score": candidate['semantic_score'],
                "experience_score": candidate['experience_score'],
                "skills_score": candidate['skills_score'],
                "explanation": candidate['explanation']
            }
        
        finally:
            # Ensure temp file is deleted
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
    
    except Exception as e:
        logger.error(f"Error scoring file: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/batch-score")
async def batch_score_resumes(request: BatchScoreRequest):
    """
    Score multiple resumes in batch.
    
    Args:
        request: BatchScoreRequest with job description and resume texts
        
    Returns:
        List of ranked candidates
    """
    try:
        # Prepare resume data
        resumes = []
        for resume in request.resumes:
            cleaned_text = preprocess_text(resume['text'])
            entities = extract_entities(cleaned_text)
            
            resumes.append({
                'filename': resume.get('filename', 'unknown'),
                'raw_text': resume['text'],
                'cleaned_text': cleaned_text,
                'entities': entities
            })
        
        # Rank all
        results = ranker.rank_candidates(
            resumes=resumes,
            job_description=request.job_description,
            embedding_manager=embedding_manager
        )
        
        # Format response
        return {
            "total_candidates": len(results),
            "job_description": request.job_description[:100] + "...",
            "ranked_candidates": [
                {
                    "rank": idx + 1,
                    "filename": r['filename'],
                    "total_score": r['total_score'],
                    "scores": {
                        "keyword": r['keyword_score'],
                        "semantic": r['semantic_score'],
                        "experience": r['experience_score'],
                        "skills": r['skills_score']
                    },
                    "explanation": r['explanation']
                }
                for idx, r in enumerate(results)
            ]
        }
    
    except Exception as e:
        logger.error(f"Error in batch scoring: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/update-weights")
async def update_scoring_weights(
    keyword: float = 0.3,
    semantic: float = 0.4,
    experience: float = 0.2,
    skills: float = 0.1
):
    """
    Update scoring weights.
    
    Args:
        keyword: Keyword match weight
        semantic: Semantic similarity weight
        experience: Experience match weight
        skills: Skills bonus weight
        
    Returns:
        Updated weights
    """
    try:
        ranker.set_weights(
            keyword=keyword,
            semantic=semantic,
            experience=experience,
            skills=skills
        )
        
        return {
            "message": "Weights updated successfully",
            "weights": {
                "keyword": ranker.weight_keyword,
                "semantic": ranker.weight_semantic,
                "experience": ranker.weight_experience,
                "skills": ranker.weight_skills
            }
        }
    
    except Exception as e:
        logger.error(f"Error updating weights: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)