"""
Explainer Module

Provides natural language explanations for candidate rankings.
Supports both rule-based and LLM-powered explanations.
"""

import logging
from typing import Dict, Any, List, Optional
import os

logger = logging.getLogger(__name__)


class ExplainerBot:
    """Chatbot for explaining candidate rankings."""
    
    def __init__(self, use_llm: bool = False):
        """
        Initialize explainer.
        
        Args:
            use_llm: Whether to use LLM (OpenAI/Anthropic) for explanations
        """
        self.use_llm = use_llm
        self.llm_client = None
        
        if use_llm:
            self._initialize_llm()
    
    def _initialize_llm(self):
        """Initialize LLM client if API keys are available."""
        openai_key = os.getenv('OPENAI_API_KEY')
        anthropic_key = os.getenv('ANTHROPIC_API_KEY')
        
        if openai_key:
            try:
                from openai import OpenAI
                self.llm_client = OpenAI(api_key=openai_key)
                self.llm_provider = 'openai'
                logger.info("Initialized OpenAI client")
            except Exception as e:
                logger.error(f"Failed to initialize OpenAI: {e}")
        
        elif anthropic_key:
            try:
                from anthropic import Anthropic
                self.llm_client = Anthropic(api_key=anthropic_key)
                self.llm_provider = 'anthropic'
                logger.info("Initialized Anthropic client")
            except Exception as e:
                logger.error(f"Failed to initialize Anthropic: {e}")
        
        if not self.llm_client:
            logger.warning("No LLM API key found, falling back to rule-based explanations")
            self.use_llm = False
    
    def answer_question(
        self,
        question: str,
        candidate: Dict[str, Any],
        job_description: str,
        all_candidates: List[Dict[str, Any]]
    ) -> str:
        """
        Answer a question about a candidate.
        
        Args:
            question: User's question
            candidate: Candidate data
            job_description: Job description
            all_candidates: List of all candidates for comparison
            
        Returns:
            Answer string
        """
        question_lower = question.lower()
        
        # Route to appropriate handler
        if 'rank higher' in question_lower or 'why' in question_lower and 'rank' in question_lower:
            return self._explain_ranking(candidate, all_candidates)
        
        elif 'skills' in question_lower and 'lack' in question_lower:
            return self._explain_missing_skills(candidate, job_description)
        
        elif 'experience' in question_lower or 'years' in question_lower:
            return self._explain_experience(candidate)
        
        elif 'qualif' in question_lower or 'strength' in question_lower:
            return self._explain_strengths(candidate)
        
        elif 'weakness' in question_lower or 'improve' in question_lower:
            return self._explain_weaknesses(candidate)
        
        elif 'compare' in question_lower:
            return self._compare_candidates(candidate, all_candidates)
        
        else:
            # Use LLM for open-ended questions if available
            if self.use_llm and self.llm_client:
                return self._llm_answer(question, candidate, job_description)
            else:
                return self._default_answer(candidate)
    
    def _explain_ranking(
        self,
        candidate: Dict[str, Any],
        all_candidates: List[Dict[str, Any]]
    ) -> str:
        """Explain why this candidate ranked where they did."""
        rank = next(
            (i + 1 for i, c in enumerate(all_candidates) if c['filename'] == candidate['filename']),
            None
        )
        
        if rank is None:
            return "Unable to determine candidate ranking."
        
        total_score = candidate['total_score']
        explanation = candidate.get('explanation', {})
        
        response = f"**Candidate ranked #{rank} out of {len(all_candidates)} with a score of {total_score:.1%}.**\n\n"
        
        # Score breakdown
        response += "**Score Breakdown:**\n"
        response += f"- Keyword Match: {candidate['keyword_score']:.1%}\n"
        response += f"- Semantic Similarity: {candidate['semantic_score']:.1%}\n"
        response += f"- Experience Match: {candidate['experience_score']:.1%}\n"
        response += f"- Skills Bonus: {candidate['skills_score']:.1%}\n\n"
        
        # Key strengths
        strengths = explanation.get('strengths', [])
        if strengths:
            response += "**Key Strengths:**\n"
            for strength in strengths[:3]:
                response += f"✓ {strength}\n"
            response += "\n"
        
        # Areas for improvement
        weaknesses = explanation.get('weaknesses', [])
        if weaknesses:
            response += "**Areas for Improvement:**\n"
            for weakness in weaknesses[:3]:
                response += f"• {weakness}\n"
        
        return response
    
    def _explain_missing_skills(
        self,
        candidate: Dict[str, Any],
        job_description: str
    ) -> str:
        """Explain what skills the candidate is missing."""
        from .preprocessing import extract_entities
        
        # Extract skills from job and resume
        job_entities = extract_entities(job_description)
        resume_entities = candidate.get('entities', extract_entities(candidate['cleaned_text']))
        
        required_skills = set(s.lower() for s in job_entities.get('SKILL', []))
        candidate_skills = set(s.lower() for s in resume_entities.get('SKILL', []))
        
        missing_skills = required_skills - candidate_skills
        matching_skills = required_skills.intersection(candidate_skills)
        
        response = f"**Skills Analysis for {candidate['filename']}:**\n\n"
        
        if matching_skills:
            response += f"**Has {len(matching_skills)} out of {len(required_skills)} required skills:**\n"
            for skill in sorted(matching_skills)[:10]:
                response += f"✓ {skill.title()}\n"
            response += "\n"
        
        if missing_skills:
            response += f"**Missing {len(missing_skills)} skills:**\n"
            for skill in sorted(missing_skills)[:10]:
                response += f"✗ {skill.title()}\n"
        else:
            response += "This candidate has all explicitly mentioned required skills!\n"
        
        return response
    
    def _explain_experience(self, candidate: Dict[str, Any]) -> str:
        """Explain candidate's experience."""
        from .preprocessing import extract_years_of_experience, extract_education_level
        
        text = candidate['cleaned_text']
        years = extract_years_of_experience(text)
        education = extract_education_level(text)
        
        response = f"**Experience Profile for {candidate['filename']}:**\n\n"
        response += f"**Years of Experience:** ~{years:.0f} years\n"
        response += f"**Education Level:** {education}\n\n"
        
        # Extract organizations
        entities = candidate.get('entities', {})
        orgs = entities.get('ORG', [])
        if orgs:
            response += f"**Previous Organizations:** {', '.join(orgs[:5])}\n"
        
        return response
    
    def _explain_strengths(self, candidate: Dict[str, Any]) -> str:
        """Explain candidate's key strengths."""
        explanation = candidate.get('explanation', {})
        strengths = explanation.get('strengths', [])
        
        response = f"**Key Strengths of {candidate['filename']}:**\n\n"
        
        if strengths:
            for i, strength in enumerate(strengths, 1):
                response += f"{i}. {strength}\n"
        else:
            response += "This candidate shows potential in several areas based on their background.\n"
        
        response += f"\n**Overall Assessment:** {explanation.get('summary', 'Relevant candidate')}"
        
        return response
    
    def _explain_weaknesses(self, candidate: Dict[str, Any]) -> str:
        """Explain candidate's areas for improvement."""
        explanation = candidate.get('explanation', {})
        weaknesses = explanation.get('weaknesses', [])
        
        response = f"**Areas for Improvement for {candidate['filename']}:**\n\n"
        
        if weaknesses:
            for i, weakness in enumerate(weaknesses, 1):
                response += f"{i}. {weakness}\n"
        else:
            response += "This candidate appears to be well-qualified with no major gaps identified.\n"
        
        return response
    
    def _compare_candidates(
        self,
        candidate: Dict[str, Any],
        all_candidates: List[Dict[str, Any]]
    ) -> str:
        """Compare candidate to others."""
        rank = next(
            (i + 1 for i, c in enumerate(all_candidates) if c['filename'] == candidate['filename']),
            None
        )
        
        response = f"**Comparison for {candidate['filename']} (Rank #{rank}):**\n\n"
        
        # Compare scores
        avg_keyword = sum(c['keyword_score'] for c in all_candidates) / len(all_candidates)
        avg_semantic = sum(c['semantic_score'] for c in all_candidates) / len(all_candidates)
        avg_total = sum(c['total_score'] for c in all_candidates) / len(all_candidates)
        
        response += "**vs. Average Candidate:**\n"
        response += f"- Keyword Match: {candidate['keyword_score']:.1%} (avg: {avg_keyword:.1%})\n"
        response += f"- Semantic Similarity: {candidate['semantic_score']:.1%} (avg: {avg_semantic:.1%})\n"
        response += f"- Overall Score: {candidate['total_score']:.1%} (avg: {avg_total:.1%})\n\n"
        
        # Relative position
        percentile = ((len(all_candidates) - rank + 1) / len(all_candidates)) * 100
        response += f"This candidate is in the top {percentile:.0f}% of applicants.\n"
        
        return response
    
    def _default_answer(self, candidate: Dict[str, Any]) -> str:
        """Default answer when question type is unclear."""
        explanation = candidate.get('explanation', {})
        
        response = f"**Summary for {candidate['filename']}:**\n\n"
        response += f"{explanation.get('summary', 'Relevant candidate for this position.')}\n\n"
        response += f"**Overall Score:** {candidate['total_score']:.1%}\n\n"
        response += "You can ask me:\n"
        response += "- Why did this candidate rank higher?\n"
        response += "- What skills does this candidate lack?\n"
        response += "- How many years of experience do they have?\n"
        response += "- What are their strongest qualifications?\n"
        
        return response
    
    def _llm_answer(
        self,
        question: str,
        candidate: Dict[str, Any],
        job_description: str
    ) -> str:
        """Use LLM to answer question."""
        try:
            # Prepare context
            context = f"""
Job Description:
{job_description[:1000]}

Candidate: {candidate['filename']}
Overall Score: {candidate['total_score']:.1%}
Keyword Score: {candidate['keyword_score']:.1%}
Semantic Score: {candidate['semantic_score']:.1%}
Experience Score: {candidate['experience_score']:.1%}
Skills Score: {candidate['skills_score']:.1%}

Resume Excerpt:
{candidate['cleaned_text'][:2000]}
"""
            
            prompt = f"{context}\n\nQuestion: {question}\n\nProvide a helpful, concise answer:"
            
            if self.llm_provider == 'openai':
                response = self.llm_client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": "You are a helpful HR assistant explaining candidate qualifications."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=300,
                    temperature=0.7
                )
                return response.choices[0].message.content
            
            elif self.llm_provider == 'anthropic':
                response = self.llm_client.messages.create(
                    model="claude-3-haiku-20240307",
                    max_tokens=300,
                    messages=[
                        {"role": "user", "content": prompt}
                    ]
                )
                return response.content[0].text
        
        except Exception as e:
            logger.error(f"LLM answer failed: {e}")
            return self._default_answer(candidate)