"""
AI Resume Screening System - Streamlit Web Interface

This is the main entry point for the web application.
"""

import streamlit as st
import os
import sys
from pathlib import Path
from typing import List, Dict, Any
import tempfile
import logging

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.text_extraction import extract_text_from_pdf, extract_text_from_docx
from src.preprocessing import preprocess_text, extract_entities
from src.embedding import EmbeddingManager
from src.ranking import ResumeRanker
from src.explainer import ExplainerBot
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="AI Resume Screener",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .candidate-card {
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #ddd;
        margin: 0.5rem 0;
        background-color: #f9f9f9;
    }
    .score-badge {
        font-size: 1.5rem;
        font-weight: bold;
        color: #ffffff;
        padding: 0.5rem 1rem;
        border-radius: 0.5rem;
        display: inline-block;
    }
    .score-high {
        background-color: #28a745;
    }
    .score-medium {
        background-color: #ffc107;
    }
    .score-low {
        background-color: #dc3545;
    }
</style>
""", unsafe_allow_html=True)


# Initialize session state
def init_session_state():
    """Initialize Streamlit session state variables."""
    if 'resumes' not in st.session_state:
        st.session_state.resumes = []
    if 'ranked_candidates' not in st.session_state:
        st.session_state.ranked_candidates = []
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'selected_candidate' not in st.session_state:
        st.session_state.selected_candidate = None
    if 'embedding_manager' not in st.session_state:
        st.session_state.embedding_manager = EmbeddingManager()
    if 'ranker' not in st.session_state:
        st.session_state.ranker = ResumeRanker()
    if 'explainer' not in st.session_state:
        use_llm = st.session_state.get('use_llm_explainer', False)
        st.session_state.explainer = ExplainerBot(use_llm=use_llm)


def process_uploaded_file(uploaded_file) -> Dict[str, Any]:
    """Process an uploaded resume file."""
    try:
        # Save to temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = tmp_file.name
        
        # Extract text based on file type
        if uploaded_file.name.lower().endswith('.pdf'):
            text = extract_text_from_pdf(tmp_path)
        elif uploaded_file.name.lower().endswith(('.docx', '.doc')):
            text = extract_text_from_docx(tmp_path)
        else:
            raise ValueError(f"Unsupported file format: {uploaded_file.name}")
        
        # Clean up temp file
        os.unlink(tmp_path)
        
        # Preprocess
        cleaned_text = preprocess_text(text)
        entities = extract_entities(cleaned_text)
        
        return {
            'filename': uploaded_file.name,
            'raw_text': text,
            'cleaned_text': cleaned_text,
            'entities': entities
        }
    
    except Exception as e:
        logger.error(f"Error processing file {uploaded_file.name}: {e}")
        st.error(f"Failed to process {uploaded_file.name}: {str(e)}")
        return None


def render_score_badge(score: float) -> str:
    """Render a colored score badge."""
    if score >= 0.7:
        css_class = "score-high"
    elif score >= 0.5:
        css_class = "score-medium"
    else:
        css_class = "score-low"
    
    return f'<span class="score-badge {css_class}">{score:.1%}</span>'


def main():
    """Main application."""
    init_session_state()
    
    # Header
    st.markdown('<h1 class="main-header">📄 AI Resume Screening System</h1>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Scoring weights
        st.subheader("Scoring Weights")
        weight_keyword = st.slider("Keyword Match", 0.0, 1.0, 0.3, 0.05)
        weight_semantic = st.slider("Semantic Similarity", 0.0, 1.0, 0.4, 0.05)
        weight_experience = st.slider("Experience", 0.0, 1.0, 0.2, 0.05)
        weight_skills = st.slider("Skills Bonus", 0.0, 1.0, 0.1, 0.05)
        
        # Update ranker weights
        st.session_state.ranker.set_weights(
            keyword=weight_keyword,
            semantic=weight_semantic,
            experience=weight_experience,
            skills=weight_skills
        )
        
        st.divider()
        
        # Explainer configuration
        st.subheader("Chatbot Configuration")
        use_llm = st.checkbox("Use LLM for Explanations", value=False,
                             help="Enable OpenAI/Anthropic for richer explanations")
        
        if use_llm:
            llm_provider = st.radio("LLM Provider", ["OpenAI", "Anthropic"])
            api_key_set = bool(os.getenv(f"{llm_provider.upper()}_API_KEY"))
            if api_key_set:
                st.success(f"✓ {llm_provider} API key detected")
            else:
                st.warning(f"⚠️ Set {llm_provider.upper()}_API_KEY in .env")
        
        if st.session_state.get('use_llm_explainer') != use_llm:
            st.session_state.use_llm_explainer = use_llm
            st.session_state.explainer = ExplainerBot(use_llm=use_llm)
        
        st.divider()
        
        # Demo data
        if st.button("🎲 Load Demo Data"):
            demo_path = Path("data/demo")
            if demo_path.exists():
                st.info("Demo data loaded! Upload the PDF files from data/demo/resumes/")
            else:
                st.warning("Run: python scripts/generate_demo_data.py first")
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("1️⃣ Upload Resumes")
        uploaded_files = st.file_uploader(
            "Upload resume files (PDF or DOCX)",
            type=['pdf', 'docx', 'doc'],
            accept_multiple_files=True
        )
        
        if uploaded_files:
            if st.button("Process Resumes"):
                with st.spinner("Processing resumes..."):
                    st.session_state.resumes = []
                    progress_bar = st.progress(0)
                    
                    for idx, file in enumerate(uploaded_files):
                        resume_data = process_uploaded_file(file)
                        if resume_data:
                            st.session_state.resumes.append(resume_data)
                        progress_bar.progress((idx + 1) / len(uploaded_files))
                    
                    st.success(f"✅ Processed {len(st.session_state.resumes)} resumes")
    
    with col2:
        st.header("2️⃣ Job Description")
        
        # Demo job descriptions
        demo_jobs = {
            "Custom": "",
            "Software Engineer": """
We are seeking a Senior Software Engineer with 5+ years of experience in Python and machine learning.
Required skills: Python, TensorFlow, scikit-learn, REST APIs, Docker.
Experience with NLP and computer vision is a plus.
Bachelor's degree in Computer Science or related field required.
            """,
            "Data Scientist": """
Looking for a Data Scientist with strong statistical background and 3+ years of experience.
Required: Python, R, SQL, statistical modeling, machine learning.
Preferred: Deep learning, big data tools (Spark, Hadoop), cloud platforms (AWS/GCP).
Master's degree preferred.
            """,
            "Frontend Developer": """
Frontend Developer needed with expertise in modern web technologies.
Required: JavaScript, React, TypeScript, HTML/CSS, responsive design.
Experience: 3+ years in web development.
Nice to have: Next.js, GraphQL, UI/UX design skills.
            """
        }
        
        selected_demo = st.selectbox("Select Demo Job", list(demo_jobs.keys()))
        
        job_description = st.text_area(
            "Enter job description",
            value=demo_jobs[selected_demo],
            height=200
        )
    
    # Ranking section
    if st.session_state.resumes and job_description:
        st.divider()
        st.header("3️⃣ Candidate Rankings")
        
        if st.button("🚀 Rank Candidates", type="primary"):
            with st.spinner("Ranking candidates..."):
                # Rank candidates
                ranked_results = st.session_state.ranker.rank_candidates(
                    resumes=st.session_state.resumes,
                    job_description=job_description,
                    embedding_manager=st.session_state.embedding_manager
                )
                st.session_state.ranked_candidates = ranked_results
        
        # Display rankings
        if st.session_state.ranked_candidates:
            st.subheader(f"Top {len(st.session_state.ranked_candidates)} Candidates")
            
            for idx, candidate in enumerate(st.session_state.ranked_candidates):
                with st.expander(
                    f"#{idx + 1} - {candidate['filename']} - Score: {candidate['total_score']:.1%}",
                    expanded=(idx < 3)
                ):
                    col_a, col_b = st.columns([1, 2])
                    
                    with col_a:
                        st.markdown(render_score_badge(candidate['total_score']), unsafe_allow_html=True)
                        
                        if st.button(f"💬 Ask About Candidate", key=f"chat_{idx}"):
                            st.session_state.selected_candidate = candidate
                    
                    with col_b:
                        st.write("**Score Breakdown:**")
                        st.write(f"- Keyword Match: {candidate['keyword_score']:.1%}")
                        st.write(f"- Semantic Similarity: {candidate['semantic_score']:.1%}")
                        st.write(f"- Experience Match: {candidate['experience_score']:.1%}")
                        st.write(f"- Skills Bonus: {candidate['skills_score']:.1%}")
                        
                        # Quick explanation
                        explanation = candidate.get('explanation', {})
                        if explanation:
                            st.write("**Key Strengths:**")
                            for strength in explanation.get('strengths', [])[:3]:
                                st.write(f"✓ {strength}")
    
    # Chatbot section
    if st.session_state.ranked_candidates:
        st.divider()
        st.header("4️⃣ Interactive Chatbot")
        
        col_chat1, col_chat2 = st.columns([1, 1])
        
        with col_chat1:
            if st.session_state.selected_candidate:
                st.info(f"💬 Chatting about: **{st.session_state.selected_candidate['filename']}**")
                
                # Chat history
                chat_container = st.container()
                with chat_container:
                    for message in st.session_state.chat_history:
                        if message['role'] == 'user':
                            st.markdown(f"**You:** {message['content']}")
                        else:
                            st.markdown(f"**Bot:** {message['content']}")
                
                # Chat input
                user_question = st.text_input(
                    "Ask a question about this candidate:",
                    placeholder="e.g., Why did this candidate rank higher?",
                    key="chat_input"
                )
                
                if st.button("Send") and user_question:
                    # Add user message
                    st.session_state.chat_history.append({
                        'role': 'user',
                        'content': user_question
                    })
                    
                    # Generate response
                    with st.spinner("Thinking..."):
                        response = st.session_state.explainer.answer_question(
                            question=user_question,
                            candidate=st.session_state.selected_candidate,
                            job_description=job_description,
                            all_candidates=st.session_state.ranked_candidates
                        )
                    
                    # Add bot response
                    st.session_state.chat_history.append({
                        'role': 'assistant',
                        'content': response
                    })
                    
                    st.rerun()
                
                if st.button("Clear Chat"):
                    st.session_state.chat_history = []
                    st.session_state.selected_candidate = None
                    st.rerun()
            
            else:
                st.info("👆 Click 'Ask About Candidate' button on any candidate to start chatting")
        
        with col_chat2:
            st.subheader("Quick Questions")
            st.write("Try asking:")
            quick_questions = [
                "Why did this candidate rank higher?",
                "What skills does this candidate lack?",
                "How many years of experience do they have?",
                "What are their strongest qualifications?",
                "How does this compare to the job requirements?"
            ]
            
            for q in quick_questions:
                if st.button(q, key=f"quick_{q}"):
                    if st.session_state.selected_candidate:
                        st.session_state.chat_history.append({'role': 'user', 'content': q})
                        response = st.session_state.explainer.answer_question(
                            question=q,
                            candidate=st.session_state.selected_candidate,
                            job_description=job_description,
                            all_candidates=st.session_state.ranked_candidates
                        )
                        st.session_state.chat_history.append({'role': 'assistant', 'content': response})
                        st.rerun()
    
    # Footer
    st.divider()
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>AI Resume Screening System | Built with Streamlit, spaCy, and Sentence Transformers</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()