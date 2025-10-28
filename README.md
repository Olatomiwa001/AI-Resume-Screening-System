# AI Resume Screening System

A complete AI-powered resume screening and ranking system with explainable results and an interactive chatbot interface.

## Overview

This system accepts resume uploads (PDF/DOCX), compares them against job descriptions, and ranks candidates based on:
- Keyword matching (TF-IDF)
- Semantic similarity (Sentence Transformers)
- Experience matching (years of relevant experience)
- Skills and certification bonuses

The system provides explainable rankings and includes an integrated chatbot to answer questions about candidate rankings.

## Architecture
```
┌─────────────────┐
│ Resume Upload   │
│ (PDF/DOCX)      │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Text Extraction │ (pdfminer.six, python-docx)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Preprocessing   │ (spaCy NER, cleaning)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Feature Extract │
│ - TF-IDF        │
│ - Embeddings    │ (sentence-transformers)
│ - Experience    │
│ - Skills        │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Ranking Engine  │ (Weighted scoring)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Explainer       │ (Rule-based + Optional LLM)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Streamlit UI    │
│ + Chatbot       │
└─────────────────┘
```

## Features

- **Multi-format Support**: PDF and DOCX resume parsing
- **Intelligent Ranking**: Composite scoring from multiple signals
- **Explainable AI**: Clear breakdown of ranking factors
- **Interactive Chatbot**: Ask questions about candidate rankings
- **Configurable Weights**: Adjust scoring components
- **Caching**: Efficient embedding caching for demos
- **Extensible**: Easy to swap models and add features

## Quick Start

### 1. Setup Environment
```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download spaCy model
python -m spacy download en_core_web_sm
```

### 2. Generate Demo Data
```bash
python scripts/generate_demo_data.py
```

This creates synthetic resumes and job descriptions in `data/demo/`.

### 3. (Optional) Train Ranking Model
```bash
# Open and run the training notebook
jupyter notebook notebooks/training_demo.ipynb
```

### 4. Launch Streamlit App
```bash
streamlit run streamlit_app.py
```

The app will open at `http://localhost:8501`.

## Usage

### Web Interface

1. **Upload Resumes**: Drop PDF or DOCX files
2. **Enter Job Description**: Paste or select a demo job description
3. **View Rankings**: See scored and ranked candidates
4. **Chat**: Click a candidate and ask questions like:
   - "Why did this candidate rank higher?"
   - "What skills does this candidate lack?"
   - "How many years of Python experience do they have?"

### API Usage (Optional)
```bash
# Start FastAPI server
uvicorn src.api:app --reload --port 8000

# Score a resume
curl -X POST http://localhost:8000/score \
  -H "Content-Type: application/json" \
  -d '{
    "resume_text": "...",
    "job_description": "..."
  }'
```

### Configuration

Create a `.env` file for optional features:
```bash
# Optional: Use OpenAI for richer explanations
OPENAI_API_KEY=your_key_here

# Optional: Use Anthropic Claude for explanations
ANTHROPIC_API_KEY=your_key_here

# Scoring weights (defaults shown)
WEIGHT_KEYWORD=0.3
WEIGHT_SEMANTIC=0.4
WEIGHT_EXPERIENCE=0.2
WEIGHT_SKILLS=0.1
```

## Docker Deployment
```bash
# Build image
docker build -t resume-screener .

# Run container
docker run -p 8501:8501 resume-screener

# With API keys
docker run -p 8501:8501 \
  -e OPENAI_API_KEY=your_key \
  resume-screener
```

## Testing
```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html
```

## Project Structure

- `src/`: Core modules
  - `text_extraction.py`: PDF/DOCX parsing
  - `preprocessing.py`: Text cleaning and NER
  - `embedding.py`: Sentence transformer embeddings
  - `ranking.py`: Scoring and ranking logic
  - `explainer.py`: Generate natural language explanations
  - `trainer.py`: Train optional ML models
  - `api.py`: FastAPI wrapper
- `streamlit_app.py`: Main web interface
- `notebooks/`: Training demonstrations
- `data/`: Synthetic data generation
- `tests/`: Unit tests
- `scripts/`: Helper scripts

## Extending the System

### Use a Larger Model

Replace the embedding model in `src/embedding.py`:
```python
# Current: all-MiniLM-L6-v2 (lightweight)
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# Option 1: Better accuracy
model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

# Option 2: Multilingual
model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-mpnet-base-v2')
```

### Add Custom Features

Extend `ranking.py` to add new scoring components:
```python
def compute_custom_score(resume_text, job_description):
    # Add your logic here
    return score
```

### Production Deployment

For production use, consider:

1. **Database**: Store resumes and scores in PostgreSQL
2. **Queue**: Use Celery for async processing
3. **Auth**: Add user authentication (OAuth, JWT)
4. **Monitoring**: Add logging (ELK stack, DataDog)
5. **Scale**: Deploy on AWS/GCP with load balancing
6. **Storage**: Use S3 for resume storage
7. **API Gateway**: Add rate limiting and caching

### Integration Examples

**ATS Integration** (e.g., Greenhouse, Lever):
```python
# Webhook receiver
@app.post("/webhook/new_application")
async def handle_new_application(data: dict):
    resume_url = data['resume_url']
    # Download, score, and post back
```

**LinkedIn Integration**:
```python
# Use LinkedIn API to fetch profiles
# Parse and score like resumes
```

## Performance

- **Single Resume**: ~200ms (extraction + scoring)
- **Batch 100 Resumes**: ~15s (with caching)
- **Memory**: ~500MB (with embeddings cached)

## Limitations

- This is a demo system for local evaluation
- No production authentication or user management
- Synthetic training data only
- Embedding cache grows with unique resumes
- Chatbot uses rule-based logic (upgrade to LLM recommended)

## License

MIT License - see LICENSE file

## Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Add tests for new features
4. Submit a pull request

## Support

For issues or questions, please open a GitHub issue.

## Acknowledgments

- Sentence Transformers: https://www.sbert.net/
- spaCy: https://spacy.io/
- Streamlit: https://streamlit.io/"# AI-Resume-Screening-System" 
