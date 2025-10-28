"""
Synthetic Data Generator

Generates synthetic resumes and job descriptions for training and demo.
"""

import random
from pathlib import Path
from typing import List, Dict
import json
from datetime import datetime, timedelta


# Sample data pools
FIRST_NAMES = ["John", "Jane", "Michael", "Sarah", "David", "Emily", "Robert", "Lisa", 
               "James", "Mary", "William", "Patricia", "Richard", "Jennifer", "Thomas", "Linda"]

LAST_NAMES = ["Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller", "Davis",
              "Rodriguez", "Martinez", "Hernandez", "Lopez", "Wilson", "Anderson", "Taylor"]

SKILLS = {
    'software': ['Python', 'Java', 'JavaScript', 'C++', 'React', 'Node.js', 'Django', 
                 'Flask', 'SQL', 'MongoDB', 'AWS', 'Docker', 'Kubernetes', 'Git'],
    'data_science': ['Python', 'R', 'TensorFlow', 'PyTorch', 'scikit-learn', 'Pandas', 
                     'NumPy', 'SQL', 'Tableau', 'Spark', 'Machine Learning', 'Deep Learning'],
    'frontend': ['JavaScript', 'React', 'Angular', 'Vue.js', 'HTML', 'CSS', 'TypeScript',
                 'Webpack', 'Redux', 'Material-UI'],
    'devops': ['Docker', 'Kubernetes', 'Jenkins', 'AWS', 'Azure', 'Terraform', 'Ansible',
               'Linux', 'Bash', 'Python']
}

COMPANIES = ["TechCorp", "DataSystems Inc", "CloudSolutions", "AI Innovations", 
             "WebDynamics", "CodeCraft", "DigitalWorks", "SmartTech", "InfoSystems"]

UNIVERSITIES = ["MIT", "Stanford University", "UC Berkeley", "Carnegie Mellon", 
                "Georgia Tech", "University of Washington", "Cornell University"]

DEGREES = {
    'bachelors': ["Bachelor of Science in Computer Science", "BS in Software Engineering",
                  "Bachelor of Engineering", "BS in Information Technology"],
    'masters': ["Master of Science in Computer Science", "MS in Data Science", 
                "Master of Engineering", "MS in Artificial Intelligence"],
    'phd': ["PhD in Computer Science", "PhD in Machine Learning", "PhD in Artificial Intelligence"]
}

JOB_TITLES = {
    'software': ["Software Engineer", "Senior Software Developer", "Backend Developer",
                 "Full Stack Engineer", "Software Architect"],
    'data_science': ["Data Scientist", "Machine Learning Engineer", "Data Analyst",
                     "AI Research Scientist", "Senior Data Scientist"],
    'frontend': ["Frontend Developer", "UI/UX Engineer", "Web Developer",
                 "React Developer", "Frontend Architect"],
    'devops': ["DevOps Engineer", "Site Reliability Engineer", "Cloud Engineer",
               "Infrastructure Engineer", "Platform Engineer"]
}


def generate_resume(
    category: str = 'software',
    experience_years: int = None,
    education_level: str = 'bachelors'
) -> Dict[str, str]:
    """
    Generate a synthetic resume.
    
    Args:
        category: Job category
        experience_years: Years of experience (random if None)
        education_level: Education level
        
    Returns:
        Dictionary with resume data
    """
    if experience_years is None:
        experience_years = random.randint(1, 15)
    
    name = f"{random.choice(FIRST_NAMES)} {random.choice(LAST_NAMES)}"
    email = f"{name.lower().replace(' ', '.')}@email.com"
    phone = f"+1-{random.randint(200,999)}-{random.randint(100,999)}-{random.randint(1000,9999)}"
    
    # Skills
    skill_pool = SKILLS.get(category, SKILLS['software'])
    num_skills = random.randint(5, min(len(skill_pool), 12))
    skills = random.sample(skill_pool, num_skills)
    
    # Education
    university = random.choice(UNIVERSITIES)
    degree = random.choice(DEGREES[education_level])
    grad_year = datetime.now().year - experience_years - random.randint(0, 4)
    
    # Work experience
    experiences = []
    years_covered = 0
    job_titles = JOB_TITLES.get(category, JOB_TITLES['software'])
    
    while years_covered < experience_years:
        duration = random.randint(2, 4)
        years_covered += duration
        
        company = random.choice(COMPANIES)
        title = random.choice(job_titles)
        start_year = datetime.now().year - years_covered
        end_year = start_year + duration if years_covered < experience_years else datetime.now().year
        
        # Generate responsibilities
        responsibilities = generate_responsibilities(category, skills)
        
        exp = {
            'title': title,
            'company': company,
            'duration': f"{start_year} - {end_year if end_year != datetime.now().year else 'Present'}",
            'responsibilities': responsibilities
        }
        experiences.append(exp)
    
    # Build resume text
    resume_text = f"""
{name}
Email: {email} | Phone: {phone}
LinkedIn: linkedin.com/in/{name.lower().replace(' ', '-')}

PROFESSIONAL SUMMARY
{generate_summary(category, experience_years)}

SKILLS
{', '.join(skills)}

WORK EXPERIENCE

"""
    
    for exp in experiences:
        resume_text += f"{exp['title']} at {exp['company']}\n"
        resume_text += f"{exp['duration']}\n"
        for resp in exp['responsibilities']:
            resume_text += f"• {resp}\n"
        resume_text += "\n"
    
    resume_text += f"""
EDUCATION

{degree}
{university}, {grad_year}

CERTIFICATIONS
"""
    
    # Add random certifications
    certifications = generate_certifications(category)
    for cert in certifications:
        resume_text += f"• {cert}\n"
    
    return {
        'name': name,
        'email': email,
        'category': category,
        'experience_years': experience_years,
        'education_level': education_level,
        'text': resume_text.strip()
    }


def generate_summary(category: str, years: int) -> str:
    """Generate professional summary."""
    templates = [
        f"Experienced {category} professional with {years}+ years of expertise in developing robust software solutions.",
        f"Results-driven {category} specialist with {years} years of experience in building scalable applications.",
        f"Accomplished {category} engineer with {years}+ years of hands-on experience in modern technologies.",
    ]
    return random.choice(templates)


def generate_responsibilities(category: str, skills: List[str]) -> List[str]:
    """Generate job responsibilities."""
    templates = {
        'software': [
            f"Developed and maintained applications using {random.choice(skills)}",
            "Collaborated with cross-functional teams to deliver features",
            "Implemented automated testing and CI/CD pipelines",
            "Optimized application performance and reduced latency by 30%",
            "Mentored junior developers and conducted code reviews"
        ],
        'data_science': [
            f"Built machine learning models using {random.choice(skills)}",
            "Analyzed large datasets to derive actionable insights",
            "Deployed models to production with 95% accuracy",
            "Created data visualizations and dashboards for stakeholders",
            "Collaborated with engineering teams to integrate ML solutions"
        ],
        'frontend': [
            f"Developed responsive web applications using {random.choice(skills)}",
            "Implemented pixel-perfect UI designs",
            "Optimized frontend performance and accessibility",
            "Collaborated with designers and backend engineers",
            "Built reusable component libraries"
        ],
        'devops': [
            f"Managed cloud infrastructure on {random.choice(['AWS', 'Azure', 'GCP'])}",
            "Implemented CI/CD pipelines using Jenkins and Docker",
            "Automated deployment processes and reduced deployment time by 50%",
            "Monitored system performance and ensured 99.9% uptime",
            "Implemented infrastructure as code using Terraform"
        ]
    }
    
    responsibilities = templates.get(category, templates['software'])
    return random.sample(responsibilities, min(4, len(responsibilities)))


def generate_certifications(category: str) -> List[str]:
    """Generate certifications."""
    all_certs = {
        'software': ["AWS Certified Developer", "Oracle Certified Professional", 
                     "Microsoft Certified: Azure Developer"],
        'data_science': ["TensorFlow Developer Certificate", "AWS Machine Learning Specialty",
                         "Google Professional Data Engineer"],
        'frontend': ["Google Mobile Web Specialist", "W3C Certified Frontend Developer"],
        'devops': ["AWS Certified DevOps Engineer", "Kubernetes Administrator (CKA)",
                   "Docker Certified Associate"]
    }
    
    certs = all_certs.get(category, all_certs['software'])
    return random.sample(certs, min(2, len(certs)))


def generate_job_description(category: str, experience_required: int = 5) -> str:
    """Generate job description."""
    
    templates = {
        'software': f"""
Senior Software Engineer

We are seeking a talented Senior Software Engineer with {experience_required}+ years of experience 
to join our growing team.

Requirements:
- {experience_required}+ years of professional software development experience
- Strong proficiency in Python, Java, or similar languages
- Experience with modern web frameworks (Django, Flask, Spring)
- Knowledge of SQL and NoSQL databases
- Experience with cloud platforms (AWS, Azure, or GCP)
- Strong understanding of software design patterns and best practices
- Bachelor's degree in Computer Science or related field

Nice to have:
- Experience with Docker and Kubernetes
- Knowledge of microservices architecture
- Contributions to open-source projects
- Master's degree in Computer Science

Key Skills: Python, Java, React, Node.js, AWS, Docker, SQL, REST APIs
""",
        'data_science': f"""
Data Scientist

Looking for an experienced Data Scientist with {experience_required}+ years to drive 
data-driven decision making.

Requirements:
- {experience_required}+ years of experience in data science or machine learning
- Strong programming skills in Python and R
- Experience with machine learning frameworks (TensorFlow, PyTorch, scikit-learn)
- Proficiency in SQL and data manipulation (Pandas, NumPy)
- Strong statistical analysis and modeling skills
- Experience deploying ML models to production
- Master's degree in Computer Science, Statistics, or related field

Nice to have:
- PhD in relevant field
- Experience with big data tools (Spark, Hadoop)
- Knowledge of deep learning and NLP
- Publications in ML conferences

Key Skills: Python, R, TensorFlow, Machine Learning, Deep Learning, SQL, Statistics
""",
        'frontend': f"""
Frontend Developer

Join our team as a Frontend Developer with {experience_required}+ years of experience 
building modern web applications.

Requirements:
- {experience_required}+ years of frontend development experience
- Expert knowledge of JavaScript, HTML, and CSS
- Strong experience with React, Angular, or Vue.js
- Understanding of responsive design and cross-browser compatibility
- Experience with state management (Redux, MobX)
- Knowledge of build tools (Webpack, Babel)
- Bachelor's degree in Computer Science or related field

Nice to have:
- TypeScript experience
- UI/UX design skills
- Experience with Next.js or Gatsby
- Knowledge of accessibility standards

Key Skills: JavaScript, React, TypeScript, HTML, CSS, Redux, Responsive Design
""",
        'devops': f"""
DevOps Engineer

Seeking a skilled DevOps Engineer with {experience_required}+ years to manage our 
infrastructure and deployment pipelines.

Requirements:
- {experience_required}+ years of DevOps or system administration experience
- Strong experience with AWS, Azure, or GCP
- Proficiency in Docker and Kubernetes
- Experience with CI/CD tools (Jenkins, GitLab CI, CircleCI)
- Knowledge of infrastructure as code (Terraform, Ansible)
- Strong scripting skills (Bash, Python)
- Bachelor's degree in Computer Science or related field

Nice to have:
- Kubernetes certifications (CKA, CKAD)
- Experience with monitoring tools (Prometheus, Grafana)
- Knowledge of security best practices
- Experience with service mesh technologies

Key Skills: AWS, Docker, Kubernetes, Jenkins, Terraform, Python, Linux, CI/CD
"""
    }
    
    return templates.get(category, templates['software']).strip()


def generate_dataset(
    output_dir: str = "./data/demo",
    num_resumes: int = 20,
    num_jobs: int = 4
):
    """
    Generate complete synthetic dataset.
    
    Args:
        output_dir: Output directory
        num_resumes: Number of resumes to generate
        num_jobs: Number of job descriptions to generate
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Create subdirectories
    resumes_dir = output_path / "resumes"
    jobs_dir = output_path / "jobs"
    resumes_dir.mkdir(exist_ok=True)
    jobs_dir.mkdir(exist_ok=True)
    
    print(f"Generating {num_resumes} synthetic resumes...")
    
    # Generate resumes
    resumes_data = []
    categories = list(SKILLS.keys())
    
    for i in range(num_resumes):
        category = random.choice(categories)
        education = random.choice(['bachelors', 'masters', 'phd'])
        experience = random.randint(1, 15)
        
        resume = generate_resume(category, experience, education)
        
        # Save as text file
        filename = f"resume_{i+1:03d}_{resume['name'].replace(' ', '_')}.txt"
        filepath = resumes_dir / filename
        
        with open(filepath, 'w') as f:
            f.write(resume['text'])
        
        resumes_data.append({
            'filename': filename,
            'name': resume['name'],
            'category': category,
            'experience_years': experience,
            'education_level': education
        })
        
        print(f"  ✓ Generated: {filename}")
    
    # Save resume metadata
    with open(output_path / "resumes_metadata.json", 'w') as f:
        json.dump(resumes_data, f, indent=2)
    
    print(f"\nGenerating {num_jobs} job descriptions...")
    
    # Generate job descriptions
    jobs_data = []
    for i, category in enumerate(categories[:num_jobs]):
        experience_required = random.choice([3, 5, 7])
        job_desc = generate_job_description(category, experience_required)
        
        filename = f"job_{i+1}_{category}.txt"
        filepath = jobs_dir / filename
        
        with open(filepath, 'w') as f:
            f.write(job_desc)
        
        jobs_data.append({
            'filename': filename,
            'category': category,
            'experience_required': experience_required
        })
        
        print(f"  ✓ Generated: {filename}")
    
    # Save job metadata
    with open(output_path / "jobs_metadata.json", 'w') as f:
        json.dump(jobs_data, f, indent=2)
    
    print(f"\n✅ Dataset generation complete!")
    print(f"   Resumes: {resumes_dir}")
    print(f"   Jobs: {jobs_dir}")
    print(f"   Total files: {num_resumes + num_jobs}")


if __name__ == "__main__":
    generate_dataset(num_resumes=20, num_jobs=4)