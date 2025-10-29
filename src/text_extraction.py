"""
Text Extraction Module

Handles extraction of text content from PDF and DOCX files.
Attempts to preserve section structure when possible.
"""

import logging
from pathlib import Path
from typing import Optional, Dict
import re

# Try different pdfminer imports for compatibility
try:
    from pdfminer.high_level import extract_text as pdf_extract
    from pdfminer.layout import LAParams
    USE_PDFMINER = True
except ImportError:
    # Fallback if pdfminer.high_level not available
    try:
        from pdfminer.pdfpage import PDFPage
        from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
        from pdfminer.converter import TextConverter
        from pdfminer.layout import LAParams
        from io import StringIO
        USE_PDFMINER = True
    except ImportError:
        USE_PDFMINER = False
        print("Warning: pdfminer not available")

import docx2txt

logger = logging.getLogger(__name__)


def extract_text_from_pdf(file_path: str) -> str:
    """
    Extract text from a PDF file.
    
    Args:
        file_path: Path to PDF file
        
    Returns:
        Extracted text content
        
    Raises:
        FileNotFoundError: If file doesn't exist
        Exception: If extraction fails
    """
    try:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"PDF file not found: {file_path}")
        
        logger.info(f"Extracting text from PDF: {file_path}")
        
        if not USE_PDFMINER:
            raise ImportError("pdfminer.six not available")
        
        # Try using high_level extract_text first
        try:
            laparams = LAParams(
                line_margin=0.5,
                word_margin=0.1,
                char_margin=2.0,
                detect_vertical=True
            )
            text = pdf_extract(file_path, laparams=laparams)
        except (NameError, TypeError):
            # Fallback to manual extraction
            output = StringIO()
            manager = PDFResourceManager()
            converter = TextConverter(manager, output, laparams=LAParams())
            interpreter = PDFPageInterpreter(manager, converter)
            
            with open(file_path, 'rb') as f:
                for page in PDFPage.get_pages(f, check_extractable=True):
                    interpreter.process_page(page)
            
            converter.close()
            text = output.getvalue()
            output.close()
        
        if not text or len(text.strip()) == 0:
            logger.warning(f"No text extracted from PDF: {file_path}")
            return ""
        
        # Basic cleanup
        text = _clean_extracted_text(text)
        
        logger.info(f"Successfully extracted {len(text)} characters from PDF")
        return text
    
    except Exception as e:
        logger.error(f"Failed to extract text from PDF {file_path}: {e}")
        raise


def extract_text_from_docx(file_path: str) -> str:
    """
    Extract text from a DOCX file.
    
    Args:
        file_path: Path to DOCX file
        
    Returns:
        Extracted text content
        
    Raises:
        FileNotFoundError: If file doesn't exist
        Exception: If extraction fails
    """
    try:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"DOCX file not found: {file_path}")
        
        logger.info(f"Extracting text from DOCX: {file_path}")
        
        # Extract text using docx2txt (preserves more structure than python-docx)
        text = docx2txt.process(file_path)
        
        if not text or len(text.strip()) == 0:
            logger.warning(f"No text extracted from DOCX: {file_path}")
            return ""
        
        # Basic cleanup
        text = _clean_extracted_text(text)
        
        logger.info(f"Successfully extracted {len(text)} characters from DOCX")
        return text
    
    except Exception as e:
        logger.error(f"Failed to extract text from DOCX {file_path}: {e}")
        raise


def _clean_extracted_text(text: str) -> str:
    """
    Clean extracted text while preserving structure.
    
    Args:
        text: Raw extracted text
        
    Returns:
        Cleaned text
    """
    # Remove excessive whitespace while preserving line breaks
    text = re.sub(r' +', ' ', text)  # Multiple spaces to single
    text = re.sub(r'\n\n+', '\n\n', text)  # Multiple newlines to double
    text = re.sub(r'\t+', ' ', text)  # Tabs to space
    
    # Remove form feed and other control characters
    text = text.replace('\f', '\n')
    text = text.replace('\r', '')
    
    # Remove leading/trailing whitespace per line
    lines = [line.strip() for line in text.split('\n')]
    text = '\n'.join(lines)
    
    return text.strip()


def identify_sections(text: str) -> Dict[str, str]:
    """
    Attempt to identify common resume sections.
    
    Args:
        text: Resume text
        
    Returns:
        Dictionary mapping section names to content
    """
    sections: Dict[str, str] = {}
    
    # Common section headers
    section_patterns = {
        'summary': r'(?i)(summary|profile|objective|about)',
        'experience': r'(?i)(experience|work history|employment|professional experience)',
        'education': r'(?i)(education|academic|qualifications)',
        'skills': r'(?i)(skills|technical skills|competencies|expertise)',
        'certifications': r'(?i)(certifications?|licenses?|credentials)',
        'projects': r'(?i)(projects|portfolio)',
        'awards': r'(?i)(awards?|honors?|achievements?)',
    }
    
    lines = text.split('\n')
    current_section = 'header'
    section_lines = {current_section: []}
    
    for line in lines:
        # Check if line is a section header
        is_section_header = False
        for section_name, pattern in section_patterns.items():
            if re.match(pattern, line.strip()) and len(line.strip()) < 50:
                current_section = section_name
                section_lines[current_section] = []
                is_section_header = True
                break
        
        if not is_section_header and line.strip():
            section_lines[current_section].append(line)
    
    # Join lines in each section
    for section, lines_list in section_lines.items():
        sections[section] = '\n'.join(lines_list)
    
    return sections


def extract_contact_info(text: str) -> Dict[str, str]:
    """
    Extract basic contact information from resume text.
    
    Args:
        text: Resume text
        
    Returns:
        Dictionary with extracted contact info
    """
    contact_info: Dict[str, str] = {}
    
    # Email pattern
    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    emails = re.findall(email_pattern, text)
    if emails:
        contact_info['email'] = emails[0]
    
    # Phone pattern (US format, can be extended)
    phone_pattern = r'(\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}'
    phones = re.findall(phone_pattern, text)
    if phones:
        contact_info['phone'] = phones[0] if isinstance(phones[0], str) else ''.join(phones[0])
    
    # LinkedIn pattern
    linkedin_pattern = r'linkedin\.com/in/[\w-]+'
    linkedin = re.findall(linkedin_pattern, text.lower())
    if linkedin:
        contact_info['linkedin'] = linkedin[0]
    
    return contact_info