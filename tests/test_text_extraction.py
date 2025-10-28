"""
Tests for text extraction module.
"""

import pytest
from pathlib import Path
import tempfile
import os

from src.text_extraction import (
    extract_text_from_pdf,
    extract_text_from_docx,
    identify_sections,
    extract_contact_info
)


def test_extract_text_from_nonexistent_file():
    """Test extraction from non-existent file raises error."""
    with pytest.raises(FileNotFoundError):
        extract_text_from_pdf("nonexistent.pdf")


def test_identify_sections():
    """Test section identification."""
    sample_text = """
John Doe
john@email.com

Summary
Experienced software engineer

Experience
Software Engineer at TechCorp
2020-2023

Skills
Python, Java, SQL

Education
BS Computer Science, MIT
"""
    
    sections = identify_sections(sample_text)
    assert 'summary' in sections
    assert 'experience' in sections
    assert 'skills' in sections
    assert 'education' in sections


def test_extract_contact_info():
    """Test contact information extraction."""
    sample_text = """
John Doe
Email: john.doe@email.com
Phone: +1-555-123-4567
LinkedIn: linkedin.com/in/john-doe
"""
    
    contact = extract_contact_info(sample_text)
    assert 'email' in contact
    assert contact['email'] == 'john.doe@email.com'
    assert 'phone' in contact
    assert 'linkedin' in contact


def test_extract_contact_info_missing():
    """Test contact extraction with missing info."""
    sample_text = "Just some text without contact information"
    contact = extract_contact_info(sample_text)
    assert isinstance(contact, dict)