"""
AI Resume Screening System - Core Package

This package contains all core modules for resume screening and ranking.
"""

__version__ = "1.0.0"
__author__ = "AI Resume Screener Team"

from . import text_extraction
from . import preprocessing
from . import embedding
from . import ranking
from . import explainer
from . import trainer

__all__ = [
    'text_extraction',
    'preprocessing',
    'embedding',
    'ranking',
    'explainer',
    'trainer'
]