"""
Trainer Module

Handles training of optional ML models for ranking.
Includes model persistence and evaluation.
"""

import logging
from typing import List, Dict, Any, Tuple, Optional
from pathlib import Path
import numpy as np
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, roc_auc_score
from xgboost import XGBClassifier

logger = logging.getLogger(__name__)


class RankingModelTrainer:
    """Trains ML models for resume ranking."""
    
    def __init__(self, model_dir: str = "./models"):
        """
        Initialize trainer.
        
        Args:
            model_dir: Directory to save trained models
        """
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.model = None
        self.model_type = None
    
    def prepare_training_data(
        self,
        ranked_results: List[Dict[str, Any]],
        threshold: float = 0.6
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare training data from ranked results.
        
        Args:
            ranked_results: List of candidates with scores
            threshold: Score threshold for positive class
            
        Returns:
            Tuple of (X, y) arrays
        """
        X = []
        y = []
        
        for candidate in ranked_results:
            # Features: individual score components
            features = [
                candidate['keyword_score'],
                candidate['semantic_score'],
                candidate['experience_score'],
                candidate['skills_score']
            ]
            
            # Label: binary based on threshold
            label = 1 if candidate['total_score'] >= threshold else 0
            
            X.append(features)
            y.append(label)
        
        return np.array(X), np.array(y)
    
    def train_logistic_regression(
        self,
        X: np.ndarray,
        y: np.ndarray,
        test_size: float = 0.2,
        random_state: int = 42
    ) -> Dict[str, Any]:
        """
        Train logistic regression model.
        
        Args:
            X: Feature matrix
            y: Labels
            test_size: Test set size
            random_state: Random seed
            
        Returns:
            Training results dictionary
        """
        logger.info("Training Logistic Regression model")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # Train model
        self.model = LogisticRegression(random_state=random_state, max_iter=1000)
        self.model.fit(X_train, y_train)
        self.model_type = 'logistic_regression'
        
        # Evaluate
        train_score = self.model.score(X_train, y_train)
        test_score = self.model.score(X_test, y_test)
        
        y_pred = self.model.predict(X_test)
        report = classification_report(y_test, y_pred, output_dict=True)
        
        # Cross-validation
        cv_scores = cross_val_score(self.model, X, y, cv=5)
        
        results = {
            'model_type': self.model_type,
            'train_accuracy': train_score,
            'test_accuracy': test_score,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'classification_report': report
        }
        
        logger.info(f"Logistic Regression - Train: {train_score:.3f}, Test: {test_score:.3f}")
        return results
    
    def train_random_forest(
        self,
        X: np.ndarray,
        y: np.ndarray,
        test_size: float = 0.2,
        random_state: int = 42
    ) -> Dict[str, Any]:
        """
        Train random forest model.
        
        Args:
            X: Feature matrix
            y: Labels
            test_size: Test set size
            random_state: Random seed
            
        Returns:
            Training results dictionary
        """
        logger.info("Training Random Forest model")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # Train model
        self.model = RandomForestClassifier(
            n_estimators=100,
            random_state=random_state,
            max_depth=10
        )
        self.model.fit(X_train, y_train)
        self.model_type = 'random_forest'
        
        # Evaluate
        train_score = self.model.score(X_train, y_train)
        test_score = self.model.score(X_test, y_test)
        
        y_pred = self.model.predict(X_test)
        report = classification_report(y_test, y_pred, output_dict=True)
        
        # Feature importance
        feature_names = ['keyword', 'semantic', 'experience', 'skills']
        feature_importance = dict(zip(feature_names, self.model.feature_importances_))
        
        results = {
            'model_type': self.model_type,
            'train_accuracy': train_score,
            'test_accuracy': test_score,
            'feature_importance': feature_importance,
            'classification_report': report
        }
        
        logger.info(f"Random Forest - Train: {train_score:.3f}, Test: {test_score:.3f}")
        return results
    
    def train_xgboost(
        self,
        X: np.ndarray,
        y: np.ndarray,
        test_size: float = 0.2,
        random_state: int = 42
    ) -> Dict[str, Any]:
        """
        Train XGBoost model.
        
        Args:
            X: Feature matrix
            y: Labels
            test_size: Test set size
            random_state: Random seed
            
        Returns:
            Training results dictionary
        """
        logger.info("Training XGBoost model")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # Train model
        self.model = XGBClassifier(
            n_estimators=100,
            random_state=random_state,
            max_depth=5,
            learning_rate=0.1
        )
        self.model.fit(X_train, y_train)
        self.model_type = 'xgboost'
        
        # Evaluate
        train_score = self.model.score(X_train, y_train)
        test_score = self.model.score(X_test, y_test)
        
        y_pred = self.model.predict(X_test)
        report = classification_report(y_test, y_pred, output_dict=True)
        
        # Feature importance
        feature_names = ['keyword', 'semantic', 'experience', 'skills']
        feature_importance = dict(zip(feature_names, self.model.feature_importances_))
        
        results = {
            'model_type': self.model_type,
            'train_accuracy': train_score,
            'test_accuracy': test_score,
            'feature_importance': feature_importance,
            'classification_report': report
        }
        
        logger.info(f"XGBoost - Train: {train_score:.3f}, Test: {test_score:.3f}")
        return results
    
    def save_model(self, filename: str = "ranking_model.pkl"):
        """Save trained model to disk."""
        if self.model is None:
            raise ValueError("No model to save. Train a model first.")
        
        filepath = self.model_dir / filename
        joblib.dump({
            'model': self.model,
            'model_type': self.model_type
        }, filepath)
        
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filename: str = "ranking_model.pkl"):
        """Load trained model from disk."""
        filepath = self.model_dir / filename
        
        if not filepath.exists():
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        data = joblib.load(filepath)
        self.model = data['model']
        self.model_type = data['model_type']
        
        logger.info(f"Model loaded from {filepath}")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions with trained model."""
        if self.model is None:
            raise ValueError("No model loaded. Train or load a model first.")
        
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get prediction probabilities."""
        if self.model is None:
            raise ValueError("No model loaded. Train or load a model first.")
        
        return self.model.predict_proba(X)