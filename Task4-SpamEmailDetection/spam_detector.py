"""Email Spam Detection Module"""
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import warnings
warnings.filterwarnings('ignore')

class EmailSpamDetector:
    """Machine Learning-based Email Spam Detection System"""
    
    def __init__(self, model_type='naive_bayes'):
        self.model_type = model_type
        self.vectorizer = TfidfVectorizer(max_features=5000, stop_words='english', ngram_range=(1, 2))
        self.model = self._initialize_model()
        
    def _initialize_model(self):
        if self.model_type == 'naive_bayes':
            return MultinomialNB(alpha=1.0)
        elif self.model_type == 'logistic_regression':
            return LogisticRegression(max_iter=1000, random_state=42)
        elif self.model_type == 'random_forest':
            return RandomForestClassifier(n_estimators=100, random_state=42)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def train(self, X, y):
        """Train the spam detection model"""
        X_transformed = self.vectorizer.fit_transform(X)
        self.model.fit(X_transformed, y)
        
    def predict(self, emails):
        """Predict spam/ham for given emails"""
        if isinstance(emails, str):
            emails = [emails]
        X_transformed = self.vectorizer.transform(emails)
        predictions = self.model.predict(X_transformed)
        return predictions
    
    def evaluate(self, X_test, y_test):
        """Evaluate model performance"""
        X_transformed = self.vectorizer.transform(X_test)
        predictions = self.model.predict(X_transformed)
        
        metrics = {
            'accuracy': accuracy_score(y_test, predictions),
            'precision': precision_score(y_test, predictions),
            'recall': recall_score(y_test, predictions),
            'f1': f1_score(y_test, predictions)
        }
        return metrics
