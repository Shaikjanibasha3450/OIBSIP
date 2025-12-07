"""Data Loader Module
This module provides utilities for loading and preprocessing email spam detection data.
It includes text cleaning, data preparation, and utility functions for the spam detection project.
"""
import re
import pandas as pd
import numpy as np
from typing import Tuple, Optional

class DataLoader:
    """Utility class for loading and preprocessing email data."""
    
    @staticmethod
    def clean_text(text: str) -> str:
        """
        Clean and preprocess text data by removing special characters and normalizing whitespace.
        
        Args:
            text (str): Raw text to clean
        
        Returns:
            str: Cleaned and normalized text
        """
        if not isinstance(text, str):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+', '', text)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove special characters and digits
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text
    
    @staticmethod
    def load_and_prepare(
        filepath: str,
        text_column: str = 'Message',
        label_column: str = 'Category',
        clean: bool = True,
        shuffle: bool = True,
        random_state: Optional[int] = None
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Load email data from CSV and prepare for model training.
        
        Args:
            filepath (str): Path to the data CSV file
            text_column (str): Name of the text column
            label_column (str): Name of the label column
            clean (bool): Whether to clean text data
            shuffle (bool): Whether to shuffle the data
            random_state (int): Random seed for reproducibility
        
        Returns:
            Tuple[pd.DataFrame, pd.Series]: Preprocessed features and labels
        """
        # Load data
        df = pd.read_csv(filepath, encoding='latin-1')
        
        # Select relevant columns
        df = df[[text_column, label_column]]
        
        # Remove missing values
        df = df.dropna()
        
        # Clean text if requested
        if clean:
            df[text_column] = df[text_column].apply(DataLoader.clean_text)
        
        # Separate features and labels
        X = df[text_column]
        y = df[label_column]
        
        # Shuffle if requested
        if shuffle:
            indices = np.arange(len(df))
            if random_state is not None:
                np.random.seed(random_state)
            np.random.shuffle(indices)
            X = X.iloc[indices].reset_index(drop=True)
            y = y.iloc[indices].reset_index(drop=True)
        
        return X, y
    
    @staticmethod
    def get_data_stats(X: pd.Series, y: pd.Series) -> dict:
        """
        Generate statistics about the loaded data.
        
        Args:
            X (pd.Series): Feature data
            y (pd.Series): Label data
        
        Returns:
            dict: Dictionary containing data statistics
        """
        stats = {
            'total_samples': len(X),
            'feature_shape': X.shape,
            'label_shape': y.shape,
            'unique_labels': y.unique().tolist(),
            'label_counts': y.value_counts().to_dict(),
            'avg_text_length': X.str.len().mean(),
            'max_text_length': X.str.len().max(),
            'min_text_length': X.str.len().min()
        }
        return stats
