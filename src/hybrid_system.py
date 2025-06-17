"""
Hybrid Recommendation System Implementation

This module combines ALS collaborative filtering and Two-Tower content-based 
filtering models using adaptive weighted fusion based on F1 scores.
"""

import numpy as np
import pandas as pd
import os
import pickle
from sklearn.preprocessing import MinMaxScaler
from .als_model import ALSModel
from .two_tower_model import TwoTowerModel
from .evaluation import compute_f1_score
import warnings

warnings.filterwarnings('ignore')


class HybridRecommendationSystem:
    """
    Hybrid Recommender System combining ALS and Two-Tower models
    
    The system uses pre-trained models and adaptive weighting based on F1 scores
    to combine predictions from both models.
    """
    
    def __init__(self):
        # Model instances
        self.als_model = None
        self.twotower_model = None
        
        # Scalers for normalization
        self.als_scaler = MinMaxScaler()
        self.twotower_scaler = MinMaxScaler()
        
        # Model performance metrics
        self.als_f1_score = 0.0
        self.twotower_f1_score = 0.0
        
        # System status
        self.models_loaded = False

    def load_models(self, als_model_path, twotower_model_path):
        """
        Load pre-trained models from disk
        
        Parameters:
        -----------
        als_model_path : str
            Path to saved ALS model directory
        twotower_model_path : str
            Path to saved Two-Tower model file
        """
        try:
            print("=== Loading Pre-trained Models ===")
            
            # Load ALS model
            print("\n1. Loading ALS Model...")
            self.als_model = ALSModel()
            self.als_model.load_model(als_model_path)
            
            # Load Two-Tower model
            print("\n2. Loading Two-Tower Model...")
            self.twotower_model = TwoTowerModel(0, 0)  # Dummy initialization
            self.twotower_model.load_model(twotower_model_path)
            
            self.models_loaded = True
            print("\n=== Models loaded successfully ===")
            return True
            
        except Exception as e:
            print(f"Error loading models: {str(e)}")
            return False

    def evaluate_individual_models(self, test_user_id, actual_ratings, all_items, k=10):
        """
        Evaluate individual model performance to determine F1 scores
        
        Parameters:
        -----------
        test_user_id : int
            User ID for evaluation
        actual_ratings : dict
            Dictionary mapping itemId to actual rating
        all_items : list
            List of all available item IDs
        k : int
            Top-k for evaluation metrics
            
        Returns:
        --------
        tuple : (als_f1, twotower_f1) F1 scores
        """
        if not self.models_loaded:
            raise ValueError("Models not loaded yet. Call load_models() first.")
            
        try:
            # Get predictions from both models
            als_predictions = self.als_model.predict_for_user(test_user_id, all_items)
            twotower_predictions = self.twotower_model.predict_for_user(test_user_id, all_items)
            
            # Convert to dictionaries for processing
            als_pred_dict = dict(als_predictions)
            twotower_pred_dict = dict(twotower_predictions)
            
            # Calculate F1 scores
            self.als_f1_score = compute_f1_score(actual_ratings, als_pred_dict, k)
            self.twotower_f1_score = compute_f1_score(actual_ratings, twotower_pred_dict, k)
            
            print(f"Model F1 Scores - ALS: {self.als_f1_score:.4f}, "
                  f"Two-Tower: {self.twotower_f1_score:.4f}")
            return self.als_f1_score, self.twotower_f1_score
            
        except Exception as e:
            print(f"Error evaluating models: {str(e)}")
            return 0.0, 0.0

    def adaptive_fusion(self, als_predictions, twotower_predictions):
        """
        Combine predictions from both models using adaptive weighting
        
        Parameters:
        -----------
        als_predictions : list
            List of (itemId, score) tuples from ALS model
        twotower_predictions : list
            List of (itemId, score) tuples from Two-Tower model
            
        Returns:
        --------
        list : List of (itemId, combined_score) tuples
        """
        try:
            # Convert to dictionaries
            als_dict = dict(als_predictions)
            twotower_dict = dict(twotower_predictions)
            
            # Get all items
            all_items = set(als_dict.keys()).union(set(twotower_dict.keys()))
            
            # Extract scores for normalization
            als_scores = [als_dict.get(item, 0) for item in all_items]
            twotower_scores = [twotower_dict.get(item, 0) for item in all_items]
            
            # Normalize scores
            als_scores_norm = self.als_scaler.fit_transform(
                np.array(als_scores).reshape(-1, 1)
            ).flatten()
            twotower_scores_norm = self.twotower_scaler.fit_transform(
                np.array(twotower_scores).reshape(-1, 1)
            ).flatten()
            
            # Determine weights based on F1 scores
            if self.als_f1_score > self.twotower_f1_score:
                als_weight, twotower_weight = 0.8, 0.2
            else:
                als_weight, twotower_weight = 0.2, 0.8
            
            # Combine predictions
            combined = []
            for i, item in enumerate(all_items):
                score = (als_weight * als_scores_norm[i] +
                        twotower_weight * twotower_scores_norm[i])
                combined.append((item, score))
            
            return combined
            
        except Exception as e:
            print(f"Error in adaptive fusion: {str(e)}")
            return []

    def get_hybrid_recommendations(self, user_id, all_items, actual_ratings=None, top_k=5):
        """
        Generate hybrid recommendations for a user
        
        Parameters:
        -----------
        user_id : int
            User ID for recommendations
        all_items : list
            List of all available item IDs
        actual_ratings : dict, optional
            Actual ratings for F1 score calculation
        top_k : int
            Number of top recommendations to return
            
        Returns:
        --------
        list : List of top-K (itemId, score) tuples
        """
        if not self.models_loaded:
            raise ValueError("Models not loaded. Call load_models() first.")
            
        try:
            # Get predictions from both models
            als_predictions = self.als_model.predict_for_user(user_id, all_items)
            twotower_predictions = self.twotower_model.predict_for_user(user_id, all_items)
            
            # Update weights if actual ratings provided
            if actual_ratings:
                self.evaluate_individual_models(user_id, actual_ratings, all_items)
            
            # Combine predictions
            combined = self.adaptive_fusion(als_predictions, twotower_predictions)
            
            # Sort and return top-K
            return sorted(combined, key=lambda x: x[1], reverse=True)[:top_k]
            
        except Exception as e:
            print(f"Error generating recommendations: {str(e)}")
            return []

    def cleanup(self):
        """Clean up resources"""
        if self.als_model:
            self.als_model.stop_spark()


if __name__ == "__main__":
    # Simplified usage example
    print("Hybrid Recommendation System module loaded successfully")
    print("Example usage:")
    print("hrs = HybridRecommendationSystem()")
    print("hrs.load_models('models/als', 'models/twotower.keras')")
    print("recommendations = hrs.get_hybrid_recommendations(462, all_items)")
