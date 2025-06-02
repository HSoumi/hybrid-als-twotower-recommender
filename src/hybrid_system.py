"""
Hybrid Recommendation System Implementation

This module combines ALS collaborative filtering and Two-Tower content-based 
filtering models using adaptive weighted fusion based on F1 scores.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from .als_model import ALSModel
from .two_tower_model import TwoTowerModel
from .evaluation import compute_f1_score
import warnings

warnings.filterwarnings('ignore')


class HybridRecommendationSystem:
    """
    Hybrid Recommender System combining ALS and Two-Tower models
    
    The system uses an adaptive weighting strategy based on individual model
    F1 scores to combine predictions from both models.
    
    Parameters:
    -----------
    als_params : dict
        Parameters for ALS model initialization
    twotower_params : dict
        Parameters for Two-Tower model initialization
    """
    
    def __init__(self, als_params=None, twotower_params=None):
        # Default parameters
        self.als_params = als_params or {
            'rank': 10, 'max_iter': 10, 'reg_param': 0.1
        }
        self.twotower_params = twotower_params or {
            'embedding_size': 50, 'learning_rate': 0.001
        }
        
        # Model instances
        self.als_model = None
        self.twotower_model = None
        
        # Scalers for normalization
        self.als_scaler = MinMaxScaler()
        self.twotower_scaler = MinMaxScaler()
        
        # Model performance metrics
        self.als_f1_score = 0.0
        self.twotower_f1_score = 0.0
        
        # Training status
        self.is_trained = False
        
    def train_models(self, data, twotower_epochs=10, twotower_batch_size=256):
        """
        Train both ALS and Two-Tower models
        
        Parameters:
        -----------
        data : pd.DataFrame
            Training data with columns: userId, itemId, average_review_rating
        twotower_epochs : int
            Number of epochs for Two-Tower training
        twotower_batch_size : int
            Batch size for Two-Tower training
            
        Returns:
        --------
        bool : Success status of training
        """
        try:
            print("=== Training Hybrid Recommendation System ===")
            
            # Initialize models
            self.als_model = ALSModel(**self.als_params)
            
            num_users = data['userId'].nunique()
            num_items = data['itemId'].nunique()
            self.twotower_model = TwoTowerModel(
                num_users=num_users,
                num_items=num_items,
                **self.twotower_params
            )
            
            # Train ALS model
            print("\n1. Training ALS Model...")
            als_success = self.als_model.train(data)
            if not als_success:
                print("Failed to train ALS model")
                return False
            
            # Train Two-Tower model
            print("\n2. Training Two-Tower Model...")
            twotower_history = self.twotower_model.train(
                data, 
                epochs=twotower_epochs,
                batch_size=twotower_batch_size
            )
            if twotower_history is None:
                print("Failed to train Two-Tower model")
                return False
            
            self.is_trained = True
            print("\n=== Hybrid system training completed successfully ===")
            return True
            
        except Exception as e:
            print(f"Error training hybrid system: {str(e)}")
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
        if not self.is_trained:
            raise ValueError("Models not trained yet. Call train_models() first.")
        
        try:
            # Get predictions from both models
            als_predictions = self.als_model.predict_for_user(test_user_id, all_items)
            twotower_predictions = self.twotower_model.predict_for_user(test_user_id, all_items)
            
            # Convert to dictionaries for easier processing
            als_pred_dict = {item_id: score for item_id, score in als_predictions}
            twotower_pred_dict = {item_id: score for item_id, score in twotower_predictions}
            
            # Calculate F1 scores (simplified binary relevance)
            self.als_f1_score = self._calculate_f1_for_predictions(
                als_pred_dict, actual_ratings, k
            )
            self.twotower_f1_score = self._calculate_f1_for_predictions(
                twotower_pred_dict, actual_ratings, k
            )
            
            print(f"Model F1 Scores - ALS: {self.als_f1_score:.4f}, "
                  f"Two-Tower: {self.twotower_f1_score:.4f}")
            
            return self.als_f1_score, self.twotower_f1_score
            
        except Exception as e:
            print(f"Error evaluating individual models: {str(e)}")
            return 0.0, 0.0
    
    def _calculate_f1_for_predictions(self, predictions, actual_ratings, k=10):
        """
        Calculate F1 score for predictions using binary relevance
        
        Parameters:
        -----------
        predictions : dict
            Dictionary mapping itemId to prediction score
        actual_ratings : dict
            Dictionary mapping itemId to actual rating
        k : int
            Top-k for evaluation
            
        Returns:
        --------
        float : F1 score
        """
        try:
            # Get top-k predictions
            sorted_predictions = sorted(
                predictions.items(), 
                key=lambda x: x[1], 
                reverse=True
            )[:k]
            
            # Convert to binary relevance (simplified)
            predicted_items = set([item_id for item_id, _ in sorted_predictions])
            
            # Assume items with rating >= mean rating are relevant
            mean_rating = np.mean(list(actual_ratings.values()))
            relevant_items = set([
                item_id for item_id, rating in actual_ratings.items()
                if rating >= mean_rating
            ])
            
            # Calculate precision, recall, F1
            true_positives = len(predicted_items.intersection(relevant_items))
            
            if len(predicted_items) == 0:
                precision = 0.0
            else:
                precision = true_positives / len(predicted_items)
            
            if len(relevant_items) == 0:
                recall = 0.0
            else:
                recall = true_positives / len(relevant_items)
            
            if precision + recall == 0:
                f1 = 0.0
            else:
                f1 = 2 * (precision * recall) / (precision + recall)
            
            return f1
            
        except Exception as e:
            print(f"Error calculating F1 score: {str(e)}")
            return 0.0
    
    def adaptive_fusion(self, als_predictions, twotower_predictions):
        """
        Combine predictions from both models using adaptive weighting
        
        Algorithm:
        - If ALS F1 > Two-Tower F1: 80% ALS + 20% Two-Tower
        - Otherwise: 20% ALS + 80% Two-Tower
        
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
            als_dict = {item_id: score for item_id, score in als_predictions}
            twotower_dict = {item_id: score for item_id, score in twotower_predictions}
            
            # Get all items
            all_items = set(als_dict.keys()).union(set(twotower_dict.keys()))
            
            # Extract scores for normalization
            als_scores = [als_dict.get(item_id, 0) for item_id in all_items]
            twotower_scores = [twotower_dict.get(item_id, 0) for item_id in all_items]
            
            # Normalize scores using MinMaxScaler
            als_scores_norm = self.als_scaler.fit_transform(
                np.array(als_scores).reshape(-1, 1)
            ).flatten()
            twotower_scores_norm = self.twotower_scaler.fit_transform(
                np.array(twotower_scores).reshape(-1, 1)
            ).flatten()
            
            # Determine weights based on F1 scores
            if self.als_f1_score > self.twotower_f1_score:
                als_weight, twotower_weight = 0.8, 0.2
                print("Using ALS-dominant weighting (80-20)")
            else:
                als_weight, twotower_weight = 0.2, 0.8
                print("Using Two-Tower-dominant weighting (20-80)")
            
            # Combine predictions
            combined_predictions = []
            for i, item_id in enumerate(all_items):
                combined_score = (
                    als_weight * als_scores_norm[i] + 
                    twotower_weight * twotower_scores_norm[i]
                )
                combined_predictions.append((item_id, combined_score))
            
            return combined_predictions
            
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
        if not self.is_trained:
            raise ValueError("Models not trained yet. Call train_models() first.")
        
        try:
            # Get predictions from both models
            als_predictions = self.als_model.predict_for_user(user_id, all_items)
            twotower_predictions = self.twotower_model.predict_for_user(user_id, all_items)
            
            # Evaluate models if actual ratings provided
            if actual_ratings:
                self.evaluate_individual_models(user_id, actual_ratings, all_items)
            
            # Combine predictions using adaptive fusion
            combined_predictions = self.adaptive_fusion(als_predictions, twotower_predictions)
            
            # Sort and return top-K
            sorted_predictions = sorted(
                combined_predictions, 
                key=lambda x: x[1], 
                reverse=True
            )
            
            top_recommendations = sorted_predictions[:top_k]
            
            print(f"Generated {len(top_recommendations)} hybrid recommendations for user {user_id}")
            return top_recommendations
            
        except Exception as e:
            print(f"Error generating hybrid recommendations: {str(e)}")
            return []
    
    def cleanup(self):
        """Clean up resources"""
        if self.als_model:
            self.als_model.stop_spark()


if __name__ == "__main__":
    # Example usage
    print("Hybrid Recommendation System module loaded successfully")
    
    # Example configuration
    example_config = {
        'als_params': {'rank': 10, 'max_iter': 10, 'reg_param': 0.1},
        'twotower_params': {'embedding_size': 50, 'learning_rate': 0.001}
    }
    
    print("Example configuration:", example_config)

