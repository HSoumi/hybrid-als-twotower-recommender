"""
Utility Functions for Hybrid Recommender System

This module contains helper functions that support the evaluation and 
data processing needs of the hybrid recommender system implementation.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import warnings

warnings.filterwarnings('ignore')


def scale_ratings_to_5(ratings):
    """
    Scale ratings to 1-5 range for MAE/RMSE calculation
    
    This function is used in the research to standardize ratings before
    calculating MAE and RMSE metrics, ensuring fair comparison across models.
    
    Parameters:
    -----------
    ratings : list or np.array
        Ratings to scale
        
    Returns:
    --------
    list : Scaled ratings on 1-5 scale
    """
    try:
        ratings_array = np.array(ratings)
        min_rating = ratings_array.min()
        max_rating = ratings_array.max()
        
        if max_rating == min_rating:
            return [3.0] * len(ratings)  # Default to middle value
        
        # Scale to 1-5 range
        scaled = 1 + 4 * (ratings_array - min_rating) / (max_rating - min_rating)
        return scaled.tolist()
        
    except Exception as e:
        print(f"Error scaling ratings: {str(e)}")
        return ratings


def normalize_predictions(predictions_dict):
    """
    Normalize prediction scores using MinMaxScaler (used in hybrid fusion)
    
    This function is specifically used in the hybrid model's adaptive fusion
    strategy to normalize ALS and Two-Tower predictions before combining them.
    
    Parameters:
    -----------
    predictions_dict : dict
        Dictionary mapping itemId to prediction score
        
    Returns:
    --------
    dict : Dictionary with normalized scores
    """
    try:
        if not predictions_dict:
            return {}
        
        items = list(predictions_dict.keys())
        scores = list(predictions_dict.values())
        
        scaler = MinMaxScaler()
        normalized_scores = scaler.fit_transform(np.array(scores).reshape(-1, 1)).flatten()
        
        return dict(zip(items, normalized_scores))
        
    except Exception as e:
        print(f"Error normalizing predictions: {str(e)}")
        return predictions_dict


def get_user_item_interactions(data, user_id):
    """
    Get item interactions for a specific user (used in evaluation)
    
    This function extracts all interactions for a given user from the dataset,
    which is used during the evaluation phase of the hybrid recommender system.
    
    Parameters:
    -----------
    data : pd.DataFrame
        Dataset with userId, itemId, rating columns
    user_id : int
        User ID to get interactions for
        
    Returns:
    --------
    dict : Dictionary mapping itemId to rating
    """
    try:
        user_data = data[data['userId'] == user_id]
        interactions = dict(zip(user_data['itemId'], user_data['average_review_rating']))
        
        print(f"Found {len(interactions)} interactions for user {user_id}")
        return interactions
        
    except Exception as e:
        print(f"Error getting user interactions: {str(e)}")
        return {}


def print_evaluation_results(results_dict, user_id=None):
    """
    Print evaluation results in a formatted way for clear output
    
    Parameters:
    -----------
    results_dict : dict
        Dictionary containing evaluation metrics
    user_id : int, optional
        User ID for the results
    """
    try:
        print("=" * 50)
        if user_id:
            print(f"Evaluation Results for User {user_id}")
        else:
            print("Evaluation Results")
        print("=" * 50)
        
        for metric, value in results_dict.items():
            if isinstance(value, float):
                print(f"{metric:15}: {value:.4f}")
            else:
                print(f"{metric:15}: {value}")
        
        print("=" * 50)
        
    except Exception as e:
        print(f"Error printing results: {str(e)}")


def display_dataset_info(data):
    """
    Display basic dataset information (used in preprocessing phase)
    
    Parameters:
    -----------
    data : pd.DataFrame
        Dataset to analyze
    """
    try:
        print("Dataset Information:")
        print(f"Shape: {data.shape}")
        print(f"Unique users: {data['userId'].nunique()}")
        print(f"Unique items: {data['itemId'].nunique()}")
        print(f"Total interactions: {len(data)}")
        
        # Sparsity calculation (relevant for recommender systems)
        total_possible = data['userId'].nunique() * data['itemId'].nunique()
        sparsity = (1 - (len(data) / total_possible)) * 100
        print(f"Sparsity: {sparsity:.2f}%")
        
    except Exception as e:
        print(f"Error displaying dataset info: {str(e)}")


if __name__ == "__main__":
    # Example usage of utility functions
    print("Utility functions for Hybrid Recommender System loaded successfully")
    
    print("\nAvailable utility functions:")
    print("- scale_ratings_to_5(): Scale ratings to 1-5 range for evaluation")
    print("- normalize_predictions(): Normalize scores for hybrid fusion")
    print("- get_user_item_interactions(): Extract user interaction data")
    print("- print_evaluation_results(): Format evaluation output")
    print("- display_dataset_info(): Show dataset statistics")
    
    # Example demonstration
    sample_ratings = [1, 2, 3, 4, 5]
    scaled = scale_ratings_to_5(sample_ratings)
    print(f"\nExample: Sample ratings {sample_ratings} scaled to: {scaled}")

