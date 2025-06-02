"""
Hybrid Recommender System Package

This package implements a hybrid recommendation system combining ALS collaborative
filtering and Two-Tower content-based filtering for Amazon e-commerce recommendations.

Modules:
--------
- data_preprocessing: Data cleaning and feature engineering
- als_model: ALS collaborative filtering implementation
- two_tower_model: Two-Tower neural content-based filtering
- hybrid_system: Hybrid model combining both approaches
- evaluation: Comprehensive evaluation metrics
- utils: Utility functions and helpers

Example Usage:
--------------
from src.hybrid_system import HybridRecommendationSystem
from src.evaluation import RecommenderEvaluator
from src.utils import normalize_predictions

# Initialize hybrid system
hybrid_system = HybridRecommendationSystem()

# Train models
hybrid_system.train_models(training_data)

# Get recommendations
recommendations = hybrid_system.get_hybrid_recommendations(user_id=462, all_items=item_list)

# Evaluate performance
evaluator = RecommenderEvaluator()
results = evaluator.comprehensive_evaluation(actual_ratings, predicted_scores)
"""

__version__ = "1.0.0"
__author__ = "Soumi Hazra"

# Import main classes for easy access
from .hybrid_system import HybridRecommendationSystem
from .als_model import ALSModel
from .two_tower_model import TwoTowerModel
from .evaluation import RecommenderEvaluator
from .utils import (
    normalize_predictions,
    get_user_item_interactions,
    print_evaluation_results,
    display_dataset_info,
    scale_ratings_to_5
)

# Package metadata
__all__ = [
    'HybridRecommendationSystem',
    'ALSModel', 
    'TwoTowerModel',
    'RecommenderEvaluator',
    'normalize_predictions',
    'get_user_item_interactions',
    'print_evaluation_results',
    'display_dataset_info',
    'scale_ratings_to_5'
]

# Package information
PACKAGE_INFO = {
    'name': 'Hybrid Recommender System',
    'version': __version__,
    'description': 'A hybrid recommendation system combining ALS and Two-Tower models',
    'author': __author__,
    'url': 'https://github.com/yourusername/hybrid-als-twotower-recommender',
    'license': 'MIT'
}

def get_package_info():
    """
    Get package information
    
    Returns:
    --------
    dict : Package information dictionary
    """
    return PACKAGE_INFO

def print_package_info():
    """Print package information"""
    print(f"Package: {PACKAGE_INFO['name']}")
    print(f"Version: {PACKAGE_INFO['version']}")
    print(f"Description: {PACKAGE_INFO['description']}")
    print(f"Author: {PACKAGE_INFO['author']}")
    print(f"License: {PACKAGE_INFO['license']}")

# Configuration constants
DEFAULT_CONFIG = {
    'ALS_PARAMS': {
        'rank': 10,
        'max_iter': 10,
        'reg_param': 0.1,
        'cold_start_strategy': 'drop'
    },
    'TWO_TOWER_PARAMS': {
        'embedding_size': 50,
        'learning_rate': 0.001
    },
    'EVALUATION_PARAMS': {
        'k_values': [5, 10, 15, 20],
        'top_k': 5
    }
}

def get_default_config():
    """
    Get default configuration parameters
    
    Returns:
    --------
    dict : Default configuration dictionary
    """
    return DEFAULT_CONFIG.copy()

# Module initialization
print(f"Hybrid Recommender System Package v{__version__} loaded successfully")
print("Available modules: als_model, two_tower_model, hybrid_system, evaluation, utils")

