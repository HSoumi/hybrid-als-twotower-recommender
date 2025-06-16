"""
ALS Collaborative Filtering Model Implementation

This module implements the Alternating Least Squares (ALS) model using PySpark
for collaborative filtering recommendations, exactly as described in the manuscript.
"""

import warnings
from pyspark.sql import SparkSession
from pyspark.ml.recommendation import ALS
from pyspark.sql.types import *
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from .evaluation import compute_f1_score

warnings.filterwarnings('ignore')


class ALSModel:
    """
    ALS Collaborative Filtering Model using PySpark
    
    Parameters (matches manuscript Table 1):
    -----------
    rank : int
        Number of latent factors (default: 10)
    max_iter : int
        Maximum iterations for training (default: 10)
    reg_param : float
        Regularization parameter (default: 0.1)
    cold_start_strategy : str
        Strategy for handling cold start problem (default: "drop")
    """
    
    def __init__(self, rank=10, max_iter=10, reg_param=0.1, cold_start_strategy="drop"):
        self.rank = rank
        self.max_iter = max_iter
        self.reg_param = reg_param
        self.cold_start_strategy = cold_start_strategy
        self.model = None
        self.spark = None
        
    def initialize_spark(self, app_name="ALSRecommender", driver_memory="10g"):
        """Initialize Spark session with optimized configuration"""
        try:
            self.spark = SparkSession.builder \
                .appName(app_name) \
                .config("spark.driver.memory", driver_memory) \
                .getOrCreate()
            print(f"Spark session initialized successfully")
            return True
        except Exception as e:
            print(f"Error initializing Spark: {str(e)}")
            return False
    
    def train(self, data):
        """
        Train ALS model on user-item interaction data (matches manuscript §4.2)
        
        Parameters:
        -----------
        data : pd.DataFrame
            DataFrame with columns: userId, itemId, average_review_rating
            
        Returns:
        --------
        bool : Success status of training
        """
        try:
            if self.spark is None:
                if not self.initialize_spark():
                    return False
            
            # Convert pandas DataFrame to Spark DataFrame
            spark_df = self.spark.createDataFrame(data)
            
            # Initialize ALS model with parameters from Table 1
            als = ALS(
                rank=self.rank,
                maxIter=self.max_iter,
                regParam=self.reg_param,
                userCol="userId",
                itemCol="itemId",
                ratingCol="average_review_rating",
                coldStartStrategy=self.cold_start_strategy
            )
            
            # Train the model
            self.model = als.fit(spark_df)
            print(f"ALS model trained successfully with rank={self.rank}, "
                  f"maxIter={self.max_iter}, regParam={self.reg_param}")
            return True
            
        except Exception as e:
            print(f"Error training ALS model: {str(e)}")
            return False
    
    def predict_for_user(self, user_id, all_items):
        """
        Generate predictions for a specific user across all items
        (matches manuscript evaluation methodology)
        
        Parameters:
        -----------
        user_id : int
            User ID for which to generate predictions
        all_items : list
            List of all item IDs
            
        Returns:
        --------
        list : List of (itemId, prediction_score) tuples
        """
        try:
            if self.model is None:
                raise ValueError("Model not trained yet. Call train() first.")
            
            # Create user-item pairs for prediction
            user_item_pairs = [(user_id, item_id) for item_id in all_items]
            user_item_df = self.spark.createDataFrame(
                user_item_pairs, 
                schema=StructType([
                    StructField("userId", IntegerType(), True),
                    StructField("itemId", IntegerType(), True)
                ])
            )
            
            # Generate predictions
            predictions = self.model.transform(user_item_df)
            predictions_list = predictions.select("itemId", "prediction").collect()
            result = [(row.itemId, float(row.prediction)) for row in predictions_list]
    
            # Cold-start handling (manuscript's probability imputation)
            from src.data_preprocessing import get_placeholder_rating  # New import
            all_items_set = set(all_items)
            predicted_items = {item_id for item_id, _ in result}
            cold_items = all_items_set - predicted_items
    
            for item_id in cold_items:
                placeholder = get_placeholder_rating(item_id)  # Implemented in data_preprocessing
                result.append((item_id, placeholder))
            return result
            
        except Exception as e:
            print(f"Error generating predictions: {str(e)}")
            return []
    
    def get_top_recommendations(self, user_id, all_items, top_k=5):
        """
        Get top-K recommendations for a user (matches manuscript §4.4)
        
        Parameters:
        -----------
        user_id : int
            User ID for recommendations
        all_items : list
            List of all available item IDs
        top_k : int
            Number of top recommendations to return
            
        Returns:
        --------
        list : List of top-K (itemId, score) tuples sorted by score
        """
        predictions = self.predict_for_user(user_id, all_items)
        if not predictions:
            return []
        
        # Sort by prediction score in descending order and get top-K
        sorted_predictions = sorted(predictions, key=lambda x: x[1], reverse=True)
        return sorted_predictions[:top_k]
    
    def stop_spark(self):
        """Stop Spark session"""
        if self.spark:
            self.spark.stop()
            self.spark = None
            print("Spark session stopped")


def hyperparameter_tuning(data, param_grid, test_size=0.2, random_state=42):
    """
    Perform F1-based hyperparameter tuning as described in manuscript §4.3
    
    Parameters:
    -----------
    data : pd.DataFrame
        Training data with columns: userId, itemId, average_review_rating
    param_grid : list
        List of parameter dictionaries to try (matches Table 1)
    test_size : float
        Proportion of data to use for validation
    random_state : int
        Random seed for reproducibility
        
    Returns:
    --------
    dict : Best parameters found based on F1@10
    """
    best_params = None
    best_f1 = 0.0
    
    # Split data into train and validation sets
    train_data, val_data = train_test_split(data, test_size=test_size, 
                                          random_state=random_state)
    
    print(f"Hyperparameter tuning with {len(param_grid)} combinations")
    print(f"Training samples: {len(train_data)}, Validation samples: {len(val_data)}")
    
    for params in param_grid:
        print(f"\nTesting parameters: {params}")
        model = ALSModel(**params)
        
        try:
            # Train model
            if not model.train(train_data):
                continue
            
            # Prepare validation users (5% sample for efficiency)
            val_users = val_data['userId'].unique()
            sample_users = np.random.choice(val_users, 
                                           size=int(len(val_users)*0.05), 
                                           replace=False)
            
            f1_scores = []
            for user_id in sample_users:
                # Get user's actual ratings
                user_ratings = val_data[val_data['userId'] == user_id]
                actual = dict(zip(user_ratings['itemId'], 
                                user_ratings['average_review_rating']))
                
                # Get all items for prediction
                all_items = val_data['itemId'].unique().tolist()
                
                # Generate predictions
                preds = model.predict_for_user(user_id, all_items)
                pred_dict = {item: score for item, score in preds}
                
                # Compute F1@10
                f1 = compute_f1_score(actual, pred_dict, k=10)
                f1_scores.append(f1)
            
            # Calculate average F1 score
            avg_f1 = np.mean(f1_scores) if f1_scores else 0.0
            print(f"Average F1@10: {avg_f1:.4f}")
            
            # Update best parameters
            if avg_f1 > best_f1:
                best_f1 = avg_f1
                best_params = params.copy()
                print(f"New best F1: {best_f1:.4f}")
            
        finally:
            # Cleanup Spark resources
            model.stop_spark()
    
    print(f"\nBest parameters: {best_params} (F1@10: {best_f1:.4f})")
    return best_params


if __name__ == "__main__":
    # Example usage matching manuscript parameters
    print("ALS Model module loaded successfully")
    
    # Parameter grid from manuscript Table 1
    param_grid = [
        {'rank': 10, 'max_iter': 10, 'reg_param': 0.1},
        {'rank': 20, 'max_iter': 20, 'reg_param': 0.05},
        {'rank': 15, 'max_iter': 15, 'reg_param': 0.5},
        {'rank': 20, 'max_iter': 5, 'reg_param': 0.1},
        {'rank': 15, 'max_iter': 12, 'reg_param': 0.2}
    ]
    
    print("Available hyperparameter combinations:", len(param_grid))

