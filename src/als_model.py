"""
ALS Collaborative Filtering Model Implementation

This module implements the Alternating Least Squares (ALS) model using PySpark
for collaborative filtering recommendations.
"""

import warnings
from pyspark.sql import SparkSession
from pyspark.ml.recommendation import ALS
from pyspark.sql.types import *
import pandas as pd

warnings.filterwarnings('ignore')


class ALSModel:
    """
    ALS Collaborative Filtering Model using PySpark
    
    Parameters:
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
        Train ALS model on user-item interaction data
        
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
            
            # Initialize ALS model
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
            
            # Convert to list of tuples
            result = [(row.itemId, float(row.prediction)) for row in predictions_list]
            
            print(f"Generated {len(result)} predictions for user {user_id}")
            return result
            
        except Exception as e:
            print(f"Error generating predictions: {str(e)}")
            return []
    
    def get_top_recommendations(self, user_id, all_items, top_k=5):
        """
        Get top-K recommendations for a user
        
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


def hyperparameter_tuning(data, param_grid):
    """
    Perform hyperparameter tuning for ALS model
    
    Parameters:
    -----------
    data : pd.DataFrame
        Training data
    param_grid : dict
        Dictionary containing parameter combinations to try
        
    Returns:
    --------
    dict : Best parameters found
    """
    best_params = None
    best_score = float('-inf')
    
    for params in param_grid:
        model = ALSModel(
            rank=params['rank'],
            max_iter=params['max_iter'],
            reg_param=params['reg_param']
        )
        
        if model.train(data):
            # Here you would evaluate the model and get a score
            # For now, we'll just return the first valid params
            print(f"Tested params: {params}")
            model.stop_spark()
            
            if best_params is None:
                best_params = params
    
    return best_params


if __name__ == "__main__":
    # Example usage
    print("ALS Model module loaded successfully")
    
    # Example parameter grid for hyperparameter tuning
    param_grid = [
        {'rank': 10, 'max_iter': 10, 'reg_param': 0.1},
        {'rank': 20, 'max_iter': 20, 'reg_param': 0.05},
        {'rank': 15, 'max_iter': 15, 'reg_param': 0.5},
        {'rank': 20, 'max_iter': 5, 'reg_param': 0.1},
        {'rank': 15, 'max_iter': 12, 'reg_param': 0.2}
    ]
    
    print("Available hyperparameter combinations:", len(param_grid))

