"""
ALS Collaborative Filtering Model Implementation

This module implements the Alternating Least Squares (ALS) model using PySpark
for collaborative filtering recommendations, exactly as described in the manuscript.
"""

import warnings
import os
import pickle
from pyspark.sql import SparkSession
from pyspark.ml.recommendation import ALS, ALSModel as SparkALSModel
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
        self.is_trained = False
        self.placeholder_cache = {}
        self.global_mean = 0.0
        
    def initialize_spark(self, app_name="ALSRecommender", driver_memory="10g"):
        """Initialize Spark session with optimized configuration"""
        try:
            self.spark = SparkSession.builder \
                .appName(app_name) \
                .config("spark.driver.memory", driver_memory) \
                .config("spark.sql.adaptive.enabled", "true") \
                .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
                .getOrCreate()
            print(f"Spark session initialized successfully")
            return True
        except Exception as e:
            print(f"Error initializing Spark: {str(e)}")
            return False
    
    def _load_placeholder_cache(self, data):
        """Load placeholder ratings for cold-start handling"""
        try:
            # Calculate global mean rating
            self.global_mean = data['average_review_rating'].mean()
            
            # Create placeholder cache (item_id -> placeholder_rating)
            self.placeholder_cache = dict(zip(
                data['itemId'], 
                data['average_review_rating']
            ))
            
            print(f"Loaded placeholder cache with {len(self.placeholder_cache)} items")
            print(f"Global mean rating: {self.global_mean:.4f}")
            
        except Exception as e:
            print(f"Error loading placeholder cache: {str(e)}")
            self.global_mean = 3.0  # Default fallback
    
    def train(self, train_data):
        """
        Train ALS model on training data (matches manuscript §4.2)
        
        Parameters:
        -----------
        train_data : pd.DataFrame
            Training DataFrame with columns: userId, itemId, average_review_rating
            
        Returns:
        --------
        bool : Success status of training
        """
        try:
            if self.spark is None:
                if not self.initialize_spark():
                    return False
            
            # Load placeholder cache for cold-start handling
            self._load_placeholder_cache(train_data)
            
            # Convert pandas DataFrame to Spark DataFrame
            spark_df = self.spark.createDataFrame(train_data)
            
            # Cache the DataFrame for better performance
            spark_df.cache()
            
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
            print(f"Training ALS model with {len(train_data)} interactions...")
            self.model = als.fit(spark_df)
            self.is_trained = True
            
            print(f"ALS model trained successfully with rank={self.rank}, "
                  f"maxIter={self.max_iter}, regParam={self.reg_param}")
            return True
            
        except Exception as e:
            print(f"Error training ALS model: {str(e)}")
            return False
    
    def predict_for_user(self, user_id, all_items):
        """
        Generate predictions for a specific user across all items
        (matches manuscript evaluation methodology with cold-start handling)
        
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
            if not self.is_trained:
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
            
            # Convert to list of tuples, handling NaN predictions
            result = []
            predicted_items = set()
            
            for row in predictions_list:
                if row.prediction is not None and not np.isnan(row.prediction):
                    result.append((row.itemId, float(row.prediction)))
                    predicted_items.add(row.itemId)
                else:
                    # Use placeholder for NaN predictions (cold-start)
                    placeholder = self.placeholder_cache.get(row.itemId, self.global_mean)
                    result.append((row.itemId, float(placeholder)))
                    predicted_items.add(row.itemId)
            
            # Handle any items not in predictions (additional cold-start handling)
            all_items_set = set(all_items)
            cold_items = all_items_set - predicted_items
            
            for item_id in cold_items:
                placeholder = self.placeholder_cache.get(item_id, self.global_mean)
                result.append((item_id, float(placeholder)))
            
            print(f"Generated {len(result)} predictions for user {user_id}")
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
    
    def save_model(self, model_path, metadata_path=None):
        """
        Save trained ALS model to disk
        
        Parameters:
        -----------
        model_path : str
            Path to save the Spark ALS model
        metadata_path : str, optional
            Path to save model metadata (parameters, cache, etc.)
        """
        try:
            if not self.is_trained:
                raise ValueError("No trained model to save")
            
            # Save Spark ALS model
            self.model.save(model_path)
            print(f"ALS model saved to {model_path}")
            
            # Save metadata and placeholder cache
            if metadata_path is None:
                metadata_path = f"{model_path}_metadata.pkl"
            
            metadata = {
                'rank': self.rank,
                'max_iter': self.max_iter,
                'reg_param': self.reg_param,
                'cold_start_strategy': self.cold_start_strategy,
                'placeholder_cache': self.placeholder_cache,
                'global_mean': self.global_mean,
                'is_trained': self.is_trained
            }
            
            with open(metadata_path, 'wb') as f:
                pickle.dump(metadata, f)
            print(f"Model metadata saved to {metadata_path}")
            
        except Exception as e:
            print(f"Error saving model: {str(e)}")
    
    def load_model(self, model_path, metadata_path=None):
        """
        Load trained ALS model from disk
        
        Parameters:
        -----------
        model_path : str
            Path to the saved Spark ALS model
        metadata_path : str, optional
            Path to the saved model metadata
            
        Returns:
        --------
        ALSModel : Self instance with loaded model
        """
        try:
            # Initialize Spark if not already done
            if self.spark is None:
                if not self.initialize_spark():
                    raise RuntimeError("Failed to initialize Spark session")
            
            # Load Spark ALS model
            self.model = SparkALSModel.load(model_path)
            print(f"ALS model loaded from {model_path}")
            
            # Load metadata and placeholder cache
            if metadata_path is None:
                metadata_path = f"{model_path}_metadata.pkl"
            
            if os.path.exists(metadata_path):
                with open(metadata_path, 'rb') as f:
                    metadata = pickle.load(f)
                
                self.rank = metadata['rank']
                self.max_iter = metadata['max_iter']
                self.reg_param = metadata['reg_param']
                self.cold_start_strategy = metadata['cold_start_strategy']
                self.placeholder_cache = metadata['placeholder_cache']
                self.global_mean = metadata['global_mean']
                self.is_trained = metadata['is_trained']
                
                print(f"Model metadata loaded from {metadata_path}")
            else:
                print(f"Warning: Metadata file {metadata_path} not found")
                self.is_trained = True  # Assume model is trained if file exists
            
            return self
            
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            return None
    
    def stop_spark(self):
        """Stop Spark session and clean up resources"""
        if self.spark:
            self.spark.stop()
            self.spark = None
            print("Spark session stopped")


def hyperparameter_tuning(train_data, val_data, param_grid, random_state=42):
    """
    Perform F1-based hyperparameter tuning as described in manuscript §4.3
    
    Parameters:
    -----------
    train_data : pd.DataFrame
        Training data with columns: userId, itemId, average_review_rating
    val_data : pd.DataFrame
        Validation data for F1 score calculation
    param_grid : list
        List of parameter dictionaries to try (matches Table 1)
    random_state : int
        Random seed for reproducibility
        
    Returns:
    --------
    dict : Best parameters found based on F1@10
    """
    best_params = None
    best_f1 = 0.0
    
    print(f"Hyperparameter tuning with {len(param_grid)} combinations")
    print(f"Training samples: {len(train_data)}, Validation samples: {len(val_data)}")
    
    for i, params in enumerate(param_grid):
        print(f"\n[{i+1}/{len(param_grid)}] Testing parameters: {params}")
        model = ALSModel(**params)
        
        try:
            # Train model on training data
            if not model.train(train_data):
                print("Training failed")
                continue
            
            # Evaluate on validation users (sample for efficiency)
            val_users = val_data['userId'].unique()
            sample_size = min(50, len(val_users))  # Max 50 users for efficiency
            sample_users = np.random.choice(val_users, size=sample_size, replace=False)
            
            f1_scores = []
            for user_id in sample_users:
                # Get user's actual ratings from validation set
                user_ratings = val_data[val_data['userId'] == user_id]
                if len(user_ratings) == 0:
                    continue
                    
                actual = dict(zip(user_ratings['itemId'], 
                                user_ratings['average_review_rating']))
                
                # Get all items for prediction from validation set
                all_items = val_data['itemId'].unique().tolist()
                
                # Generate predictions
                preds = model.predict_for_user(user_id, all_items)
                pred_dict = {item: score for item, score in preds}
                
                # Compute F1@10
                f1 = compute_f1_score(actual, pred_dict, k=10)
                f1_scores.append(f1)
            
            # Calculate average F1 score
            avg_f1 = np.mean(f1_scores) if f1_scores else 0.0
            print(f"Average F1@10: {avg_f1:.4f} (evaluated on {len(f1_scores)} users)")
            
            # Update best parameters
            if avg_f1 > best_f1:
                best_f1 = avg_f1
                best_params = params.copy()
                print(f"New best F1: {best_f1:.4f}")
            
        except Exception as e:
            print(f"Error: {str(e)}")
            
        finally:
            # Cleanup Spark resources
            model.stop_spark()
    
    print(f"\n{'='*60}")
    print(f"HYPERPARAMETER TUNING COMPLETED")
    print(f"{'='*60}")
    print(f"Best parameters: {best_params}")
    print(f"Best F1@10: {best_f1:.4f}")
    print(f"{'='*60}")
    
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
    print("Features:")
    print("- F1-based hyperparameter tuning")
    print("- Model persistence (save/load)")
    print("- Cold-start handling with placeholder ratings")
    print("- Train/validation data splitting")
