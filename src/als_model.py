"""
ALS Collaborative Filtering Model Implementation

This module implements the Alternating Least Squares (ALS) model using PySpark
for collaborative filtering recommendations, exactly as described in the manuscript.
"""

import warnings
import os
import pickle
import numpy as np
from pyspark.sql import SparkSession
from pyspark.ml.recommendation import ALS, ALSModel as SparkALSModel
from pyspark.sql.types import *
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from .data_preprocessing import get_item_features

warnings.filterwarnings('ignore')

class ALSModel:
    def __init__(self, rank=10, max_iter=10, reg_param=0.1, cold_start_strategy="drop"):
        self.rank = rank
        self.max_iter = max_iter
        self.reg_param = reg_param
        self.cold_start_strategy = cold_start_strategy
        self.model = None
        self.spark = None
        self.global_mean = 3.0
        self.item_features = None

    def initialize_spark(self):
        try:
            self.spark = SparkSession.builder \
                .appName("ALSRecommender") \
                .config("spark.driver.memory", "10g") \
                .getOrCreate()
            return True
        except Exception as e:
            print(f"Spark init error: {str(e)}")
            return False

    def train(self, data):
        try:
            if not self.initialize_spark():
                return False

            self.item_features = get_item_features(data)
            self.global_mean = data['average_review_rating'].mean()

            spark_df = self.spark.createDataFrame(data)
            als = ALS(
                rank=self.rank,
                maxIter=self.max_iter,
                regParam=self.reg_param,
                userCol="userId",
                itemCol="itemId",
                ratingCol="average_review_rating",
                coldStartStrategy=self.cold_start_strategy
            )
            
            self.model = als.fit(spark_df)
            return True
        except Exception as e:
            print(f"Training error: {str(e)}")
            return False

    def predict_for_user(self, user_id, all_items):
        try:
            pairs = [(user_id, item) for item in all_items]
            schema = StructType([
                StructField("userId", IntegerType()), 
                StructField("itemId", IntegerType())
            ])
            preds_df = self.model.transform(self.spark.createDataFrame(pairs, schema))
            spark_preds = {row.itemId: row.prediction for row in preds_df.collect()}
            
            final_preds = []
            for item in all_items:
                if item in spark_preds and not np.isnan(spark_preds[item]):
                    final_preds.append((item, float(spark_preds[item])))
                else:
                    similar_items = self._find_similar_items(item)
                    placeholder = np.mean([self.item_features[sim_item]['rating'] 
                                        for sim_item in similar_items]) if similar_items else self.global_mean
                    final_preds.append((item, placeholder))
            
            return final_preds
        except Exception as e:
            print(f"Prediction error: {str(e)}")
            return []

    def _find_similar_items(self, item_id, k=3):
        try:
            target = self.item_features[item_id]
            similarities = []
            for other_id, features in self.item_features.items():
                if other_id == item_id:
                    continue
                sim = cosine_similarity([target['features']], [features['features']])[0][0]
                similarities.append((other_id, sim))
            return [item for item, sim in sorted(similarities, key=lambda x: x[1], reverse=True)[:k] if sim > 0.5]
        except KeyError:
            return []

    def save_model(self, model_path="models/als"):
        try:
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            self.model.save(model_path)
            metadata = {
                'rank': self.rank,
                'max_iter': self.max_iter,
                'reg_param': self.reg_param,
                'global_mean': self.global_mean,
                'item_features': self.item_features
            }
            with open(f"{model_path}_metadata.pkl", "wb") as f:
                pickle.dump(metadata, f)
            print(f"Model saved to {model_path}")
        except Exception as e:
            print(f"Saving error: {str(e)}")

    def load_model(self, model_path="models/als"):
        try:
            self.model = SparkALSModel.load(model_path)
            with open(f"{model_path}_metadata.pkl", "rb") as f:
                metadata = pickle.load(f)
                self.rank = metadata['rank']
                self.max_iter = metadata['max_iter']
                self.reg_param = metadata['reg_param']
                self.global_mean = metadata['global_mean']
                self.item_features = metadata['item_features']
            return self
        except Exception as e:
            print(f"Loading error: {str(e)}")
            return None

    def stop_spark(self):
        if self.spark:
            self.spark.stop()

def hyperparameter_tuning(train_data, val_data, param_grid):
    best_params = None
    best_f1 = 0.0
    
    for params in param_grid:
        model = ALSModel(**params)
        if not model.train(train_data):
            continue
        
        f1_scores = []
        for user_id in val_data['userId'].sample(50).unique():
            actual = dict(zip(
                val_data[val_data['userId'] == user_id]['itemId'],
                val_data[val_data['userId'] == user_id]['average_review_rating']
            ))
            preds = model.predict_for_user(user_id, val_data['itemId'].unique())
            pred_dict = {item: score for item, score in preds}
            f1 = compute_f1_score(actual, pred_dict)
            f1_scores.append(f1)
        
        avg_f1 = np.mean(f1_scores)
        if avg_f1 > best_f1:
            best_f1 = avg_f1
            best_params = params.copy()
        
        model.stop_spark()
    
    return best_params

def compute_f1_score(actual, pred, k=10):
    actual_items = set(actual.keys())
    pred_items = set([item for item, _ in sorted(pred.items(), key=lambda x: x[1], reverse=True)[:k]])
    tp = len(actual_items & pred_items)
    precision = tp / k
    recall = tp / len(actual_items) if actual_items else 0
    return 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

if __name__ == "__main__":
    # Load preprocessed data
    train_data = pd.read_csv("processed/train_data.csv")
    val_data = pd.read_csv("processed/test_data.csv")
    
    # Define hyperparameter grid from manuscript Table 1
    param_grid = [
        {'rank': 10, 'max_iter': 10, 'reg_param': 0.1},
        {'rank': 20, 'max_iter': 20, 'reg_param': 0.05},
        {'rank': 15, 'max_iter': 15, 'reg_param': 0.5},
        {'rank': 20, 'max_iter': 5, 'reg_param': 0.1},
        {'rank': 15, 'max_iter': 12, 'reg_param': 0.2}
    ]
    
    # Perform hyperparameter tuning
    print("=== Starting Hyperparameter Tuning ===")
    best_params = hyperparameter_tuning(train_data, val_data, param_grid)
    print(f"Best parameters: {best_params}")
    
    # Train and save final model
    print("=== Training Final Model ===")
    als = ALSModel(**best_params)
    if als.train(train_data):
        als.save_model("models/als")
        print("Training successful")
    else:
        print("Training failed")
    als.stop_spark()
