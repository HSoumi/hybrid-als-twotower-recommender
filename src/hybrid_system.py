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
    def __init__(self):
        self.als_model = None
        self.twotower_model = None
        self.als_scaler = MinMaxScaler()
        self.twotower_scaler = MinMaxScaler()
        self.als_f1_score = 0.0
        self.twotower_f1_score = 0.0
        self.models_loaded = False

    def load_models(self, als_model_path, twotower_model_path):
        try:
            print("=== Loading Pre-trained Models ===")
            self.als_model = ALSModel().load_model(als_model_path)
            self.twotower_model = TwoTowerModel.load_model(twotower_model_path)
            self.models_loaded = True
            print("\n=== Models loaded successfully ===")
            return True
        except Exception as e:
            print(f"Error loading models: {str(e)}")
            return False

    def evaluate_individual_models(self, test_user_id, actual_ratings, all_items, k=10):
        try:
            als_preds = self.als_model.predict_for_user(test_user_id, all_items)
            tt_preds = self.twotower_model.predict_for_user(test_user_id, all_items)
            
            self.als_f1_score = compute_f1_score(actual_ratings, dict(als_preds))
            self.twotower_f1_score = compute_f1_score(actual_ratings, dict(tt_preds))
            
            print(f"Model F1 Scores - ALS: {self.als_f1_score:.4f}, "
                  f"Two-Tower: {self.twotower_f1_score:.4f}")
            return self.als_f1_score, self.twotower_f1_score
        except Exception as e:
            print(f"Error evaluating models: {str(e)}")
            return 0.0, 0.0

    def adaptive_fusion(self, als_predictions, twotower_predictions):
        try:
            als_dict = dict(als_predictions)
            tt_dict = dict(twotower_predictions)
            all_items = set(als_dict.keys()).union(set(tt_dict.keys()))
            
            als_scores = [als_dict.get(item, 0) for item in all_items]
            tt_scores = [tt_dict.get(item, 0) for item in all_items]
            
            als_norm = self.als_scaler.fit_transform(np.array(als_scores).reshape(-1, 1)).flatten()
            tt_norm = self.twotower_scaler.fit_transform(np.array(tt_scores).reshape(-1, 1)).flatten()
            
            weights = (0.8, 0.2) if self.als_f1_score > self.twotower_f1_score else (0.2, 0.8)
            
            return [(item, weights[0]*als_norm[i] + weights[1]*tt_norm[i]) 
                   for i, item in enumerate(all_items)]
        except Exception as e:
            print(f"Error in adaptive fusion: {str(e)}")
            return []

    def save_predictions(self, user_id, predictions, save_dir="results/predictions"):
        os.makedirs(save_dir, exist_ok=True)
        file_path = os.path.join(save_dir, f"user_{user_id}_predictions.csv")
        df = pd.DataFrame(predictions, columns=['itemId', 'hybrid_score'])
        df['userId'] = user_id
        df['prediction_rank'] = range(1, len(df)+1)
        df['timestamp'] = pd.Timestamp.now()
        df.to_csv(file_path, index=False)
        print(f"Predictions saved to {file_path}")
        return file_path

    def load_predictions(self, user_id, save_dir="results/predictions"):
        file_path = os.path.join(save_dir, f"user_{user_id}_predictions.csv")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"No predictions found for user {user_id}")
        df = pd.read_csv(file_path)
        return list(zip(df['itemId'], df['hybrid_score']))

    def get_hybrid_recommendations(self, user_id, all_items, actual_ratings=None, 
                                  top_k=5, save_predictions=False):
        if not self.models_loaded:
            raise ValueError("Models not loaded. Call load_models() first.")
        
        try:
            als_preds = self.als_model.predict_for_user(user_id, all_items)
            tt_preds = self.twotower_model.predict_for_user(user_id, all_items)
            
            if actual_ratings:
                self.evaluate_individual_models(user_id, actual_ratings, all_items)
            
            combined = self.adaptive_fusion(als_preds, tt_preds)
            top_recommendations = sorted(combined, key=lambda x: x[1], reverse=True)[:top_k]
            
            if save_predictions:
                self.save_predictions(user_id, combined)
            
            return top_recommendations
        except Exception as e:
            print(f"Error generating recommendations: {str(e)}")
            return []

    def cleanup(self):
        if self.als_model:
            self.als_model.stop_spark()

if __name__ == "__main__":
    print("Hybrid Recommendation System module loaded successfully")
    print("Example usage for research users:")
    print("hrs = HybridRecommendationSystem()")
    print("hrs.load_models('models/als', 'models/twotower.keras')")
    
    # Process both research users
    research_users = [462, 9435]  # The chosen users as per the manuscript
    for user_id in research_users:
        print(f"recommendations_{user_id} = hrs.get_hybrid_recommendations({user_id}, all_items, save_predictions=True)")
