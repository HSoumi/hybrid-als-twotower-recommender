"""
Evaluation Metrics for Recommender Systems

This module implements various evaluation metrics including Precision@k, Recall@k,
F1 Score, NDCG, MAE, and RMSE for recommender system evaluation.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, ndcg_score, f1_score
from sklearn.preprocessing import MinMaxScaler
import warnings
import os

warnings.filterwarnings('ignore')

class RecommenderEvaluator:
    def __init__(self):
        self.scaler = MinMaxScaler()
        self.rating_scaler = MinMaxScaler()

    def precision_at_k(self, actual_ratings, predicted_scores, k=10):
        """Calculate Precision@k with manuscript's thresholding"""
        sorted_preds = sorted(predicted_scores.items(), key=lambda x: x[1], reverse=True)[:k]
        pred_items = [item for item, _ in sorted_preds]
        
        # Binarize using manuscript's 3-threshold rule
        threshold = np.mean(list(actual_ratings.values()))
        relevant_items = set([item for item, rating in actual_ratings.items() 
                            if rating >= threshold - 0.1 and rating <= threshold + 0.1])
        
        hits = len([item for item in pred_items if item in relevant_items])
        return hits / k if k > 0 else 0.0

    def recall_at_k(self, actual_ratings, predicted_scores, k=10):
        """Calculate Recall@k with manuscript's methodology"""
        sorted_preds = sorted(predicted_scores.items(), key=lambda x: x[1], reverse=True)[:k]
        pred_items = set([item for item, _ in sorted_preds])
        
        # Same threshold as precision calculation
        threshold = np.mean(list(actual_ratings.values()))
        relevant_items = set([item for item, rating in actual_ratings.items() 
                            if rating >= threshold - 0.1 and rating <= threshold + 0.1])
        
        if not relevant_items:
            return 0.0
        return len(pred_items & relevant_items) / len(relevant_items)

    def ndcg_at_k(self, actual_ratings, predicted_scores, k=10):
        """Calculate NDCG@k with manuscript's relevance grading"""
        common = set(actual_ratings.keys()) & set(predicted_scores.keys())
        if not common:
            return 0.0
            
        y_true = [actual_ratings[item] for item in common]
        y_pred = [predicted_scores[item] for item in common]
        
        # Normalize and bin per manuscript
        y_true_norm = self.rating_scaler.fit_transform(np.array(y_true).reshape(-1, 1)).flatten()
        y_pred_norm = self.rating_scaler.transform(np.array(y_pred).reshape(-1, 1)).flatten()
        
        true_grades = np.digitize(y_true_norm, [0.33, 0.66])
        pred_grades = np.digitize(y_pred_norm, [0.33, 0.66])
        
        return ndcg_score([true_grades], [pred_grades], k=k)

    def mae_rmse(self, actual_ratings, predicted_scores):
        """Calculate MAE/RMSE with manuscript's 5-point scaling"""
        common = set(actual_ratings.keys()) & set(predicted_scores.keys())
        if not common:
            return 0.0, 0.0
            
        y_true = [actual_ratings[item] for item in common]
        y_pred = [predicted_scores[item] for item in common]
        
        # Scale to 1-5 range
        y_true_scaled = 1 + 4 * (np.array(y_true) - min(y_true)) / (max(y_true) - min(y_true))
        y_pred_scaled = 1 + 4 * (np.array(y_pred) - min(y_pred)) / (max(y_pred) - min(y_pred))
        
        return (mean_absolute_error(y_true_scaled, y_pred_scaled),
                np.sqrt(mean_squared_error(y_true_scaled, y_pred_scaled)))

    def plot_precision_recall_at_k(self, results_dict, k_values, model_name, save_path=None):
        """Generate manuscript-style plots with value annotations"""
        plt.figure(figsize=(12, 6))
        
        # Precision plot
        plt.subplot(1, 2, 1)
        precisions = [results_dict[f'Precision@{k}'] for k in k_values]
        sns.lineplot(x=k_values, y=precisions, marker='o')
        plt.title(f'{model_name} - Precision@k')
        plt.xlabel('k')
        plt.ylabel('Precision')
        plt.grid(True)
        
        # Add value labels
        for k, val in zip(k_values, precisions):
            plt.text(k, val, f'{val:.4f}', ha='center', va='bottom')

        # Recall plot
        plt.subplot(1, 2, 2)
        recalls = [results_dict[f'Recall@{k}'] for k in k_values]
        sns.lineplot(x=k_values, y=recalls, marker='s')
        plt.title(f'{model_name} - Recall@k')
        plt.xlabel('k')
        plt.ylabel('Recall')
        plt.grid(True)
        
        # Add value labels
        for k, val in zip(k_values, recalls):
            plt.text(k, val, f'{val:.4f}', ha='center', va='bottom')

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, format='pdf', bbox_inches='tight')
        plt.show()

    def load_predictions(self, user_id, pred_dir="results/predictions"):
        """Load saved predictions from hybrid system"""
        path = os.path.join(pred_dir, f"user_{user_id}_predictions.csv")
        df = pd.read_csv(path)
        return list(zip(df['itemId'], df['hybrid_score']))

    def comprehensive_evaluation(self, actual_ratings, predicted_scores, k_values=[5, 10, 15, 20]):
        """Main evaluation workflow from manuscript"""
        results = {}
        
        for k in k_values:
            results[f'Precision@{k}'] = self.precision_at_k(actual_ratings, predicted_scores, k)
            results[f'Recall@{k}'] = self.recall_at_k(actual_ratings, predicted_scores, k)
        
        results['F1_Score'] = f1_score(
            self._binarize(actual_ratings),
            self._binarize(predicted_scores)
        )
        results['NDCG'] = self.ndcg_at_k(actual_ratings, predicted_scores)
        results['MAE'], results['RMSE'] = self.mae_rmse(actual_ratings, predicted_scores)
        
        return results

    def _binarize(self, ratings_dict, tolerance=0.1):
        """Internal binarization per manuscript"""
        threshold = np.mean(list(ratings_dict.values()))
        return {
            item: int(threshold - tolerance <= rating <= threshold + tolerance)
            for item, rating in ratings_dict.items()
        }

if __name__ == "__main__":
    # Original test case remains valid
    evaluator = RecommenderEvaluator()
    actual = {1:4.5, 2:3.0, 3:5.0, 4:2.5, 5:4.0}
    predicted = {1:4.2, 2:3.1, 3:4.8, 4:2.8, 5:3.9}
    
    results = evaluator.comprehensive_evaluation(actual, predicted)
    evaluator.plot_precision_recall_at_k(results, [5,10], "Test Model")
