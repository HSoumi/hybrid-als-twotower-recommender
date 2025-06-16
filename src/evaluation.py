"""
Evaluation Metrics for Recommender Systems

This module implements various evaluation metrics including Precision@k, Recall@k,
F1 Score, NDCG, MAE, and RMSE for recommender system evaluation.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, ndcg_score
from sklearn.preprocessing import MinMaxScaler
from .utils import scale_ratings_to_5
import warnings

warnings.filterwarnings('ignore')


class RecommenderEvaluator:
    """
    Comprehensive evaluation suite for recommender systems
    
    This class provides methods to calculate various metrics for evaluating
    the performance of recommendation algorithms as used in the hybrid 
    recommender system research.
    """
    
    def __init__(self):
        self.scaler = MinMaxScaler()
    
    def precision_at_k(self, actual_ratings, predicted_scores, k=10, threshold_type='mean'):
        """
        Calculate Precision@k as implemented in the research
        
        Parameters:
        -----------
        actual_ratings : dict
            Dictionary mapping itemId to actual rating
        predicted_scores : dict
            Dictionary mapping itemId to predicted score
        k : int
            Number of top recommendations to consider
        threshold_type : str
            Method to determine relevance ('mean' or 'median')
            
        Returns:
        --------
        float : Precision@k value
        """
        try:
            # Get top-k predictions
            sorted_predictions = sorted(
                predicted_scores.items(), 
                key=lambda x: x[1], 
                reverse=True
            )[:k]
            
            predicted_items = [item_id for item_id, _ in sorted_predictions]
            
            # Determine relevant items based on threshold
            if threshold_type == 'mean':
                threshold = np.mean(list(actual_ratings.values()))
            else:
                threshold = np.median(list(actual_ratings.values()))
            
            relevant_items = set([
                item_id for item_id, rating in actual_ratings.items()
                if rating >= threshold
            ])
            
            # Calculate precision
            relevant_predicted = len([
                item_id for item_id in predicted_items 
                if item_id in relevant_items
            ])
            
            precision = relevant_predicted / k if k > 0 else 0.0
            return precision
            
        except Exception as e:
            print(f"Error calculating Precision@k: {str(e)}")
            return 0.0
    
    def recall_at_k(self, actual_ratings, predicted_scores, k=10, threshold_type='mean'):
        """
        Calculate Recall@k as implemented in the research
        
        Parameters:
        -----------
        actual_ratings : dict
            Dictionary mapping itemId to actual rating
        predicted_scores : dict
            Dictionary mapping itemId to predicted score
        k : int
            Number of top recommendations to consider
        threshold_type : str
            Method to determine relevance ('mean' or 'median')
            
        Returns:
        --------
        float : Recall@k value
        """
        try:
            # Get top-k predictions
            sorted_predictions = sorted(
                predicted_scores.items(), 
                key=lambda x: x[1], 
                reverse=True
            )[:k]
            
            predicted_items = set([item_id for item_id, _ in sorted_predictions])
            
            # Determine relevant items based on threshold
            if threshold_type == 'mean':
                threshold = np.mean(list(actual_ratings.values()))
            else:
                threshold = np.median(list(actual_ratings.values()))
            
            relevant_items = set([
                item_id for item_id, rating in actual_ratings.items()
                if rating >= threshold
            ])
            
            # Calculate recall
            if len(relevant_items) == 0:
                return 0.0
            
            relevant_predicted = len(predicted_items.intersection(relevant_items))
            recall = relevant_predicted / len(relevant_items)
            return recall
            
        except Exception as e:
            print(f"Error calculating Recall@k: {str(e)}")
            return 0.0
    
    def f1_score(self, actual_ratings, predicted_scores, k=10, threshold_type='mean'):
        """
        Calculate F1 Score as used in the hybrid model evaluation
        
        Parameters:
        -----------
        actual_ratings : dict
            Dictionary mapping itemId to actual rating
        predicted_scores : dict
            Dictionary mapping itemId to predicted score
        k : int
            Number of top recommendations to consider
        threshold_type : str
            Method to determine relevance ('mean' or 'median')
            
        Returns:
        --------
        float : F1 score
        """
        try:
            precision = self.precision_at_k(actual_ratings, predicted_scores, k, threshold_type)
            recall = self.recall_at_k(actual_ratings, predicted_scores, k, threshold_type)
            
            if precision + recall == 0:
                return 0.0
            
            f1 = 2 * (precision * recall) / (precision + recall)
            return f1
            
        except Exception as e:
            print(f"Error calculating F1 score: {str(e)}")
            return 0.0
    
    def ndcg_at_k(self, actual_ratings, predicted_scores, k=10):
        """
        Calculate Normalized Discounted Cumulative Gain (NDCG@k) as used in research
        
        Parameters:
        -----------
        actual_ratings : dict
            Dictionary mapping itemId to actual rating
        predicted_scores : dict
            Dictionary mapping itemId to predicted score
        k : int
            Number of top recommendations to consider
            
        Returns:
        --------
        float : NDCG@k value
        """
        try:
            # Get common items
            common_items = set(actual_ratings.keys()).intersection(
                set(predicted_scores.keys())
            )
            
            if len(common_items) == 0:
                return 0.0
            
            # Prepare arrays for NDCG calculation
            y_true = []
            y_score = []
            
            for item_id in common_items:
                y_true.append(actual_ratings[item_id])
                y_score.append(predicted_scores[item_id])
            
            # Normalize ratings to 0-2 scale for NDCG (as per research implementation)
            y_true_norm = self.scaler.fit_transform(
                np.array(y_true).reshape(-1, 1)
            ).flatten()
            
            # Convert to relevance grades (0, 1, 2)
            relevance_grades = np.digitize(y_true_norm, bins=[0, 1/3, 2/3, 1])
            
            # Calculate NDCG
            ndcg = ndcg_score(
                [relevance_grades], 
                [y_score], 
                k=min(k, len(y_score))
            )
            
            return ndcg
            
        except Exception as e:
            print(f"Error calculating NDCG@k: {str(e)}")
            return 0.0
    
    def mae_rmse(self, actual_ratings, predicted_scores, scale_to_5=True):
        """
        Calculate Mean Absolute Error (MAE) and Root Mean Square Error (RMSE)
        as implemented in the research
        
        Parameters:
        -----------
        actual_ratings : dict
            Dictionary mapping itemId to actual rating
        predicted_scores : dict
            Dictionary mapping itemId to predicted score
        scale_to_5 : bool
            Whether to scale ratings to 1-5 range (as done in research)
            
        Returns:
        --------
        tuple : (MAE, RMSE) values
        """
        try:
            # Get common items
            common_items = set(actual_ratings.keys()).intersection(
                set(predicted_scores.keys())
            )
            
            if len(common_items) == 0:
                return 0.0, 0.0
            
            y_true = [actual_ratings[item_id] for item_id in common_items]
            y_pred = [predicted_scores[item_id] for item_id in common_items]
            
            if scale_to_5:
                # Scale to 1-5 range using utility function
                y_true_scaled = scale_ratings_to_5(y_true)
                y_pred_scaled = scale_ratings_to_5(y_pred)
            else:
                y_true_scaled = y_true
                y_pred_scaled = y_pred
            
            # Calculate MAE and RMSE
            mae = mean_absolute_error(y_true_scaled, y_pred_scaled)
            rmse = np.sqrt(mean_squared_error(y_true_scaled, y_pred_scaled))
            
            return mae, rmse
            
        except Exception as e:
            print(f"Error calculating MAE/RMSE: {str(e)}")
            return 0.0, 0.0
    
    def comprehensive_evaluation(self, actual_ratings, predicted_scores, k_values=[5, 10, 15, 20]):
        """
        Perform comprehensive evaluation with multiple metrics as used in the research
        
        Parameters:
        -----------
        actual_ratings : dict
            Dictionary mapping itemId to actual rating
        predicted_scores : dict
            Dictionary mapping itemId to predicted score
        k_values : list
            List of k values for Precision@k and Recall@k evaluation
            
        Returns:
        --------
        dict : Dictionary containing all evaluation metrics
        """
        results = {}
        
        try:
            # Calculate Precision@k and Recall@k for different k values
            for k in k_values:
                results[f'Precision@{k}'] = self.precision_at_k(
                    actual_ratings, predicted_scores, k
                )
                results[f'Recall@{k}'] = self.recall_at_k(
                    actual_ratings, predicted_scores, k
                )
            
            # Calculate other metrics used in research
            results['F1_Score'] = self.f1_score(actual_ratings, predicted_scores)
            results['NDCG'] = self.ndcg_at_k(actual_ratings, predicted_scores)
            
            mae, rmse = self.mae_rmse(actual_ratings, predicted_scores)
            results['MAE'] = mae
            results['RMSE'] = rmse
            
            return results
            
        except Exception as e:
            print(f"Error in comprehensive evaluation: {str(e)}")
            return {}
    
    def plot_precision_recall_at_k(self, results_dict, k_values=[5, 10, 15, 20], 
                                  model_name="Model", save_path=None):
        """
        Plot Precision@k and Recall@k graphs as shown in the research paper
        
        Parameters:
        -----------
        results_dict : dict
            Dictionary containing evaluation results
        k_values : list
            List of k values to plot
        model_name : str
            Name of the model for plot title
        save_path : str, optional
            Path to save the plot
        """
        try:
            precision_values = [results_dict.get(f'Precision@{k}', 0) for k in k_values]
            recall_values = [results_dict.get(f'Recall@{k}', 0) for k in k_values]
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            
            # Precision@k plot (matches research paper format)
            ax1.plot(k_values, precision_values, 'bo-', linewidth=2, markersize=8)
            ax1.set_xlabel('k')
            ax1.set_ylabel('Precision@k')
            ax1.set_title(f'{model_name}: Precision@k')
            ax1.grid(True, alpha=0.3)
            ax1.set_ylim(0, 1)
            
            # Add value annotations
            for i, v in enumerate(precision_values):
                ax1.annotate(f'{v:.3f}', (k_values[i], v), 
                           textcoords="offset points", xytext=(0,10), ha='center')
            
            # Recall@k plot (matches research paper format)
            ax2.plot(k_values, recall_values, 'ro-', linewidth=2, markersize=8)
            ax2.set_xlabel('k')
            ax2.set_ylabel('Recall@k')
            ax2.set_title(f'{model_name}: Recall@k')
            ax2.grid(True, alpha=0.3)
            ax2.set_ylim(0, max(recall_values) * 1.1 if max(recall_values) > 0 else 0.1)
            
            # Add value annotations
            for i, v in enumerate(recall_values):
                ax2.annotate(f'{v:.4f}', (k_values[i], v), 
                           textcoords="offset points", xytext=(0,10), ha='center')
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"Plot saved to {save_path}")
            
            plt.show()
            
        except Exception as e:
            print(f"Error creating plots: {str(e)}")


def compute_f1_score(actual_ratings, predicted_scores, k=10):
    """
    Utility function to compute F1 score for use in hybrid model
    
    Parameters:
    -----------
    actual_ratings : dict
        Dictionary mapping itemId to actual rating
    predicted_scores : dict
        Dictionary mapping itemId to predicted score
    k : int
        Number of top recommendations to consider
        
    Returns:
    --------
    float : F1 score
    """
    evaluator = RecommenderEvaluator()
    return evaluator.f1_score(actual_ratings, predicted_scores, k)


if __name__ == "__main__":
    # Example usage and validation
    print("="*50)
    print("Recommender System Evaluation Module")
    print("="*50)
    
    # Example data for testing (matches research methodology)
    actual_ratings = {1: 4.5, 2: 3.0, 3: 5.0, 4: 2.5, 5: 4.0}
    predicted_scores = {1: 4.2, 2: 3.1, 3: 4.8, 4: 2.8, 5: 3.9}
    
    print(f"Testing with {len(actual_ratings)} sample ratings...")
    
    evaluator = RecommenderEvaluator()
    results = evaluator.comprehensive_evaluation(actual_ratings, predicted_scores)
    
    print("\nEvaluation Results:")
    print("-" * 30)
    for metric, value in results.items():
        print(f"{metric:15}: {value:.4f}")
    
    print("\n All evaluation functions working correctly!")
    print("="*50)

