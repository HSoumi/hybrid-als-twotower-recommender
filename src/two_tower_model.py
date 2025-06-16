"""
Two-Tower Content-Based Filtering Model Implementation

This module implements the Two-Tower neural network model using TensorFlow/Keras
for content-based recommendations as described in the manuscript.
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Input, Embedding, Dot, Flatten, Dense, 
                                   Concatenate, LayerNormalization)
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import warnings

warnings.filterwarnings('ignore')


class TwoTowerModel:
    """
    Two-Tower Content-Based Filtering Model using TensorFlow
    
    Architecture matches manuscript description:
    - User Tower: userId embedding
    - Item Tower: Combines itemId, manufacturer, category, and price features
    - Similarity: Dot product of user and item vectors (Equation 2)
    """
    
    def __init__(self, num_users, num_items, num_manufacturers, num_categories,
                 embedding_size=50, learning_rate=0.001):
        self.num_users = num_users
        self.num_items = num_items
        self.num_manufacturers = num_manufacturers
        self.num_categories = num_categories
        self.embedding_size = embedding_size
        self.learning_rate = learning_rate
        self.model = None
        self.scaler = MinMaxScaler()
        self.is_trained = False

    def _build_item_tower(self):
        """Simplified item tower without BERT per manuscript"""
        # Item ID embedding
        item_id_in = Input(shape=(1,), name='item_id_in')
        item_id_emb = Embedding(self.num_items, self.embedding_size, 
                              name='item_id_emb')(item_id_in)
        item_id_vec = Flatten()(item_id_emb)
        
        # Manufacturer embedding
        manufacturer_in = Input(shape=(1,), name='manufacturer_in')
        manufacturer_emb = Embedding(self.num_manufacturers, 8,
                                   name='manufacturer_emb')(manufacturer_in)
        manufacturer_vec = Flatten()(manufacturer_emb)
        
        # Category embedding
        category_in = Input(shape=(1,), name='category_in')
        category_emb = Embedding(self.num_categories, 8,
                               name='category_emb')(category_in)
        category_vec = Flatten()(category_emb)
        
        # Numeric features (price, avg rating)
        numeric_in = Input(shape=(2,), name='numeric_in')
        numeric_dense = Dense(16, activation='relu')(numeric_in)
        
        # Concatenate features 
        concat = Concatenate()([item_id_vec, manufacturer_vec, 
                              category_vec, numeric_dense])
        
        # Final item representation
        item_vec = Dense(self.embedding_size, activation=None)(concat)
        item_vec = LayerNormalization()(item_vec)
        
        return [item_id_in, manufacturer_in, category_in, numeric_in], item_vec

    def build_model(self):
        """Build complete Two-Tower architecture per manuscript"""
        # User tower
        user_in = Input(shape=(1,), name='user_in')
        user_emb = Embedding(self.num_users, self.embedding_size,
                           name='user_emb')(user_in)
        user_vec = Flatten()(user_emb)
        user_vec = LayerNormalization()(user_vec)
        
        # Item tower
        item_inputs, item_vec = self._build_item_tower()
        
        # Dot product similarity 
        dot_product = Dot(axes=1, name='dot_product')([user_vec, item_vec])
        
        # Compile model
        self.model = Model(inputs=[user_in] + item_inputs, outputs=dot_product)
        self.model.compile(optimizer=Adam(self.learning_rate),
                         loss='mean_squared_error',
                         metrics=['mae'])
        print("Two-Tower model built per manuscript specifications")
        return self.model

    def train(self, data, batch_size=256, epochs=10, validation_split=0.2):
        """Train model with full feature set as per manuscript"""
        try:
            if self.model is None:
                self.build_model()
            
            X = {
                'user_in': data['userId'].values,
                'item_id_in': data['itemId'].values,
                'manufacturer_in': data['manufacturer_id'].values,
                'category_in': data['category_id'].values,
                'numeric_in': data[['price', 'average_review_rating']].values
            }
            y = data['average_review_rating'].values
            
            # Train with validation split
            history = self.model.fit(X, y, batch_size=batch_size, epochs=epochs,
                                   validation_split=validation_split, verbose=1)
            self.is_trained = True
            return history
            
        except Exception as e:
            print(f"Training error: {str(e)}")
            return None

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
            if not self.is_trained:
                raise ValueError("Model not trained yet. Call train() first.")
            
            # Prepare input data
            user_ids = np.array([user_id] * len(all_items))
            item_ids = np.array(all_items)
            
            # Generate predictions
            predictions = self.model.predict([user_ids, item_ids], verbose=0)
            
            # Convert to list of tuples
            result = [(int(item_ids[i]), float(predictions[i])) 
                     for i in range(len(all_items))]
            
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
    
    def save_model(self, filepath):
        """Save the trained model"""
        if self.model:
            self.model.save(filepath)
            print(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load a pre-trained model"""
        try:
            self.model = tf.keras.models.load_model(filepath)
            self.is_trained = True
            print(f"Model loaded from {filepath}")
        except Exception as e:
            print(f"Error loading model: {str(e)}")


def hyperparameter_tuning(data, param_grid):
    """
    Perform hyperparameter tuning for Two-Tower model
    
    Parameters from Table 2 in manuscript:
    batch_size ∈ {32, 64, 128, 256, 512}
    epochs ∈ {50, 30, 20, 10, 5}
    
    Parameters:
    -----------
    data : pd.DataFrame
        Training data with columns: userId, itemId, average_review_rating
    param_grid : list
        List of parameter dictionaries to try
        
    Returns:
    --------
    dict : Best parameters found based on validation loss
    """
    best_params = None
    best_loss = float('inf')
    
    num_users = data['userId'].nunique()
    num_items = data['itemId'].nunique()
    num_manufacturers = data['manufacturer_id'].nunique()
    num_categories = data['category_id'].nunique()
    
    for params in param_grid:
        print(f"Testing parameters: {params}")
        
        model = TwoTowerModel(
            num_users=num_users,
            num_items=num_items,
            num_manufacturers=num_manufacturers,
            num_categories=num_categories,
            embedding_size=50,  # Fixed
            learning_rate=0.001  # Adam default
        )
        
        history = model.train(
            data,
            batch_size=params['batch_size'],
            epochs=params['epochs']
        )
        
        if history:
            final_loss = min(history.history['val_loss'])
            print(f"  Final validation loss: {final_loss:.4f}")
            if final_loss < best_loss:
                best_loss = final_loss
                best_params = params
                
        # Clear session to prevent memory issues
        tf.keras.backend.clear_session()
                
    print(f"Best parameters: {best_params} (loss: {best_loss:.4f})")
    return best_params


if __name__ == "__main__":
    print("Two-Tower Model module loaded successfully")
    
    # Parameter grid exactly matching Table 2 in manuscript
    param_grid = [
        {'batch_size': 32, 'epochs': 50},
        {'batch_size': 64, 'epochs': 30},
        {'batch_size': 128, 'epochs': 20},
        {'batch_size': 256, 'epochs': 10},
        {'batch_size': 512, 'epochs': 5}
    ]
    
    print("Available hyperparameter combinations:", len(param_grid))
