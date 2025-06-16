"""
Two-Tower Content-Based Filtering Model Implementation

This module implements the Two-Tower neural network model using TensorFlow/Keras
for content-based recommendations.
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Dot, Flatten, Dense
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
import warnings

warnings.filterwarnings('ignore')


class TwoTowerModel:
    """
    Two-Tower Content-Based Filtering Model using TensorFlow
    
    The model learns separate embeddings for users and items, then computes
    similarity using dot product between the embeddings.
    
    Parameters:
    -----------
    num_users : int
        Number of unique users in the dataset
    num_items : int
        Number of unique items in the dataset
    embedding_size : int
        Dimension of the embedding vectors (default: 50)
    learning_rate : float
        Learning rate for Adam optimizer (default: 0.001)
    """
    
    def __init__(self, num_users, num_items, embedding_size=50, learning_rate=0.001):
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_size = embedding_size
        self.learning_rate = learning_rate
        self.model = None
        self.scaler = MinMaxScaler()
        self.is_trained = False
        
    def build_model(self):
        """
        Build the Two-Tower neural network architecture
        
        Architecture:
        - User Tower: User embedding → Flatten
        - Item Tower: Item embedding → Flatten  
        - Similarity: Dot product of user and item vectors
        """
        # User tower
        user_input = Input(shape=(1,), name='user_input')
        user_embedding = Embedding(
            self.num_users, 
            self.embedding_size, 
            name='user_embedding'
        )(user_input)
        user_vec = Flatten(name='user_flatten')(user_embedding)
        
        # Item tower
        item_input = Input(shape=(6,), name='item_features') # price, manufacturer, etc
        item_dense = Dense(64, activation='relu')(item_input)
        item_embedding = Dense(embedding_size)(item_dense)
        item_embedding = Embedding(
            self.num_items, 
            self.embedding_size, 
            name='item_embedding'
        )(item_input)
        item_vec = Flatten(name='item_flatten')(item_embedding)
        
        # Dot product similarity
        dot_product = Dot(axes=1, name='dot_product')([user_vec, item_vec])
        
        # Create and compile model
        self.model = Model(inputs=[user_input, item_input], outputs=dot_product)
        self.model.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss='mean_squared_error',
            metrics=['mae']
        )
        
        print("Two-Tower model architecture built successfully")
        return self.model
    
    def train(self, data, batch_size=256, epochs=10, validation_split=0.2, verbose=1):
        """
        Train the Two-Tower model
        
        Parameters:
        -----------
        data : pd.DataFrame
            Training data with columns: userId, itemId, average_review_rating
        batch_size : int
            Training batch size (default: 256)
        epochs : int
            Number of training epochs (default: 10)
        validation_split : float
            Fraction of data to use for validation (default: 0.2)
        verbose : int
            Verbosity level (default: 1)
            
        Returns:
        --------
        History : Training history object
        """
        try:
            if self.model is None:
                self.build_model()
            
            # Prepare training data
            user_ids = data['userId'].values
            item_ids = data['itemId'].values
            ratings = data['average_review_rating'].values
            
            # Train the model
            print(f"Training Two-Tower model with {len(data)} samples...")
            history = self.model.fit(
                [user_ids, item_ids], 
                ratings,
                batch_size=batch_size,
                epochs=epochs,
                validation_split=validation_split,
                verbose=verbose
            )
            
            self.is_trained = True
            print("Two-Tower model training completed successfully")
            return history
            
        except Exception as e:
            print(f"Error training Two-Tower model: {str(e)}")
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
            result = [(int(item_ids[i]), float(predictions[i][0])) 
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
    
    Parameters:
    -----------
    data : pd.DataFrame
        Training data
    param_grid : list
        List of parameter dictionaries to try
        
    Returns:
    --------
    dict : Best parameters found
    """
    best_params = None
    best_loss = float('inf')
    
    num_users = data['userId'].nunique()
    num_items = data['itemId'].nunique()
    
    for params in param_grid:
        print(f"Testing parameters: {params}")
        
        model = TwoTowerModel(
            num_users=num_users,
            num_items=num_items,
            embedding_size=params.get('embedding_size', 50),
            learning_rate=params.get('learning_rate', 0.001)
        )
        
        history = model.train(
            data,
            batch_size=params.get('batch_size', 256),
            epochs=params.get('epochs', 10),
            verbose=0
        )
        
        if history:
            final_loss = min(history.history['val_loss'])
            if final_loss < best_loss:
                best_loss = final_loss
                best_params = params
                
    return best_params


if __name__ == "__main__":
    # Example usage
    print("Two-Tower Model module loaded successfully")
    
    # Example parameter grid for hyperparameter tuning
    param_grid = [
        {'batch_size': 32, 'epochs': 50, 'embedding_size': 50},
        {'batch_size': 64, 'epochs': 30, 'embedding_size': 50},
        {'batch_size': 128, 'epochs': 20, 'embedding_size': 50},
        {'batch_size': 256, 'epochs': 10, 'embedding_size': 50},
        {'batch_size': 512, 'epochs': 5, 'embedding_size': 50}
    ]
    
    print("Available hyperparameter combinations:", len(param_grid))

