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
    - Item Tower: Combines itemId, manufacturer, category, price, and text features
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
        """Construct item tower with metadata features as per manuscript"""
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
        
        # Text features (BERT embeddings)
        text_in = Input(shape=(768,), name='text_in')
        text_dense = Dense(64, activation='relu')(text_in)
        
        # Concatenate all item features
        concat = Concatenate()([item_id_vec, manufacturer_vec, category_vec,
                              numeric_dense, text_dense])
        
        # Final item representation
        item_vec = Dense(self.embedding_size, activation=None)(concat)
        item_vec = LayerNormalization()(item_vec)
        
        return [item_id_in, manufacturer_in, category_in, numeric_in, text_in], item_vec

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
        
        # Dot product similarity (Equation 2)
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
            
            # Prepare features
            X = {
                'user_in': data['userId'].values,
                'item_id_in': data['itemId'].values,
                'manufacturer_in': data['manufacturer_id'].values,
                'category_in': data['category_id'].values,
                'numeric_in': data[['price', 'average_review_rating']].values,
                'text_in': np.vstack(data['bert_768'].values)
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

    def predict_for_user(self, user_id, item_features):
        """Generate predictions for a user across items"""
        if not self.is_trained:
            raise ValueError("Model not trained")
            
        # Prepare input data
        inputs = {
            'user_in': np.full(len(item_features), user_id),
            'item_id_in': item_features['itemId'].values,
            'manufacturer_in': item_features['manufacturer_id'].values,
            'category_in': item_features['category_id'].values,
            'numeric_in': item_features[['price', 'avg_rating']].values,
            'text_in': np.vstack(item_features['bert_768'].values)
        }
        
        predictions = self.model.predict(inputs, verbose=0)
        return list(zip(item_features['itemId'], predictions.flatten()))


def f1_based_hyperparameter_tuning(model_class, data, param_grid, 
                                  test_size=0.2, random_state=42):
    """
    Generalized F1-based hyperparameter tuning (reusable for ALS/Two-Tower)
    
    Parameters:
    -----------
    model_class : class
        Model class to instantiate (TwoTowerModel or ALSModel)
    data : pd.DataFrame
        Training data with required features
    param_grid : list
        List of parameter dictionaries to try
    test_size : float
        Size of validation set for F1 calculation
    random_state : int
        Random seed for reproducibility
        
    Returns:
    --------
    dict : Best parameters based on F1 score
    """
    best_params = None
    best_f1 = 0
    
    # Split data for validation
    train_data, val_data = train_test_split(data, test_size=test_size, 
                                          random_state=random_state)
    
    for params in param_grid:
        print(f"Testing parameters: {params}")
        
        try:
            # Initialize model with current params
            model = model_class(**params)
            model.train(train_data)
            
            # Evaluate on validation set
            f1_scores = []
            for user_id in val_data['userId'].unique()[:50]:  # Sample users
                actual = val_data[val_data['userId'] == user_id]
                items = actual[['itemId', 'manufacturer_id', 'category_id',
                              'price', 'avg_rating', 'bert_768']]
                preds = model.predict_for_user(user_id, items)
                pred_dict = {item: score for item, score in preds}
                actual_dict = dict(zip(actual['itemId'], actual['rating']))
                f1 = compute_f1_score(actual_dict, pred_dict)
                f1_scores.append(f1)
                
            mean_f1 = np.mean(f1_scores)
            print(f"Mean F1@10: {mean_f1:.4f}")
            
            if mean_f1 > best_f1:
                best_f1 = mean_f1
                best_params = params.copy()
                
        except Exception as e:
            print(f"Skipping parameters due to error: {str(e)}")
            
    print(f"\nBest F1@10: {best_f1:.4f} with params: {best_params}")
    return best_params


# Hyperparameter grid matching manuscript Table 2
TWO_TOWER_PARAM_GRID = [
    {'batch_size': 32, 'epochs': 50},
    {'batch_size': 64, 'epochs': 30},
    {'batch_size': 128, 'epochs': 20},
    {'batch_size': 256, 'epochs': 10},
    {'batch_size': 512, 'epochs': 5}
]


if __name__ == "__main__":
    print("Two-Tower Model with F1-based tuning ready")
