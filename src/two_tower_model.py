"""
Two-Tower Content-Based Filtering Model Implementation
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model, save_model
from tensorflow.keras.layers import Input, Embedding, Flatten, Dense, Concatenate, Dot, LayerNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import pickle
import os

class TwoTowerModel:
    """
    Two-Tower Model Architecture per Manuscript §4.3
    - User Tower: userId embedding
    - Item Tower: itemId + manufacturer + category + numeric features
    - Dot product similarity with layer normalization
    """
    
    def __init__(self, num_users, num_items, num_manufacturers, num_categories,
                 embedding_size=50, learning_rate=0.001):
        # Initialize based on dataset statistics
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
        """Construct item tower per manuscript equations"""
        # Item ID embedding
        item_id_in = Input(shape=(1,), name='item_id_in')
        item_id_emb = Embedding(self.num_items, self.embedding_size)(item_id_in)
        item_id_vec = Flatten()(item_id_emb)
        
        # Manufacturer embedding (8-dim per manuscript)
        manufacturer_in = Input(shape=(1,), name='manufacturer_in')
        manufacturer_emb = Embedding(self.num_manufacturers, 8)(manufacturer_in)
        manufacturer_vec = Flatten()(manufacturer_emb)
        
        # Category embedding (8-dim per manuscript)
        category_in = Input(shape=(1,), name='category_in')
        category_emb = Embedding(self.num_categories, 8)(category_in)
        category_vec = Flatten()(category_emb)
        
        # Numeric features (price + rating)
        numeric_in = Input(shape=(2,), name='numeric_in')
        numeric_dense = Dense(16, activation='relu')(numeric_in)  # Manuscript §4.3
        
        # Concatenate all features
        concat = Concatenate()([item_id_vec, manufacturer_vec, category_vec, numeric_dense])
        
        # Final item representation
        item_vec = Dense(self.embedding_size)(concat)
        item_vec = LayerNormalization()(item_vec)  # Equation 2 normalization
        
        return [item_id_in, manufacturer_in, category_in, numeric_in], item_vec

    def build_model(self):
        """Build complete architecture per manuscript"""
        # User tower
        user_in = Input(shape=(1,), name='user_in')
        user_emb = Embedding(self.num_users, self.embedding_size)(user_in)
        user_vec = Flatten()(user_emb)
        user_vec = LayerNormalization()(user_vec)  # Equation 1 normalization
        
        # Item tower
        item_inputs, item_vec = self._build_item_tower()
        
        # Dot product similarity
        dot_product = Dot(axes=1)([user_vec, item_vec])
        
        # Compile model
        self.model = Model(inputs=[user_in] + item_inputs, outputs=dot_product)
        self.model.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss='mean_squared_error',
            metrics=['mae']
        )
        return self.model

    def train(self, train_data, val_data=None, batch_size=256, epochs=10):
        """Training process with early stopping"""
        if self.model is None:
            self.build_model()
        
        # Prepare features
        train_features = self._prepare_features(train_data)
        val_features = self._prepare_features(val_data) if val_data else None
        
        # Callbacks per manuscript methodology
        callbacks = []
        if val_data:
            callbacks.append(
                EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
            )
            callbacks.append(
                ModelCheckpoint('tmp_best.keras', save_best_only=True, monitor='val_loss')
            )
        
        # Train model
        history = self.model.fit(
            train_features,
            train_data['average_review_rating'],
            validation_data=(val_features, val_data['average_review_rating']) if val_data else None,
            batch_size=batch_size,
            epochs=epochs,
            callbacks=callbacks,
            verbose=1
        )
        self.is_trained = True
        return history

    def _prepare_features(self, data):
        """Feature preprocessing per manuscript §3.1"""
        if data is None:
            return None
            
        return {
            'user_in': data['userId'].values,
            'item_id_in': data['itemId'].values,
            'manufacturer_in': data['manufacturer_id'].values,
            'category_in': data['category_id'].values,
            'numeric_in': self.scaler.fit_transform(data[['price', 'average_review_rating']])
        }

    def predict_for_user(self, user_id, item_features):
        """Generate predictions per manuscript evaluation protocol"""
        inputs = {
            'user_in': np.full(len(item_features), user_id),
            'item_id_in': item_features['itemId'].values,
            'manufacturer_in': item_features['manufacturer_id'].values,
            'category_in': item_features['category_id'].values,
            'numeric_in': self.scaler.transform(item_features[['price', 'average_review_rating']])
        }
        predictions = self.model.predict(inputs, verbose=0)
        return list(zip(item_features['itemId'], predictions.flatten()))

    def save_model(self, model_path="models/twotower.keras"):
        """Save model with Keras format per manuscript"""
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        save_model(self.model, model_path)
        with open(f"{model_path}_scaler.pkl", 'wb') as f:
            pickle.dump(self.scaler, f)

    @classmethod
    def load_model(cls, model_path="models/twotower.keras"):
        """Load saved model with scaler"""
        model = load_model(model_path)
        with open(f"{model_path}_scaler.pkl", 'rb') as f:
            scaler = pickle.load(f)
        
        # Reconstruct model with dummy initialization
        loaded_model = cls(1, 1, 1, 1)
        loaded_model.model = model
        loaded_model.scaler = scaler
        loaded_model.is_trained = True
        return loaded_model

def hyperparameter_tuning(train_data, param_grid, val_size=0.2, random_state=42):
    """
    F1-based hyperparameter tuning per manuscript §4.3
    Returns best parameters from grid (Table 2)
    """
    best_params = None
    best_f1 = 0.0
    
    # Split training data into train/validation
    train_users = train_data['userId'].unique()
    val_users = np.random.choice(train_users, size=int(len(train_users)*val_size), 
                               replace=False, random_state=random_state)
    
    train_sub = train_data[~train_data['userId'].isin(val_users)]
    val_sub = train_data[train_data['userId'].isin(val_users)]
    
    # Get dataset dimensions
    num_users = train_sub['userId'].nunique()
    num_items = train_sub['itemId'].nunique()
    num_manufacturers = train_sub['manufacturer_id'].nunique()
    num_categories = train_sub['category_id'].nunique()
    
    for params in param_grid:
        print(f"\nTesting parameters: {params}")
        try:
            model = TwoTowerModel(
                num_users=num_users,
                num_items=num_items,
                num_manufacturers=num_manufacturers,
                num_categories=num_categories,
                embedding_size=50,  # Fixed per manuscript
                learning_rate=0.001  # Adam default
            )
            
            history = model.train(
                train_sub,
                val_sub,
                batch_size=params['batch_size'],
                epochs=params['epochs']
            )
            
            # Evaluate on validation subset
            f1_scores = []
            sample_users = val_sub['userId'].unique()[:50]  # 5% sample per manuscript
            for user_id in sample_users:
                actual = dict(zip(
                    val_sub[val_sub['userId'] == user_id]['itemId'],
                    val_sub[val_sub['userId'] == user_id]['average_review_rating']
                ))
                items = val_sub[['itemId', 'manufacturer_id', 'category_id', 
                               'price', 'average_review_rating']].drop_duplicates()
                preds = model.predict_for_user(user_id, items)
                f1 = compute_f1_score(actual, dict(preds), k=10)  # F1@10 per manuscript
                f1_scores.append(f1)
            
            avg_f1 = np.mean(f1_scores)
            print(f"  Avg F1@10: {avg_f1:.4f}")
            
            if avg_f1 > best_f1:
                best_f1 = avg_f1
                best_params = params.copy()
                print(f"  New best F1@10: {best_f1:.4f}")
                
        except Exception as e:
            print(f"  Error with params {params}: {str(e)}")
            continue
            
    return best_params

def compute_f1_score(actual, pred, k=10):
    """Calculate F1@10 per manuscript evaluation metrics"""
    actual_items = set(actual.keys())
    pred_items = set([item for item, _ in sorted(pred.items(), key=lambda x: x[1], reverse=True)[:k]])
    tp = len(actual_items & pred_items)
    precision = tp / k if k > 0 else 0
    recall = tp / len(actual_items) if len(actual_items) > 0 else 0
    return 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

if __name__ == "__main__":
    # Example usage per manuscript workflow
    print("=== Two-Tower Model Training ===")
    
    # Load preprocessed data (from data_preprocessing.py)
    train_data = pd.read_csv("processed/train_data.csv")
    
    # Manuscript Table 2 parameter grid
    param_grid = [
        {'batch_size': 32, 'epochs': 50},
        {'batch_size': 64, 'epochs': 30},
        {'batch_size': 128, 'epochs': 20},
        {'batch_size': 256, 'epochs': 10},
        {'batch_size': 512, 'epochs': 5}
    ]
    
    # Perform hyperparameter tuning
    print("\n--- Starting F1-based Hyperparameter Tuning ---")
    best_params = hyperparameter_tuning(train_data, param_grid)
    print(f"\nBest parameters: {best_params}")
    
    # Train final model with best params
    print("\n--- Training Final Model ---")
    model = TwoTowerModel(
        num_users=train_data['userId'].nunique(),
        num_items=train_data['itemId'].nunique(),
        num_manufacturers=train_data['manufacturer_id'].nunique(),
        num_categories=train_data['category_id'].nunique()
    )
    model.train(train_data, epochs=best_params['epochs'], batch_size=best_params['batch_size'])
    
    # Save model per manuscript specifications
    model.save_model("models/twotower.keras")
    print("\nModel saved to models/twotower.keras")

