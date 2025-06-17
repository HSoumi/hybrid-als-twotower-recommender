import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Embedding, Flatten, Dense, Concatenate, Dot, LayerNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import pickle
import os

class TwoTowerModel:
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
        item_id_in = Input(shape=(1,), name='item_id_in')
        item_id_emb = Embedding(self.num_items, self.embedding_size)(item_id_in)
        item_id_vec = Flatten()(item_id_emb)
        
        manufacturer_in = Input(shape=(1,), name='manufacturer_in')
        manufacturer_emb = Embedding(self.num_manufacturers, 8)(manufacturer_in)
        manufacturer_vec = Flatten()(manufacturer_emb)
        
        category_in = Input(shape=(1,), name='category_in')
        category_emb = Embedding(self.num_categories, 8)(category_in)
        category_vec = Flatten()(category_emb)
        
        numeric_in = Input(shape=(2,), name='numeric_in')
        numeric_dense = Dense(16, activation='relu')(numeric_in)
        
        concat = Concatenate()([item_id_vec, manufacturer_vec, category_vec, numeric_dense])
        item_vec = Dense(self.embedding_size)(concat)
        item_vec = LayerNormalization()(item_vec)
        
        return [item_id_in, manufacturer_in, category_in, numeric_in], item_vec

    def build_model(self):
        user_in = Input(shape=(1,), name='user_in')
        user_emb = Embedding(self.num_users, self.embedding_size)(user_in)
        user_vec = Flatten()(user_emb)
        user_vec = LayerNormalization()(user_vec)
        
        item_inputs, item_vec = self._build_item_tower()
        dot_product = Dot(axes=1)([user_vec, item_vec])
        
        self.model = Model(inputs=[user_in] + item_inputs, outputs=dot_product)
        self.model.compile(optimizer=Adam(self.learning_rate),
                         loss='mean_squared_error',
                         metrics=['mae'])
        return self.model

    def train(self, train_data, val_data=None, batch_size=256, epochs=10):
        if self.model is None:
            self.build_model()
        
        train_features = self._prepare_features(train_data)
        val_features = self._prepare_features(val_data) if val_data is not None else None
        
        early_stop = EarlyStopping(monitor='val_loss', patience=3)
        
        history = self.model.fit(
            train_features, train_data['average_review_rating'],
            validation_data=(val_features, val_data['average_review_rating']) if val_data else None,
            batch_size=batch_size,
            epochs=epochs,
            callbacks=[early_stop] if val_data else None
        )
        self.is_trained = True
        return history

    def _prepare_features(self, data):
        return {
            'user_in': data['userId'].values,
            'item_id_in': data['itemId'].values,
            'manufacturer_in': data['manufacturer_id'].values,
            'category_in': data['category_id'].values,
            'numeric_in': self.scaler.transform(data[['price', 'average_review_rating']])
        }

    def predict_for_user(self, user_id, item_features):
        inputs = {
            'user_in': np.full(len(item_features), user_id),
            'item_id_in': item_features['itemId'].values,
            'manufacturer_in': item_features['manufacturer_id'].values,
            'category_in': item_features['category_id'].values,
            'numeric_in': self.scaler.transform(item_features[['price', 'average_review_rating']])
        }
        return list(zip(item_features['itemId'], self.model.predict(inputs).flatten()))

    def save_model(self, model_path):
        self.model.save(model_path)
        with open(f"{model_path}_scaler.pkl", 'wb') as f:
            pickle.dump(self.scaler, f)

    @staticmethod
    def load_model(model_path):
        model = load_model(model_path)
        with open(f"{model_path}_scaler.pkl", 'rb') as f:
            scaler = pickle.load(f)
        
        new_model = TwoTowerModel(1,1,1,1)  # Dummy initialization
        new_model.model = model
        new_model.scaler = scaler
        new_model.is_trained = True
        return new_model

def hyperparameter_tuning(train_data, val_data, param_grid):
    best_params = None
    best_f1 = 0.0
    num_users = train_data['userId'].nunique()
    num_items = train_data['itemId'].nunique()
    num_manufacturers = train_data['manufacturer_id'].nunique()
    num_categories = train_data['category_id'].nunique()

    for params in param_grid:
        model = TwoTowerModel(num_users, num_items, num_manufacturers, num_categories)
        history = model.train(train_data, val_data, **params)
        
        f1_scores = []
        for user_id in val_data['userId'].unique()[:50]:
            actual = dict(zip(val_data[val_data['userId'] == user_id]['itemId'],
                            val_data[val_data['userId'] == user_id]['average_review_rating']))
            preds = model.predict_for_user(user_id, val_data)
            f1 = compute_f1_score(actual, dict(preds))
            f1_scores.append(f1)
        
        avg_f1 = np.mean(f1_scores)
        if avg_f1 > best_f1:
            best_f1 = avg_f1
            best_params = params.copy()
    
    return best_params

def compute_f1_score(actual, pred, k=10):
    actual_items = set(actual.keys())
    pred_items = set([item for item, _ in sorted(pred.items(), key=lambda x: x[1], reverse=True)[:k]])
    tp = len(actual_items & pred_items)
    precision = tp / k if k > 0 else 0
    recall = tp / len(actual_items) if len(actual_items) > 0 else 0
    return 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
