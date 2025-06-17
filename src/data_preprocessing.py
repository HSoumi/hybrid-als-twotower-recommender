"""
Data Preprocessing Pipeline for Amazon E-commerce Dataset

This module implements probability-based imputation, feature engineering,
and user-level train/test splitting for the hybrid recommender system.
"""

"""
Data Preprocessing Pipeline for Amazon E-commerce Dataset
Implements manuscript's cold-start strategy and user-level splitting
"""

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
import warnings

warnings.filterwarnings('ignore')

def load_amazon_dataset(filepath):
    """Load dataset with automatic download if missing"""
    if not os.path.exists(filepath):
        print("Downloading dataset...")
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        os.system(f'wget -O {filepath} https://github.com/aksharpandia/miniamazondata/raw/main/amazon_co-ecommerce_sample.csv')
    
    try:
        data = pd.read_csv(filepath)
        print(f"Dataset loaded. Shape: {data.shape}")
        return data
    except Exception as e:
        print(f"Error loading dataset: {str(e)}")
        return None

def check_missing_values(data):
    """Analyze missing values per column"""
    print("\nMissing values analysis:")
    print(data.isnull().sum())

def drop_ineffective_columns(data):
    """Remove columns per manuscript methodology"""
    cols_to_drop = ['customer_questions_and_answers', 'number_of_answered_questions']
    return data.drop([col for col in cols_to_drop if col in data.columns], axis=1)

def probability_based_imputation(data):
    """Implement manuscript's nominal feature imputation"""
    nominal_cols = data.select_dtypes(include=['object']).columns
    for col in nominal_cols:
        if data[col].isnull().sum() > 0:
            dist = data[col].dropna().value_counts(normalize=True)
            imputed = np.random.choice(dist.index, data[col].isnull().sum(), p=dist.values)
            data.loc[data[col].isnull(), col] = imputed
    return data

def assign_placeholder_ratings(data):
    """Cold-start handling per manuscript §3.1"""
    global_mean = data['average_review_rating'].mean()
    
    for user_id in data[data.groupby('userId')['userId'].transform('count') < 3]['userId'].unique():
        user_items = data[data['userId'] == user_id]
        items = data[['itemId', 'price', 'manufacturer', 'category']].drop_duplicates()
        
        # Feature engineering for similarity
        num_features = MinMaxScaler().fit_transform(items[['price']])
        cat_features = pd.get_dummies(items[['manufacturer', 'category']])
        similarity_matrix = cosine_similarity(num_features, cat_features)
        
        for idx, row in user_items.iterrows():
            similar_items = items[similarity_matrix[row['itemId']] > 0.5].index
            data.loc[idx, 'average_review_rating'] = data.loc[similar_items, 'average_review_rating'].mean() if len(similar_items) > 0 else global_mean
    
    return data

def encode_features(data):
    """Label encoding per manuscript preprocessing"""
    le = LabelEncoder()
    for col in ['average_review_rating', 'manufacturer', 'category']:
        data[col] = le.fit_transform(data[col])
    return data

def create_item_id(data):
    """Generate unique item IDs"""
    data['itemId'] = data.groupby('product_name').ngroup()
    return data

def split_data(data, test_size=0.2, random_state=42):
    """User-level split per manuscript evaluation protocol"""
    users = data['userId'].unique()
    test_users = np.random.choice(users, size=int(len(users)*test_size), 
                                replace=False, random_state=random_state)
    return (
        data[~data['userId'].isin(test_users)], 
        data[data['userId'].isin(test_users)]
    )

def main():
    """Main preprocessing pipeline"""
    print("=== Amazon Data Preprocessing Pipeline ===")
    
    # Path configuration
    raw_path = "data/amazon_co-ecommerce_sample.csv"
    processed_dir = "processed"
    
    # Load data
    data = load_amazon_dataset(raw_path)
    if data is None: return

    # Preprocessing steps
    data = (data.pipe(drop_ineffective_columns)
                .pipe(probability_based_imputation)
                .pipe(assign_placeholder_ratings)
                .pipe(encode_features)
                .pipe(create_item_id)
                .rename(columns={'uniq_id': 'userId'}))
    
    # Train/test split
    train_data, test_data = split_data(data)
    print(f"\nTrain shape: {train_data.shape}, Test shape: {test_data.shape}")
    
    # Save processed data
    os.makedirs(processed_dir, exist_ok=True)
    train_data.to_csv(f"{processed_dir}/train_data.csv", index=False)
    test_data.to_csv(f"{processed_dir}/test_data.csv", index=False)
    
    # Additional outputs per manuscript
    user_item = data[['userId', 'itemId', 'average_review_rating']]
    content_features = data[['itemId', 'manufacturer', 'price', 'category']]
    
    user_item.to_csv(f"{processed_dir}/user_item_interactions.csv", index=False)
    content_features.to_csv(f"{processed_dir}/content_features.csv", index=False)
    
    print("\nProcessing completed. Files saved in 'processed/' directory")

if __name__ == "__main__":
    main()
