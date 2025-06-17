"""
Data Preprocessing Pipeline for Amazon E-commerce Dataset

This module implements probability-based imputation, feature engineering,
and user-level train/test splitting for the hybrid recommender system.
"""

import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings('ignore')


def load_amazon_dataset(filepath):
    """
    Load Amazon e-commerce dataset with automatic download if missing
    
    Args:
        filepath (str): Path to the CSV file
    
    Returns:
        pd.DataFrame: Raw dataset with product attributes
    """
    if not os.path.exists(filepath):
        print("Downloading dataset...")
        os.system(f'wget -O {filepath} https://github.com/aksharpandia/miniamazondata/raw/main/amazon_co-ecommerce_sample.csv')
    
    try:
        data = pd.read_csv(filepath)
        print(f"Dataset loaded successfully. Shape: {data.shape}")
        return data
    except Exception as e:
        print(f"Error loading dataset: {str(e)}")
        return None


def check_missing_values(data):
    """
    Check and display missing values in the dataset
    
    Args:
        data (pd.DataFrame): Input dataframe
    """
    print("Missing values per column:")
    for column in data.columns:
        nan_count = data[column].isnull().sum()
        if nan_count != 0:
            print(f"{column}: {nan_count}")


def drop_ineffective_columns(data):
    """
    Drop columns that are ineffective for recommendations
    
    Args:
        data (pd.DataFrame): Input dataframe
    
    Returns:
        pd.DataFrame: Dataframe with dropped columns
    """
    # Drop customer_questions_and_answers as ineffective and contains large null values
    if 'customer_questions_and_answers' in data.columns:
        data.drop('customer_questions_and_answers', axis=1, inplace=True)
        print("Dropped 'customer_questions_and_answers' column")
    
    # Drop number_of_answered_questions column as it has no effect on recommendations
    if 'number_of_answered_questions' in data.columns:
        data.drop('number_of_answered_questions', axis=1, inplace=True)
        print("Dropped 'number_of_answered_questions' column")
    
    return data


def probability_based_imputation(data):
    """
    Implement probability-based imputation for categorical features
    
    Args:
        data (pd.DataFrame): Input dataframe with missing values
    
    Returns:
        pd.DataFrame: Dataframe with imputed values
    """
    # Identify nominal (categorical) and numerical columns
    nominal_columns = data.select_dtypes(include=['object']).columns
    
    print(f"Processing {len(nominal_columns)} categorical columns for imputation")
    
    for column in nominal_columns:
        # Filter out non-missing values and calculate the distribution
        non_missing = data[column].dropna()
        distribution = non_missing.value_counts(normalize=True)
        
        # Number of missing values
        num_missing = data[column].isnull().sum()
        
        if num_missing > 0:
            # Randomly select values based on the distribution to fill missing values
            imputed_values = np.random.choice(
                distribution.index, 
                size=num_missing, 
                p=distribution.values
            )
            
            # Fill missing values in the DataFrame
            data.loc[data[column].isnull(), column] = imputed_values
            print(f"Imputed {num_missing} missing values in '{column}' column")
    
    return data


def assign_placeholder_ratings(data):
    """
    Implement manuscript's cold-start strategy using cosine similarity
    
    Args:
        data (pd.DataFrame): Input dataframe
    
    Returns:
        pd.DataFrame: Dataframe with placeholder ratings
    """
    # Calculate global mean rating
    global_mean = data['average_review_rating'].mean()
    
    # For users with <3 ratings
    for user_id in data[data.groupby('userId')['userId'].transform('count') < 3]['userId'].unique():
        user_data = data[data['userId'] == user_id]
        
        # Calculate similarity between items
        items = data[['itemId', 'price', 'average_review_rating', 
                     'manufacturer', 'category']].drop_duplicates()
        
        # Numeric features
        num_features = items[['price', 'average_review_rating']]
        num_features = MinMaxScaler().fit_transform(num_features)
        
        # Categorical features
        cat_features = pd.get_dummies(items[['manufacturer', 'category']])
        
        # Combined similarity
        similarity_matrix = cosine_similarity(np.hstack([num_features, cat_features]))
        
        # For each under-rated item
        for idx, row in user_data.iterrows():
            similar_items = items[
                similarity_matrix[row['itemId']] > 0.5
            ].index.tolist()
            
            if similar_items:
                placeholder = data.loc[similar_items, 'average_review_rating'].mean()
            else:
                placeholder = global_mean
                
            data.loc[idx, 'average_review_rating'] = placeholder
    
    return data


def encode_categorical_features(data):
    """
    Label encode categorical features
    
    Args:
        data (pd.DataFrame): Input dataframe
    
    Returns:
        pd.DataFrame: Dataframe with encoded features
    """
    le = LabelEncoder()
    
    # List of columns to encode
    columns_to_encode = [
        'average_review_rating',
        'uniq_id',
        'manufacturer',
        'price',
        'amazon_category_and_sub_category'
    ]
    
    for column in columns_to_encode:
        if column in data.columns:
            data[column] = le.fit_transform(data[column])
            print(f"Label encoded '{column}' column")
    
    return data


def create_item_id(data):
    """
    Create itemId from product_name using label encoding
    
    Args:
        data (pd.DataFrame): Input dataframe
    
    Returns:
        pd.DataFrame: Dataframe with itemId column
    """
    # Initialize LabelEncoder
    label_encoder = LabelEncoder()
    
    # Column to label encode
    column_to_encode = 'product_name'
    
    if column_to_encode in data.columns:
        # Fit and transform the specified column
        encoded_column = label_encoder.fit_transform(data[column_to_encode])
        
        # Add the encoded values as a new column next to product_name
        data.insert(
            data.columns.get_loc(column_to_encode) + 1, 
            'itemId', 
            encoded_column
        )
        print("Created 'itemId' column from 'product_name'")
    
    return data


def rename_columns(data):
    """
    Rename columns for consistency
    
    Args:
        data (pd.DataFrame): Input dataframe
    
    Returns:
        pd.DataFrame: Dataframe with renamed columns
    """
    # Rename uniq_id to userId for clarity
    data = data.rename(columns={'uniq_id': 'userId'})
    print("Renamed 'uniq_id' to 'userId'")
    
    return data


def engineer_features(data):
    """
    Complete feature engineering pipeline
    
    Args:
        data (pd.DataFrame): Input dataframe
    
    Returns:
        pd.DataFrame: Processed dataframe with engineered features
    """
    print("Starting feature engineering pipeline...")
    
    # Step 1: Drop ineffective columns
    data = drop_ineffective_columns(data)
    
    # Step 2: Apply probability-based imputation
    data = probability_based_imputation(data)

    # Step 3: Apply placeholder ratings for cold-start
    data = assign_placeholder_ratings(data)
    
    # Step 4: Encode categorical features
    data = encode_categorical_features(data)
    
    # Step 5: Create itemId
    data = create_item_id(data)
    
    # Step 6: Rename columns
    data = rename_columns(data)
    
    print("Feature engineering completed!")
    print(f"Final dataset shape: {data.shape}")
    print(f"Unique users: {data['userId'].nunique()}")
    print(f"Unique items: {data['itemId'].nunique()}")
    
    return data


def split_data(data, test_size=0.2, random_state=42):
    """
    Split data into train/test sets at user level
    
    Args:
        data (pd.DataFrame): Processed dataframe
        test_size (float): Proportion of users for test set
        random_state (int): Random seed for reproducibility
    
    Returns:
        tuple: (train_data, test_data) DataFrames
    """
    # Get unique users and sample test users
    users = data['userId'].unique()
    test_users = np.random.choice(users, 
                                size=int(len(users)*test_size), 
                                replace=False,
                                random_state=random_state)
    
    # Split data
    train_data = data[~data['userId'].isin(test_users)]
    test_data = data[data['userId'].isin(test_users)]
    
    print(f"Train/test split completed ({len(train_data)}/{len(test_data)} records)")
    print(f"Train users: {train_data['userId'].nunique()}")
    print(f"Test users: {test_data['userId'].nunique()}")
    
    return train_data, test_data


def main():
    """
    Main preprocessing pipeline execution
    """
    print("=== Amazon E-commerce Data Preprocessing Pipeline ===\n")
    
    # Load dataset
    filepath = "data/amazon_co-ecommerce_sample.csv"
    data = load_amazon_dataset(filepath)
    
    if data is not None:
        # Display initial info
        print(f"\nInitial dataset shape: {data.shape}")
        
        # Check missing values
        print("\n--- Missing Values Check ---")
        check_missing_values(data)
        
        # Process data
        print("\n--- Feature Engineering ---")
        processed_data = engineer_features(data)
        
        # Create train/test split
        print("\n--- Creating Train/Test Split ---")
        train_data, test_data = split_data(processed_data)
        
        # Create datasets for different models
        user_item_data = processed_data[['userId', 'itemId', 'average_review_rating']].copy()
        content_features = processed_data[[
            'userId', 'itemId', 'manufacturer', 'price', 
            'amazon_category_and_sub_category', 'average_review_rating'
        ]].copy()
        
        final_output = {
            'train_data': train_data,
            'test_data': test_data,
            'user_item_interactions': user_item_data,
            'content_features': content_features,
            'full_data': processed_data
        }
        
        # Display final info
        print("\n--- Final Processing Summary ---")
        print(f"Missing values after processing: {processed_data.isnull().sum().sum()}")
        
        return final_output
    
    return None


if __name__ == "__main__":
    # Execute main pipeline
    processed_data = main()
    
    if processed_data:
        print("\nData preprocessing completed successfully!")
        print("Available datasets:")
        for key, dataset in processed_data.items():
            print(f"  - {key}: {dataset.shape}")
    else:
        print("Data preprocessing failed!")

