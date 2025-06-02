# Hybrid Recommender System Methodology

## Overview

This document outlines the methodology for developing a hybrid recommender system that combines Alternating Least Squares (ALS) collaborative filtering with Two-Tower content-based filtering for Amazon e-commerce recommendations.

## System Architecture

The hybrid system consists of three main components:

1. **ALS Collaborative Filtering Model** - Captures user-item interaction patterns
2. **Two-Tower Content-Based Model** - Leverages product features and metadata
3. **Adaptive Fusion Strategy** - Intelligently combines predictions from both models

## 1. Data Preprocessing Pipeline

### Dataset Characteristics
- **Source**: Amazon e-commerce sample dataset
- **Size**: 10,000 products with 17 attributes each
- **Features**: Product names, descriptions, categories, ratings, prices, reviews

### Preprocessing Steps

1. **Missing Value Analysis**
   - Identify null values across all columns
   - Categorize features into nominal and numerical types

2. **Probability-Based Imputation**
   ```
   P(imputed_value = x) = count(x) / total_non_null_count
   ```

   - Maintains original data distribution
   - Applied to categorical features with missing values

3. **Feature Engineering**
   - Label encoding for categorical variables
   - Creation of `itemId` from `product_name`
   - Renaming `uniq_id` to `userId` for consistency

4. **Data Cleaning**
   - Removal of ineffective features (`customer_questions_and_answers`, `number_of_answered_questions`)
   - Final validation and quality checks

## 2. ALS Collaborative Filtering Model

### Mathematical Foundation

The ALS model factorizes the sparse user-item interaction matrix into lower-dimensional matrices:
```
minimize: Σ(r_ui - u^T × i)² + λ(||u||² + ||i||²)
```
Where:
- `r_ui`: observed rating of user u for item i
- `u`, `i`: latent factor vectors for user and item
- `λ`: regularization parameter

### Model Parameters
- **Rank**: 10 (latent factor dimensions)
- **Max Iterations**: 10
- **Regularization Parameter**: 0.1
- **Cold Start Strategy**: Drop unknown users/items

### Implementation Details
- Uses PySpark for scalable matrix factorization
- Alternating optimization of user and item matrices
- Handles sparse data effectively

## 3. Two-Tower Content-Based Model


### Mathematical Formulation

Prediction formula:
```
ŷ_ui = u^T × i
```

Loss function (MSE):
```
L = (1/N) × Σ(r_ui - ŷ_ui)²
```

### Model Parameters
- **Embedding Size**: 50 dimensions
- **Optimizer**: Adam (learning rate: 0.001)
- **Batch Size**: 256
- **Epochs**: 10

## 4. Hybrid Fusion Strategy

### Adaptive Weighting Algorithm

The hybrid system uses F1-score based adaptive weighting:

```
IF als_f1 > twotower_f1:
combined_score = 0.8 × als_normalized + 0.2 × twotower_normalized
ELSE:
combined_score = 0.2 × als_normalized + 0.8 × twotower_normalized
```

### Normalization Process
1. Apply MinMaxScaler to both model predictions
2. Calculate F1 scores for each model on validation set
3. Apply adaptive weighting based on performance
4. Generate top-5 recommendations

### Final Selection
```
top_5_recommendations = sort(combined_scores, reverse=True)[:5]
```

## 5. Evaluation Methodology

### Metrics Implemented

1. **Precision@k and Recall@k** (k = 5, 10, 15, 20)
```
Precision@k = relevant_items_in_top_k / k
Recall@k = relevant_items_in_top_k / total_relevant_items
```

2. **F1 Score**
```
F1 = 2 × (Precision × Recall) / (Precision + Recall)
```


3. **Normalized Discounted Cumulative Gain (NDCG)**
```
NDCG@k = DCG@k / IDCG@k
```


4. **Mean Absolute Error (MAE) and Root Mean Square Error (RMSE)**
- Scaled to 1-5 rating range for interpretability

### Evaluation Process

1. **Data Splitting**: Reserved specific users for testing
2. **Ground Truth**: Actual user ratings from the dataset
3. **Prediction Generation**: All three models generate predictions
4. **Comparative Analysis**: Performance comparison across models
5. **Statistical Validation**: Multiple user profiles tested

## 6. Hyperparameter Tuning

### ALS Model Tuning
Tested combinations of:
- Max Iterations: [10, 15, 20]
- Rank: [5, 10, 12, 15, 20]
- Regularization: [0.05, 0.1, 0.2, 0.5]

### Two-Tower Model Tuning
Tested combinations of:
- Batch Size: [32, 64, 128, 256, 512]
- Epochs: [5, 10, 20, 30, 50]
- Embedding Size: [50] (fixed)

### Selection Criteria
Final parameters chosen based on:
- Best F1 score performance
- Computational efficiency
- Generalization across different users

## 7. Key Research Contributions

1. **Novel Integration**: First combination of ALS and Two-Tower models for e-commerce
2. **Adaptive Fusion**: F1-score based dynamic weighting strategy
3. **Comprehensive Evaluation**: Multi-metric assessment across diverse user profiles
4. **Practical Implementation**: Production-ready code for real-world deployment

## 8. Implementation Notes

### Dependencies
- PySpark 3.5.1 for ALS implementation
- TensorFlow 2.8.0 for Two-Tower neural network
- Scikit-learn 1.0.2 for evaluation metrics
- Pandas/NumPy for data processing

### Scalability Considerations
- Spark configuration optimized for large datasets
- Batch processing for efficient training
- Memory management for real-time inference

### Reproducibility
- Fixed random seeds for consistent results
- Documented preprocessing steps
- Version-controlled parameters and configurations

---

## References

This methodology supports the research paper: "A Hybrid Recommender System for Amazon E-commerce Combining ALS Collaborative Filtering and Two-Tower Content-Based Filtering"

For implementation details, see the `src/` directory containing modular Python code for each component.



