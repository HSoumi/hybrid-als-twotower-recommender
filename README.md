# Hybrid Recommender System: ALS + Two-Tower for Amazon E-commerce

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

This repository contains the implementation of a hybrid recommender system that combines Alternating Least Squares (ALS) collaborative filtering with Two-Tower content-based filtering for Amazon e-commerce recommendations.

## Research Paper

**Title:** "A Hybrid Recommender System for Amazon E-commerce Combining ALS Collaborative Filtering and Two-Tower Content-Based Filtering"

**Abstract:**
Recommender Systems are vital for enhancing user experience on e-commerce platforms like Amazon, but traditional approaches face challenges such as data sparsity. This paper proposes a hybrid recommender system combining Alternating Least Squares (ALS) collaborative filtering with a two-tower content-based filtering model for Amazon e-commerce. The ALS model utilizes user-item interactions, while the two-tower model incorporates product description features. The system adaptively combines predictions from both models using a weighted approach. Experiments on a large-scale Amazon dataset demonstrate that the proposed method outperforms state-of-the-art baselines, offering an improved solution for recommendation quality and diversity on the Amazon e-commerce platform.

## Methodology Overview

### 1. Data Preprocessing Pipeline
- **Input**: Amazon e-commerce dataset (10,000 products, 17 attributes)
- **Process**: 
  - Probability-based imputation for missing values
  - Label encoding for categorical features  
  - Feature engineering for user-item interactions
- **Output**: Clean, processed dataset ready for model training

### 2. ALS Collaborative Filtering
- **Algorithm**: Alternating Least Squares (PySpark implementation)
- **Parameters**: rank=10, maxIter=10, regParam=0.1
- **Purpose**: Captures user-item interaction patterns
- **Output**: User and item latent factor matrices

### 3. Two-Tower Content-Based Model
- **Architecture**: Neural embedding-based two-tower design
- **Implementation**: TensorFlow/Keras
- **Features**: Product descriptions, categories, manufacturer data
- **Output**: Content-based similarity scores

### 4. Hybrid Fusion Strategy
- **Approach**: Adaptive weighted combination
- **Logic**: 80-20 weighting based on individual model F1 scores
- **Decision Rule**: Higher-performing model gets 80% weight
- **Final Output**: Top-5 personalized recommendations

## Key Results

- **User 1 (ID: 462)**: Hybrid F1=0.6396, NDCG=0.9775, MAE=1.0372
- **User 2 (ID: 9435)**: Hybrid F1=0.6177, NDCG=0.9826, MAE=1.7688
- **Performance**: Consistent outperformance over individual models

## Repository Structure
├── src/ # Core implementation modules
├── notebooks/ # Jupyter notebooks for exploration
├── data/ # Dataset and preprocessing info
├── results/ # Experimental results and figures
└── docs/ # Additional documentation

undefined


## Quick Start

1. **Install Dependencies**
pip install -r requirements.txt


2. **Run Data Preprocessing**
python src/data_preprocessing.py


3. **Train Models**
python src/als_model.py
python src/two_tower_model.py


4. **Generate Hybrid Recommendations**
python src/hybrid_system.py


## Reproducibility

All experiments can be reproduced using the provided notebooks:
1. `01_data_exploration.ipynb` - Dataset analysis and preprocessing
2. `02_methodology_overview.ipynb` - Model implementation details  
3. `03_results_analysis.ipynb` - Results and evaluation

## Citation
@article{hazra2025hybrid,
title={A Hybrid Recommender System for Amazon E-commerce Combining ALS Collaborative Filtering and Two-Tower Content-Based Filtering},
author={Hazra, Soumi},
journal={Submitted to PeerJ Computer Science},
year={2025}
}


## License

MIT License - see LICENSE file for details.

