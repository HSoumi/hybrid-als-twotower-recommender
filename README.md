# Hybrid Recommender System: ALS + Two-Tower for Amazon E-commerce

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.15586080.svg)](https://zenodo.org/records/15586080)

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
  - Label encoding for categorical features (manufacturer, price, categories, ratings)
  - Feature engineering (userId from uniq_id, itemId from product_name)
- **Output**: Clean, processed dataset with 0 missing values

### 2. ALS Collaborative Filtering
- **Algorithm**: Alternating Least Squares (PySpark implementation)
- **Parameters**: rank=10, maxIter=10, regParam=0.1
- **Purpose**: Captures user-item interaction patterns through matrix factorization
- **Output**: User and item latent factor matrices

### 3. Two-Tower Content-Based Model
- **Architecture**: Neural embedding-based two-tower design
- **Implementation**: TensorFlow/Keras with 50-dimensional embeddings
- **Features**: Product metadata (manufacturer, categories, descriptions)
- **Output**: Content-based similarity scores

### 4. Hybrid Fusion Strategy
- **Approach**: Adaptive weighted combination using F1-score based weighting
- **Logic**: 80-20 weighting favoring the better-performing model
- **Decision Rule**: IF ALS_F1 > TwoTower_F1: 80% ALS + 20% Two-Tower, ELSE: 20% ALS + 80% Two-Tower
- **Final Output**: Top-5 personalized recommendations

## Key Results

- **User 1 (ID: 462)**: Hybrid F1=0.6396, NDCG=0.9775, MAE=1.0372
- **User 2 (ID: 9435)**: Hybrid F1=0.6177, NDCG=0.9826, MAE=1.7688
- **Performance**: Consistent outperformance over individual models across multiple evaluation metrics

## Repository Structure
```
├── src/ # Core implementation modules
│ ├── init.py
│ ├── data_preprocessing.py # Data cleaning and feature engineering
│ ├── als_model.py # ALS collaborative filtering implementation
│ ├── two_tower_model.py # Two-Tower neural content-based filtering
│ ├── hybrid_system.py # Hybrid model combining both approaches
│ ├── evaluation.py # Comprehensive evaluation metrics
│ └── utils.py # Utility functions and helpers
├── data/ # Dataset documentation and access instructions
│ └── README.md # Detailed dataset information
├── docs/ # Additional documentation
│ └── methodology.md # Complete methodology documentation
├── requirements.txt # Python dependencies
├── LICENSE # MIT license
└── README.md # This file
```

## Quick Start

### 1. Installation
```
git clone https://github.com/yourusername/hybrid-als-twotower-recommender.git
cd hybrid-als-twotower-recommender
pip install -r requirements.txt
```

### 2. Data Setup

Follow instructions in data/README.md to obtain the Amazon dataset
Place the CSV file in the data/ directory


### 3. Run Complete Pipeline

Data preprocessing
```
python src/data_preprocessing.py
```

Train individual models (optional - for experimentation)
```
python -c "from src.als_model import ALSModel; print('ALS module ready')"
python -c "from src.two_tower_model import TwoTowerModel; print('Two-Tower module ready')"
```

Run hybrid system
```
python -c "from src.hybrid_system import HybridRecommendationSystem; print('Hybrid system ready')"
```

### 4. Evaluate Results
Comprehensive evaluation:
```
python -c "from src.evaluation import RecommenderEvaluator; print('Evaluation module ready')"
```

## Module Usage Examples

### Data Preprocessing
```
from src.data_preprocessing import main
```

Run complete preprocessing pipeline:
```
processed_data = main()
if processed_data:
print("Preprocessing completed successfully!")
```

### Hybrid Recommendations
```
from src.hybrid_system import HybridRecommendationSystem
```

Initialize and train hybrid system:
```
hybrid_system = HybridRecommendationSystem()
hybrid_system.train_models(training_data)
```

Get recommendations for a user:
```
recommendations = hybrid_system.get_hybrid_recommendations(
user_id=462,
all_items=item_list,
top_k=5
)
```

### Evaluation
```
from src.evaluation import RecommenderEvaluator
```

Comprehensive evaluation:
```
evaluator = RecommenderEvaluator()
results = evaluator.comprehensive_evaluation(actual_ratings, predicted_scores)
```

Generate evaluation plots:
```
evaluator.plot_precision_recall_at_k(results, model_name="Hybrid System")
```

## Dependencies

- **PySpark 3.5.1**: For ALS collaborative filtering (Note: PySpark requires Java 8/11 and proper Spark configuration)
- **TensorFlow 2.8.0**: For Two-Tower neural network
- **scikit-learn 1.0.2**: For evaluation metrics and preprocessing
- **pandas 1.4.2**: For data manipulation
- **numpy 1.21.5**: For numerical computations
- **matplotlib 3.5.1**: For visualization

## Documentation

- **[Methodology](docs/methodology.md)**: Complete technical methodology
- **[Data Documentation](data/README.md)**: Dataset information and access instructions
- **[Source Code](src/)**: Well-documented Python modules with examples

## Reproducibility

This implementation provides:
- **Complete source code** with comprehensive documentation
- **Modular design** allowing individual component testing
- **Reproducible preprocessing pipeline** with probability-based imputation
- **Consistent evaluation framework** with multiple metrics
- **Clear methodology documentation** for academic replication

## Data and Code Availability

**Source Code**: Openly available in this GitHub repository under MIT License

**Dataset**: Amazon e-commerce data available from [https://github.com/aksharpandia/miniamazon_data](https://github.com/aksharpandia/miniamazon_data). Due to licensing restrictions, the dataset cannot be redistributed. Complete access instructions are provided in `data/README.md`.

## Citation
```
@article{hazra2025hybrid,
title={A Hybrid Recommender System for Amazon E-commerce Combining ALS Collaborative Filtering and Two-Tower Content-Based Filtering},
author={Hazra, Soumi},
journal={Submitted to PeerJ Computer Science},
year={2025}
}
```

## Contributing

This repository is part of academic research. For questions or collaboration inquiries, please open an issue or contact the author.

## License

MIT License - see [LICENSE](LICENSE) file for details.

---

**Note**: This implementation is designed for academic research and educational purposes. The hybrid approach demonstrates novel techniques for combining collaborative and content-based filtering in e-commerce recommendation systems.



