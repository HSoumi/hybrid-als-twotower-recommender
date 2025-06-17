#!/usr/bin/env bash
set -e

# ---- Step 1: Data preprocessing -------------------------------------------
echo "=== Step 1: Data preprocessing ==="
python - <<'PY'
from data_preprocessing import main
main()  # Saves processed data to processed/ directory
PY

# ---- Step 2: Train Models with Hyperparameter Tuning ----------------------
echo "=== Step 2: Model Training ==="

# Train ALS Model
echo "-- Training ALS Model --"
python - <<'PY'
import pandas as pd
from als_model import hyperparameter_tuning, ALSModel

train_data = pd.read_csv("processed/train_data.csv")
val_data = pd.read_csv("processed/test_data.csv")

param_grid = [
    {'rank': 10, 'max_iter': 10, 'reg_param': 0.1},
    {'rank': 20, 'max_iter': 20, 'reg_param': 0.05},
    {'rank': 15, 'max_iter': 15, 'reg_param': 0.5},
    {'rank': 20, 'max_iter': 5, 'reg_param': 0.1},
    {'rank': 15, 'max_iter': 12, 'reg_param': 0.2}
]

best_params = hyperparameter_tuning(train_data, val_data, param_grid)
model = ALSModel(**best_params)
model.train(train_data)
model.save_model("models/als")
PY

# Train Two-Tower Model
echo "-- Training Two-Tower Model --"
python - <<'PY'
import pandas as pd
from two_tower_model import hyperparameter_tuning, TwoTowerModel

train_data = pd.read_csv("processed/train_data.csv")

param_grid = [
    {'batch_size': 32, 'epochs': 50},
    {'batch_size': 64, 'epochs': 30},
    {'batch_size': 128, 'epochs': 20},
    {'batch_size': 256, 'epochs': 10},
    {'batch_size': 512, 'epochs': 5}
]

best_params = hyperparameter_tuning(train_data, param_grid)
model = TwoTowerModel(
    train_data['userId'].nunique(),
    train_data['itemId'].nunique(),
    train_data['manufacturer_id'].nunique(),
    train_data['category_id'].nunique()
)
model.train(train_data, **best_params)
model.save_model("models/twotower.keras")
PY

# ---- Step 3: Generate Recommendations & Evaluate --------------------------
echo "=== Step 3: Generating Results ==="
python - <<'PY'
import pandas as pd
import json
import os
from hybrid_system import HybridRecommendationSystem
from evaluation import RecommenderEvaluator

# Initialize components
hrs = HybridRecommendationSystem()
hrs.load_models("models/als", "models/twotower.keras")
evaluator = RecommenderEvaluator()

# Load test data
test_data = pd.read_csv("processed/test_data.csv")
users = test_data['userId'].unique()
all_items = test_data['itemId'].unique().tolist()

os.makedirs("results", exist_ok=True)

for user_id in users:
    try:
        # Generate and save predictions
        items = test_data[test_data['userId'] == user_id]
        recs = hrs.get_hybrid_recommendations(user_id, items, save_predictions=True)
        
        # Load saved predictions for evaluation
        preds = hrs.load_predictions(user_id)
        actual = dict(zip(items['itemId'], items['average_review_rating']))
        
        # Evaluate
        results = evaluator.comprehensive_evaluation(actual, dict(preds))
        
        # Save metrics
        with open(f"results/metrics_user_{user_id}.json", "w") as f:
            json.dump(results, f, indent=2)
            
        # Generate plots
        evaluator.plot_precision_recall_at_k(
            results, 
            k_values=[5, 10, 15, 20],
            model_name=f"Hybrid-User{user_id}",
            save_path=f"results/precision_recall_u{user_id}.pdf"
        )
        
    except Exception as e:
        print(f"Error processing user {user_id}: {str(e)}")
        continue

print("Evaluation completed for all test users")
PY

echo "=== Pipeline Complete ==="
echo "Outputs saved to:"
echo "- Models: models/"
echo "- Results: results/"
