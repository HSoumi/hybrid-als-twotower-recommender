#!/usr/bin/env bash
set -e                                  # stop on first error

# ---- Step 1: Data preprocessing -------------------------------------------
echo "=== Step 1: Data preprocessing ==="
python - <<'PY'
from src.data_preprocessing import main
import pandas as pd, os

processed = main()                                      # runs full pipeline
os.makedirs("processed", exist_ok=True)
processed['user_item_interactions'].to_csv(
    "processed/user_item.csv", index=False)
print("Pre-processed data saved to processed/user_item.csv")
PY

# ---- Step 2: Train ALS + Two-Tower + Hybrid -------------------------------
echo "=== Step 2: Model training ==="
python - <<'PY'
import pandas as pd, pickle, os
from src.hybrid_system import HybridRecommendationSystem

df = pd.read_csv("processed/user_item.csv")
hrs = HybridRecommendationSystem()
hrs.train_models(df)                                    # trains both models

# (optional) persist the trained Keras model and Spark factors
os.makedirs("models", exist_ok=True)
hrs.twotower_model.save_model("models/two_tower.keras")
pickle.dump(hrs.als_model.model, open("models/als.pkl", "wb"))
print("Models saved in models/ directory")
PY

# ---- Step 3: Generate tables & figures ------------------------------------
echo "=== Step 3: Generating results ==="
python - <<'PY'
import pandas as pd, json, os
from src.hybrid_system import HybridRecommendationSystem
from src.evaluation import RecommenderEvaluator
from src.utils import get_user_item_interactions, print_evaluation_results

# reload minimal artefacts
df   = pd.read_csv("processed/user_item.csv")
hrs  = HybridRecommendationSystem()
hrs.train_models(df)                         # quick retrain to have the objects
users = [462, 9435]
all_items = df['itemId'].unique().tolist()

evaluator = RecommenderEvaluator()
os.makedirs("results", exist_ok=True)

for uid in users:
    actual = get_user_item_interactions(df, uid)
    recs   = hrs.get_hybrid_recommendations(uid, all_items, actual, top_k=20)

    # convert list of tuples → dict for evaluation
    pred_dict = {item: score for item, score in recs}

    results = evaluator.comprehensive_evaluation(actual, pred_dict)
    json.dump(results,
              open(f"results/metrics_user_{uid}.json", "w"),
              indent=2)
    evaluator.plot_precision_recall_at_k(
        results, model_name=f"Hybrid-U{uid}",
        save_path=f"results/precision_recall_u{uid}.png"
    )
    print_evaluation_results(results, user_id=uid)
PY

echo "Pipeline finished. Outputs are in the results/ folder."
