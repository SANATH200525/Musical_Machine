"""
train_and_save.py
-----------------
Run once to train all models from dataset.csv and save to artifacts.pkl.
After this, the FastAPI server loads in ~3 seconds instead of training from scratch.

Training time estimate: ~20-40 minutes (4-model ensemble × 2 classifiers on ~105k rows after filtering).

Usage:
    python train_and_save.py [--dataset PATH] [--output PATH]
"""
import os
import sys
import argparse
from ml_pipeline import fit_all, save_artifacts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="dataset.csv")
    parser.add_argument("--output", default="artifacts.pkl")
    args = parser.parse_args()

    if not os.path.exists(args.dataset):
        print(f"ERROR: '{args.dataset}' not found. Place dataset.csv in project root.")
        sys.exit(1)

    if os.path.exists(args.output):
        print(f"NOTE: '{args.output}' already exists - will be overwritten.")

    print("Training all models (XGBoost+LightGBM+CatBoost+MLP ensemble, 40 fine + 10 coarse genres)...")
    print("Expected time: 5-10 minutes on a 4-core machine (4-model ensemble x 2 classifiers).\n")
    try:
        arts = fit_all(args.dataset)
        save_artifacts(arts, args.output)
        print(f"\nDone. '{args.output}' saved.")

        m = arts.eval_metrics

        # Exporting Academic Performance Report
        report = (
            "MoodTune Model Performance Report\n"
            "==================================\n\n"
            "Taxonomy Architecture:\n"
            "  Fine Genres: 37 (Collapsed from 114 to remove unlearnable/linguistic tags)\n"
            "  Super-Genres: 10 (Acoustically cohesive clusters)\n\n"
            "Fine Genre Prediction (37 classes):\n"
            f"  Top-1 Accuracy:  {m.get('fine_top1', 0):.4f}\n"
            f"  Top-3 Accuracy:  {m.get('fine_top3', 0):.4f}\n"
            f"  Top-5 Accuracy:  {m.get('fine_top5', 0):.4f}  <-- Primary Publishable Metric\n"
            f"  Top-10 Accuracy: {m.get('fine_top10', 0):.4f}\n"
            f"  Macro-F1:        {m.get('fine_macro_f1', 0):.4f}\n\n"
            "Coarse Super-Genre Prediction (10 classes):\n"
            f"  Top-1 Accuracy:  {m.get('coarse_top1', 0):.4f}\n"
            f"  Top-3 Accuracy:  {m.get('coarse_top3', 0):.4f}  <-- Primary Publishable Metric\n"
            f"  Macro-F1:        {m.get('coarse_macro_f1', 0):.4f}\n"
        )
        with open("model_performance.txt", "w", encoding="utf-8") as f:
            f.write(report)

        print("Saved performance metrics to 'model_performance.txt'.")
        print("  Server will now start in ~3 seconds (loading from pickle).")

        # Terminal Output summary
        print("\n-- Evaluation summary --------------------------------------")
        print("  Fine genre  (37 classes)")
        print(f"    top-1:  {m.get('fine_top1', 'n/a'):.4f}  (was ~0.33 with 114 classes)")
        print(f"    top-3:  {m.get('fine_top3', 'n/a'):.4f}")
        print(f"    top-5:  {m.get('fine_top5', 'n/a'):.4f}  <- primary metric")
        print(f"    top-10: {m.get('fine_top10', 'n/a'):.4f}")
        print(f"    macro-F1: {m.get('fine_macro_f1', 'n/a'):.4f}")
        print("  Super-genre (10 classes)")
        print(f"    top-1:  {m.get('coarse_top1', 'n/a'):.4f}  (was ~0.55 with 13 classes)")
        print(f"    top-3:  {m.get('coarse_top3', 'n/a'):.4f}  <- primary metric")
        print(f"    macro-F1: {m.get('coarse_macro_f1', 'n/a'):.4f}")
        print("------------------------------------------------------------")
    except Exception as exc:
        print(f"Training failed: {exc}")
        raise


if __name__ == "__main__":
    main()
