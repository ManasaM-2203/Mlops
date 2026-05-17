import json

from pipeline.config import ROOT


def run():
    models_dir = ROOT / "saved_models"
    models = ["crf", "bilstm_crf", "bert_ner", "distilbert_ner"]

    results = {}
    best_model = None
    best_f1 = 0

    for name in models:
        metrics_path = models_dir / name / "metrics.json"
        if not metrics_path.exists():
            print(f"[evaluate] {name}: no metrics found, skipping")
            continue

        with open(metrics_path) as f:
            metrics = json.load(f)

        results[name] = metrics
        print(f"[evaluate] {name}: F1={metrics['f1']:.4f} P={metrics['precision']:.4f} R={metrics['recall']:.4f}")

        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            best_model = name

    results["best_model"] = best_model
    results["best_f1"] = best_f1
    print(f"[evaluate] best model: {best_model} (F1={best_f1:.4f})")

    with open(models_dir / "comparison.json", "w") as f:
        json.dump(results, f, indent=2)

    with open(ROOT / "best_model_info.json", "w") as f:
        json.dump({"model": best_model, "f1": best_f1}, f, indent=2)


if __name__ == "__main__":
    run()
