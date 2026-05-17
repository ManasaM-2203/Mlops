import json

import mlflow
from mlflow.tracking import MlflowClient

from pipeline.config import ROOT, load_params


def run():
    params = load_params()
    mlflow.set_tracking_uri(params["mlflow"]["tracking_uri"])

    with open(ROOT / "best_model_info.json") as f:
        info = json.load(f)

    model_name = params["mlflow"]["registered_model_name"]
    client = MlflowClient()

    experiment = client.get_experiment_by_name(params["mlflow"]["experiment_name"])
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string=f"tags.mlflow.runName = '{info['model']}'",
        order_by=["metrics.f1 DESC"],
        max_results=1,
    )

    if not runs:
        print(f"[promote] no run found for {info['model']}")
        return

    run = runs[0]
    model_uri = f"runs:/{run.info.run_id}/artifacts"

    try:
        client.create_registered_model(model_name)
    except mlflow.exceptions.MlflowException:
        pass

    version = client.create_model_version(
        name=model_name,
        source=model_uri,
        run_id=run.info.run_id,
    )

    client.transition_model_version_stage(
        name=model_name,
        version=version.version,
        stage="Staging",
    )

    print(f"[promote] {info['model']} v{version.version} -> Staging")


if __name__ == "__main__":
    run()
