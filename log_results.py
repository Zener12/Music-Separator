import sys
import torch
import mlflow
from train import load_checkpoint
from evaluate import evaluate_model

def main():
    # MLFlow не смог распознать кириллицу в пути (проблема в имени пользователя)
    mlflow.set_tracking_uri("./mlruns")

    checkpoint_path = './checkpoints/checkpoint_epoch_48.pt'
    musdb_root      = './musdb18'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model, _ = load_checkpoint(checkpoint_path, device)

    metrics = evaluate_model(model, musdb_root, device)

    mlflow.set_experiment("music-separation")
    with mlflow.start_run():
        mlflow.log_param("lr", 0.001)
        mlflow.log_params({"batch_size": 8, "epochs": 50})

        for k, v in metrics.items():
            mlflow.log_metric(f"sdr_{k}", v)

        mlflow.log_artifact("./checkpoints/model.onnx")

if __name__ == '__main__':
    main()