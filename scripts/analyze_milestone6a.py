import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "results" / "milestone6a"


def read_summary():
    return json.loads((OUTPUT_DIR / "summary.json").read_text(encoding="utf-8"))["models"]


def short_name(model: str) -> str:
    return model.split("/")[-1].replace("Qwen3-", "").replace("-Base", "")


def plot_speedup(models):
    figure, axis = plt.subplots(figsize=(6.4, 4.4))
    for model, values in models.items():
        points = [
            (int(batch), data["comparison"]["latency_speedup_generation_over_label"])
            for batch, data in values["batch_sizes"].items()
            if "comparison" in data
        ]
        axis.plot(*zip(*points), marker="o", label=short_name(model))
    axis.axhline(1.0, color="0.5", linestyle="--")
    axis.set(
        xlabel="Batch size",
        ylabel="Generation / label-logit total latency",
        title="Zero-decoding latency speedup",
        xticks=[1, 8, 32, 64],
    )
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / "figures" / "speedup_by_batch.png", dpi=150)
    plt.close(figure)


def plot_method_metric(models, metric: str, ylabel: str, filename: str):
    figure, axes = plt.subplots(1, 3, figsize=(14, 4.2), sharey=True)
    for axis, (model, values) in zip(axes, models.items()):
        for method in ["generation", "label_logit"]:
            points = []
            for batch, data in values["batch_sizes"].items():
                method_data = data[method]
                if method_data["status"] == "ok":
                    points.append((int(batch), method_data[metric]))
            axis.plot(*zip(*points), marker="o", label=method)
        axis.set(title=short_name(model), xlabel="Batch size", xticks=[1, 8, 32, 64])
        axis.grid(alpha=0.25)
    axes[0].set_ylabel(ylabel)
    axes[-1].legend()
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / "figures" / filename, dpi=150)
    plt.close(figure)


def main() -> None:
    models = read_summary()
    (OUTPUT_DIR / "figures").mkdir(parents=True, exist_ok=True)
    plot_speedup(models)
    plot_method_metric(
        models,
        "throughput_samples_per_second",
        "Throughput (samples/s)",
        "throughput_by_batch.png",
    )
    plot_method_metric(
        models,
        "peak_cuda_memory_mib",
        "Peak CUDA memory (MiB)",
        "peak_memory_by_batch.png",
    )


if __name__ == "__main__":
    main()
