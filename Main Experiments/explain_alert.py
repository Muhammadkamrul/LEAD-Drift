# FILE: explain_alert.py
import argparse
import os
import numpy as np
import pandas as pd
import tensorflow as tf
import shap

# Import from your existing project files
from risk_model import create_risk_model
from compare_different_dataset_size import make_forward_looking_dataset

custom_names = [
        "CPU (%)",
        "RAM (%)",
        "Storage (%)",
        "Network (NET_CONN)",
        "Service (SERV_RESP)",
        "CPU Δ",
        "Service Δ"
    ]


def _to_scalar(x):
    """
    Best-effort conversion of SHAP/TF expected_value into a Python float.
    Handles cases where expected_value is a list/array/tensor.
    """
    try:
        # TensorFlow tensor -> numpy -> scalar
        if hasattr(x, "numpy"):
            x = x.numpy()
        x = np.asarray(x).reshape(-1)[0]
        return float(x)
    except Exception:
        return float(x)


def explain_prediction(model, background_data, instance_to_explain, feature_names):
    """
    Generates a SHAP explanation for a single prediction.

    Args:
        model (tf.keras.Model): The trained risk model.
        background_data (np.ndarray): A sample of training data for the explainer. Shape (M, F).
        instance_to_explain (np.ndarray): The single data point to explain. Shape (F,).
        feature_names (list[str]): Feature names of length F.

    Returns:
        shap.Explanation: The SHAP explanation object (values 1D, data 1D).
    """
    # SHAP's DeepExplainer for TensorFlow models
    explainer = shap.DeepExplainer(model, background_data)

    # Ensure correct shapes/dtypes for TF/SHAP
    instance_2d = instance_to_explain.reshape(1, -1).astype(np.float32)

    # shap_values is usually a list (one element for a single-output model)
    shap_values_list = explainer.shap_values(instance_2d)
    # Take the first output (shape: (1, n_features)) and squeeze to 1D (n_features,)
    values_1d = np.asarray(shap_values_list[0]).reshape(1, -1)[0]

    # Base value can be scalar, list, array, or tensor; normalize to a plain float
    base_value = _to_scalar(explainer.expected_value)

    # Construct a clean single-sample Explanation (all 1D lengths == n_features)
    explanation = shap.Explanation(
        values=values_1d,
        base_values=base_value,
        data=instance_to_explain.astype(np.float32),
        feature_names=custom_names,
    )
    return explanation


def main():
    parser = argparse.ArgumentParser(description="Generate a SHAP explanation for a specific alert.")
    parser.add_argument('--data', required=True, help="Path to the dataset CSV.")
    parser.add_argument('--model_path', required=True, help="Path to the saved .h5 model file.")
    parser.add_argument('--alert_index', type=int, required=True, help="Row index from the CSV to analyze as an alert.")
    parser.add_argument('--outdir', required=True, help="Directory to save the explanation plot.")
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    tf.random.set_seed(args.seed)
    np.random.seed(args.seed)

    # --- 1) Load Data and Model ---
    print(f"Loading data from {args.data} and model from {args.model_path}")
    df = pd.read_csv(args.data)
    model = tf.keras.models.load_model(args.model_path)

    features = ['cpu_pct', 'ram_pct', 'storage_pct', 'snet', 'sri', 'cpu_delta', 'sri_delta']

    # X: numpy array of shape (N, F)
    X, _, _ = make_forward_looking_dataset(df, [], features)
    if args.alert_index >= len(X):
        raise ValueError(
            f"Alert index {args.alert_index} is out of bounds for the processed data (length {len(X)})."
        )

    # --- 2) Select Data for Explanation ---
    instance_to_explain = X[args.alert_index].astype(np.float32)
    bg_size = min(100, X.shape[0])
    # If dataset is exactly size N, choose bg_size distinct indices
    background_sample_indices = np.random.choice(X.shape[0], bg_size, replace=False)
    background_data = X[background_sample_indices].astype(np.float32)

    # --- 3) Generate Explanation ---
    print(f"\n--- Generating SHAP explanation for data at index {args.alert_index} ---")
    explanation = explain_prediction(model, background_data, instance_to_explain, features)

    # --- 4) Numerical Results ---
    print("\n--- Numerical SHAP Values (Feature Contributions) ---")
    # explanation.values is 1D length = n_features
    for feature, value in zip(explanation.feature_names, explanation.values):
        print(f"{feature:<15}: {value:+.4f}")

    base_value = float(explanation.base_values)
    print(f"\nBase model output value: {base_value:.4f}")

    # Model prediction for the instance
    predicted_score = float(model.predict(instance_to_explain.reshape(1, -1), verbose=0)[0][0])
    print(f"Final predicted score: {predicted_score:.4f}")

    # Sanity check: base + sum(shap) ≈ prediction (for additive explanations)
    approx_pred = base_value + float(np.sum(explanation.values))
    print(f"Base + sum(SHAP):      {approx_pred:.4f}  (difference: {predicted_score - approx_pred:+.4e})")

    # --- 5) Visual: SHAP force plot ---
    plot_path = os.path.join(args.outdir, f'shap_force_plot_index_{args.alert_index}.html')

    # Build the force plot explicitly with per-argument inputs (avoids shape ambiguity)
    shap_plot = shap.force_plot(
        base_value=base_value,
        shap_values=explanation.values,      # 1D (n_features,)
        features=explanation.data,           # 1D (n_features,)
        feature_names=custom_names,
    )
    shap.save_html(plot_path, shap_plot)

    import matplotlib.pyplot as plt
    
    shap.plots.bar(explanation, show=False)
    ax = plt.gca()
    ax.set_xlabel(ax.get_xlabel(), fontsize=15)
    ax.set_ylabel(ax.get_ylabel(), fontsize=15)
    ax.tick_params(axis='both', labelsize=15)
    legend = ax.get_legend()
    if legend is not None:
        legend.set_fontsize(12)
    plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, f'shap_bar_plot_index_{args.alert_index}.png'))
    plt.close()

    print(f"\nSUCCESS: Saved interactive SHAP force plot to:\n{plot_path}")


if __name__ == '__main__':
    main()
