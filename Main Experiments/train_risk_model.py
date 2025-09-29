import os, argparse, json, random
import numpy as np, pandas as pd
import tensorflow as tf
from risk_model import create_risk_model
# MODIFIED: Import new plotting functions
from plotting import bar_attribution, plot_confusion_matrix, plot_risk_scores
# MODIFIED: Import sklearn for performance metrics
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

# Set a fixed seed for all random sources for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
os.environ['TF_DETERMINISTIC_OPS'] = '1'
os.environ['PYTHONHASHSEED'] = str(SEED)
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

def write_json(path, obj):
    with open(path, 'w') as f: json.dump(obj, f, indent=2)

def read_json(path):
    with open(path, 'r') as f: return json.load(f)

def make_dataset(df):
    """
    MODIFIED: This function now processes a dataframe (train or val)
    to prevent data leakage from the .diff() operation.
    """
    df_processed = df.copy()
    df_processed['cpu_delta'] = df_processed['cpu_pct'].diff().fillna(0)
    df_processed['sri_delta'] = df_processed['sri'].diff().fillna(0)
    
    features = ['cpu_pct', 'ram_pct', 'storage_pct', 'snet', 'sri', 'cpu_delta', 'sri_delta']
    
    # We lose the first row due to the lookahead for Y
    X = df_processed[features].values[:-1].astype('float32')
    
    # Target (Y) is based on the NEXT timestep's health
    snet_next = df_processed['snet'].values[1:]
    sri_next = df_processed['sri'].values[1:]
    
    # Risk score: 0 = healthy, 1 = total failure
    Y = 1.0 - np.minimum(snet_next, sri_next) / 100.0
    Y = Y.reshape(-1, 1).astype('float32')
    
    # Return time column for plotting
    t = df_processed['t'].values[:-1]
    
    return X, Y, t

def gradient_sensitivity(model, X_sample):
    idx = np.where((X_sample[:, 3] > 98.5) & (X_sample[:, 4] > 98.5))[0]
    if idx.size < 50: idx = np.arange(len(X_sample))
    X_tensor = tf.convert_to_tensor(X_sample[idx], dtype=tf.float32)
    with tf.GradientTape() as tape:
        tape.watch(X_tensor)
        y_pred = model(X_tensor, training=False)
    grads = tape.gradient(y_pred, X_tensor)
    abs_grads = tf.math.abs(grads)
    mean_grads = tf.reduce_mean(abs_grads, axis=0).numpy()
    names = ['cpu', 'ram', 'storage', 'snet', 'sri', 'cpu_delta', 'sri_delta']
    raw = {n: float(mean_grads[i]) for i, n in enumerate(names)}
    s = sum(raw.values()) + 1e-9
    return {k: v / s for k, v in raw.items()}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--outdir', required=True)
    ap.add_argument('--epochs', type=int, default=25)
    ap.add_argument('--batch', type=int, default=64)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    df = pd.read_csv(args.data)
    
    # --- MODIFIED: Prevent data leakage by splitting the DataFrame first ---
    split_idx = int(0.8 * len(df))
    df_train = df.iloc[:split_idx]
    df_val = df.iloc[split_idx:]

    # Create datasets from the split dataframes
    X_train, Y_train, _ = make_dataset(df_train)
    X_val, Y_val, t_val = make_dataset(df_val)
    
    # --- Model Training (Unchanged) ---
    model = create_risk_model(input_dim=X_train.shape[1])
    model.summary()

    model.fit(X_train, Y_train,
              validation_data=(X_val, Y_val),
              epochs=args.epochs,
              batch_size=args.batch,
              verbose=2)

    model.save(os.path.join(args.outdir, 'risk_model.h5'))

    # --- Feature Importance (Unchanged) ---
    print("\n--- Feature Importance Analysis ---")
    w = gradient_sensitivity(model, X_train)
    write_json(os.path.join(args.outdir, 'learned_weights.json'), w)
    bar_attribution(
        vals=[w[k] for k in ['cpu','ram','storage','snet','sri','cpu_delta', 'sri_delta']],
        labels=['cpu','ram','storage','snet','sri','cpu_delta', 'sri_delta'],
        path=os.path.join(args.outdir, 'lead_attribution.png'),
        title='LEAD-Drift Learned Feature Importance',
        ylabel='Normalized Gradient'
    )
    print(f"Training complete. Model and artifacts saved to {args.outdir}")

if __name__ == '__main__':
    main()