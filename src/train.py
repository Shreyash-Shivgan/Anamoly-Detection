import os
import pickle
import numpy as np
import pandas as pd

from src.preprocessing import select_features
from src.data_loader import load_csv_data, save_csv_data, save_numpy_array
from src.preprocessing import (
    handle_missing_values,
    normalize_data,
    create_sliding_windows,
    split_dataset
)
from src.visualization import plot_training_loss

from models.dense_autoencoder import build_dense_autoencoder
from models.sparse_autoencoder import build_sparse_autoencoder
from models.variational_autoencoder import build_vae
from models.denoising_autoencoder import build_denoising_autoencoder


def generate_synthetic_data(filepath, samples=5000):

    print("Generating synthetic data as none was found...")

    time = pd.date_range(start="2023-01-01", periods=samples, freq="h")

    hours = time.hour.to_numpy()

    base = 5.0
    daily_cycle = 3.0 * np.sin(2 * np.pi * hours / 24)
    noise = np.random.normal(0, 0.5, samples)

    power = base + daily_cycle + noise
    power = np.array(power)

    anomaly_idx = np.random.choice(samples, int(samples * 0.02))
    power[anomaly_idx] += 15

    df = pd.DataFrame({
        "timestamp": time,
        "power_consumption": power
    })

    save_csv_data(df, filepath)

    return df


def main():

    RAW_DATA_PATH = "data/raw/data.csv"
    MODELS_DIR = "outputs/models"
    PLOTS_DIR = "outputs/plots"
    PROCESSED_DATA_DIR = "data/processed"

    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

    if not os.path.exists(RAW_DATA_PATH):
        df = generate_synthetic_data(RAW_DATA_PATH)
    else:
        df = load_csv_data(RAW_DATA_PATH)

    print("Dataset shape:", df.shape)

    df = handle_missing_values(df)

    if "power_consumption" in df.columns:
        target_col = "power_consumption"
    else:
        target_col = df.select_dtypes(include=[np.number]).columns[0]


    df_features = select_features(df)

    power_data = df_features.values

    normalized_data, scaler = normalize_data(power_data)

    with open(os.path.join(MODELS_DIR, "scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)

    WINDOW_SIZE = 24

    X = create_sliding_windows(normalized_data, WINDOW_SIZE)

    print("Windowed data shape:", X.shape)

    X_train, X_test = split_dataset(X, test_size=0.2)

    save_numpy_array(X_train, os.path.join(PROCESSED_DATA_DIR, "X_train.npy"))
    save_numpy_array(X_test, os.path.join(PROCESSED_DATA_DIR, "X_test.npy"))

    print("Train size:", X_train.shape[0], "Test size:", X_test.shape[0])

    num_features = X_train.shape[2]
    input_shape = (WINDOW_SIZE, num_features)

    models = {
        "dense": build_dense_autoencoder(input_shape),
        "sparse": build_sparse_autoencoder(input_shape),
        "vae": build_vae(input_shape),
        "denoising": build_denoising_autoencoder(input_shape)
    }

    EPOCHS = 20
    BATCH_SIZE = 64

    for name, model in models.items():

        print("\nTraining", name.upper(), "Autoencoder")

        history = model.fit(
            X_train,
            X_train,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            validation_split=0.1,
            verbose=1
        )

        plot_training_loss(
            history,
            title=f"{name} training loss",
            save_name=f"{name}_loss.png"
        )

        # DO NOT SAVE VAE (causes serialization error)
        if name == "vae":
            print("VAE trained successfully (not saved due to subclass model).")
            continue

        model.save_weights(
            os.path.join(MODELS_DIR, f"{name}_autoencoder.weights.h5")
        )

        print(name, "model weights saved")

    print("\nAll models trained successfully!")


if __name__ == "__main__":
    main()