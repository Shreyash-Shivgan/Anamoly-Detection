import numpy as np
from src.data_loader import load_numpy_array
from models.dense_autoencoder import build_dense_autoencoder
from models.sparse_autoencoder import build_sparse_autoencoder
from models.denoising_autoencoder import build_denoising_autoencoder

MODEL_PATHS = {
    "dense": "outputs/models/dense_autoencoder.weights.h5",
    "sparse": "outputs/models/sparse_autoencoder.weights.h5",
    "denoising": "outputs/models/denoising_autoencoder.weights.h5",
}


def compute_reconstruction_error(model, X):
    reconstructed = model.predict(X)
    # X shape is (samples, window_size, num_features)
    # The MSE over axis 1 and 2 captures the error across the entire window and all features
    mse = np.mean(np.square(X - reconstructed), axis=(1, 2))
    return mse


def evaluate_model(name, model_path, X_test):

    print("\n--- Evaluating", name.upper(), "Autoencoder ---")

    # Get both window size and number of features from test data shape
    input_shape = (X_test.shape[1], X_test.shape[2])
    
    try:
        if name == "dense":
            model = build_dense_autoencoder(input_shape)
        elif name == "sparse":
            model = build_sparse_autoencoder(input_shape)
        elif name == "denoising":
            model = build_denoising_autoencoder(input_shape)
        else:
            print("Skipping", name, ": Unknown model type")
            return
            
        model.load_weights(model_path)
    except Exception as e:
        print("Skipping", name, f": model not found ({e})")
        return

    errors = compute_reconstruction_error(model, X_test)

    threshold = np.percentile(errors, 95)

    anomalies = errors > threshold

    print("Threshold:", round(threshold, 5))
    print("Detected anomalies:", np.sum(anomalies))
    print("Total samples:", len(errors))


def main():

    X_test = load_numpy_array("data/processed/X_test.npy")

    for name, path in MODEL_PATHS.items():
        evaluate_model(name, path, X_test)


if __name__ == "__main__":
    main()