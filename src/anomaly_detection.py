import os
import pickle
import numpy as np
from src.preprocessing import create_sliding_windows
from models.dense_autoencoder import build_dense_autoencoder
from models.sparse_autoencoder import build_sparse_autoencoder
from models.variational_autoencoder import build_vae
from models.denoising_autoencoder import build_denoising_autoencoder

class AnomalyDetector:
    def __init__(self, model_type='dense', window_size=24, num_features=1):
        self.model_type = model_type
        self.window_size = window_size
        self.num_features = num_features
        self.model = None
        self.scaler = None
        self.threshold = None
        self._load_resources()
        
    def _load_resources(self):
        MODELS_DIR = "outputs/models"
        scaler_path = os.path.join(MODELS_DIR, "scaler.pkl")
        if os.path.exists(scaler_path):
            with open(scaler_path, "rb") as f:
                self.scaler = pickle.load(f)
                # automatically infer num_features from scaler if available
                self.num_features = self.scaler.n_features_in_
        else:
            print("Warning: Scaler not found, using raw values. Results may be inaccurate.")
            
        input_shape = (self.window_size, self.num_features)
        
        if self.model_type == 'dense':
            self.model = build_dense_autoencoder(input_shape)
        elif self.model_type == 'sparse':
            self.model = build_sparse_autoencoder(input_shape)
        elif self.model_type == 'vae':
            self.model = build_vae(input_shape)
        elif self.model_type == 'denoising':
            self.model = build_denoising_autoencoder(input_shape)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
            
        weights_path = os.path.join(MODELS_DIR, f"{self.model_type}_autoencoder.weights.h5")
        if os.path.exists(weights_path):
            self.model.load_weights(weights_path)
            # Dummy predict to initialize (especially for VAE)
            dummy = np.zeros((1, self.window_size, self.num_features))
            self.model.predict(dummy, verbose=0)
        else:
            print(f"Warning: Model weights not found at {weights_path}.")

    def detect(self, series_data, threshold=None):
        """
        Detect anomalies in a continuous multivariate time series.
        series_data shape: (n_samples, num_features)
        """
        if self.scaler:
            series_data = self.scaler.transform(series_data)
        
        # Create sliding windows
        windows = create_sliding_windows(series_data, self.window_size)
        
        if len(windows) == 0:
            return np.array([]), np.array([]), np.array([])
            
        reconstructed = self.model.predict(windows, verbose=0)
        
        # Compute MSE mapped back to each original point
        # A simple approach: use the error of the last point of the window for detection
        errors = np.mean(np.square(windows - reconstructed), axis=2)
        window_errors = np.mean(errors, axis=1) # Mean error per window
        
        if threshold is None:
            threshold = np.mean(window_errors) + 3*np.std(window_errors)
            
        is_anomaly = (window_errors > threshold).astype(int)
        
        # Padding the first window_size-1 points with the first window's error
        padded_errors = np.concatenate([np.full(self.window_size-1, window_errors[0]), window_errors])
        padded_anomalies = np.concatenate([np.zeros(self.window_size-1), is_anomaly])
        
        return padded_errors, padded_anomalies, threshold
