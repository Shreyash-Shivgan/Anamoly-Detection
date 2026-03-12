import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

sns.set_theme(style="whitegrid")

def _get_save_path(filename, save_dir="outputs/plots"):
    os.makedirs(save_dir, exist_ok=True)
    return os.path.join(save_dir, filename)

def plot_time_series(df, column, title="Power Consumption Over Time", save_name=None):
    plt.figure(figsize=(15, 6))
    plt.plot(df.index, df[column], label=column, color='blue', alpha=0.7)
    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel("Power Consumption")
    plt.legend()
    plt.tight_layout()
    if save_name:
        plt.savefig(_get_save_path(save_name))
    plt.show()

def plot_training_loss(history, title="Model Training Loss", save_name=None):
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Train Loss')
    if 'val_loss' in history.history:
        plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title(title)
    plt.xlabel("Epochs")
    plt.ylabel("Loss (MSE)")
    plt.legend()
    plt.tight_layout()
    if save_name:
        plt.savefig(_get_save_path(save_name))
    plt.show()

def plot_reconstruction(original, reconstructed, sample_idx=0, title="Reconstruction vs Original", save_name=None):
    plt.figure(figsize=(12, 6))
    # Using the first feature of the specified sequence window
    plt.plot(original[sample_idx, :, 0], label='Original', color='blue')
    plt.plot(reconstructed[sample_idx, :, 0], label='Reconstructed', color='red', linestyle='dashed')
    plt.title(title)
    plt.xlabel("Time Step (in window)")
    plt.ylabel("Value")
    plt.legend()
    plt.tight_layout()
    if save_name:
        plt.savefig(_get_save_path(save_name))
    plt.show()

def plot_reconstruction_error(errors, threshold=None, title="Reconstruction Error Distribution", save_name=None):
    plt.figure(figsize=(10, 6))
    sns.histplot(errors, bins=50, kde=True, color='purple')
    if threshold:
        plt.axvline(threshold, color='red', linestyle='dashed', linewidth=2, label=f'Threshold: {threshold:.4f}')
    plt.title(title)
    plt.xlabel("Reconstruction Error (MSE)")
    plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()
    if save_name:
        plt.savefig(_get_save_path(save_name))
    plt.show()

def plot_anomalies(data, errors, threshold, anomalies_idx, title="Detected Anomalies", save_name=None):
    plt.figure(figsize=(15, 6))
    # Using the first element of each sequence for plotting the continuous line
    plt.plot(data[:, 0, 0], label='Power Consumption', color='blue', alpha=0.6)
    
    # Highlight anomalies
    anomaly_points = data[anomalies_idx, 0, 0]
    plt.scatter(anomalies_idx, anomaly_points, color='red', label='Anomaly', s=20)
    
    plt.title(title)
    plt.xlabel("Window Index")
    plt.ylabel("Value")
    plt.legend()
    plt.tight_layout()
    if save_name:
        plt.savefig(_get_save_path(save_name))
    plt.show()
