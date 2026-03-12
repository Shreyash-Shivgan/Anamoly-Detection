from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Flatten, Reshape, GaussianNoise

def build_denoising_autoencoder(input_shape, noise_factor=0.1):
    """
    Builds a Denoising Autoencoder that adds noise to input to learn robust features.
    """
    inputs = Input(shape=input_shape)
    
    # Add Gaussian noise
    noisy_inputs = GaussianNoise(noise_factor)(inputs)
    
    x = Flatten()(noisy_inputs)
    
    # Encoder
    x = Dense(32, activation='relu')(x)
    x = Dense(16, activation='relu')(x)
    encoded = Dense(8, activation='relu')(x)
    
    # Decoder
    x = Dense(16, activation='relu')(encoded)
    x = Dense(32, activation='relu')(x)
    
    flat_dim = input_shape[0] * input_shape[1]
    x = Dense(flat_dim, activation='sigmoid')(x)
    outputs = Reshape(input_shape)(x)
    
    # The Model still maps original clean inputs for predictions, 
    # but noise is applied internally during training via GaussianNoise layer.
    model = Model(inputs, outputs, name="Denoising_Autoencoder")
    model.compile(optimizer='adam', loss='mse')
    
    return model
