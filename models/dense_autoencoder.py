from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Flatten, Reshape

def build_dense_autoencoder(input_shape):
    """
    Builds a fully connected Dense Autoencoder.
    """
    inputs = Input(shape=input_shape)
    
    # Flatten timeseries window
    x = Flatten()(inputs)
    
    # Encoder
    x = Dense(32, activation='relu')(x)
    x = Dense(16, activation='relu')(x)
    encoded = Dense(8, activation='relu')(x)
    
    # Decoder
    x = Dense(16, activation='relu')(encoded)
    x = Dense(32, activation='relu')(x)
    
    flat_dim = input_shape[0] * input_shape[1]

    # Output layer
    x = Dense(flat_dim, activation='sigmoid')(x)
    outputs = Reshape(input_shape)(x)
    
    model = Model(inputs, outputs, name="Dense_Autoencoder")
    model.compile(optimizer='adam', loss='mse')
    
    return model
