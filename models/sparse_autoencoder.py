from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Flatten, Reshape
from tensorflow.keras import regularizers

def build_sparse_autoencoder(input_shape, sparsity_weight=1e-4):
    """
    Builds a Sparse Autoencoder using L1 regularization.
    """
    inputs = Input(shape=input_shape)
    
    x = Flatten()(inputs)
    
    # Encoder with sparsity constraint on the latent representation
    x = Dense(32, activation='relu')(x)
    encoded = Dense(16, activation='relu', activity_regularizer=regularizers.l1(sparsity_weight))(x)
    
    # Decoder
    x = Dense(32, activation='relu')(encoded)
    
    flat_dim = input_shape[0] * input_shape[1]
    x = Dense(flat_dim, activation='sigmoid')(x)
    outputs = Reshape(input_shape)(x)
    
    model = Model(inputs, outputs, name="Sparse_Autoencoder")
    model.compile(optimizer='adam', loss='mse')
    
    return model
