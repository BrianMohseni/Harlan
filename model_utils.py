import tensorflow as tf
from tensorflow import keras
from keras_nlp.layers import TransformerDecoder

def sinusoidal_embedding(maxlen, d_model):
    position = tf.range(maxlen, dtype=tf.float32)[:, tf.newaxis]
    div_term = tf.exp(tf.range(0, d_model, 2, dtype=tf.float32) * (-tf.math.log(10000.0) / d_model))
    angle_rads = position * div_term

    pos_encoding = tf.concat([tf.sin(angle_rads), tf.cos(angle_rads)], axis=-1)

    return pos_encoding


def build_transformer_decoder_model(vocab_size, maxlen, d_model, num_heads, num_layers, ff_dim):

    inputs = keras.Input(shape=(maxlen-1,), dtype=tf.int32, name="input_tokens")

    embedding_layer = keras.layers.Embedding(input_dim=vocab_size, output_dim=d_model, mask_zero=True)
    x = embedding_layer(inputs)

    positional_encoding = sinusoidal_embedding(maxlen-1, d_model)
    x += positional_encoding

    for _ in range(num_layers):
        decoder_layer = TransformerDecoder(intermediate_dim=ff_dim, num_heads=num_heads, activation=tf.keras.activations.swish)
        x = decoder_layer(x)

    x = keras.layers.Flatten()(x)
    outputs = keras.layers.Dense(vocab_size, activation='softmax')(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="transformer_decoder_model")

    return model
