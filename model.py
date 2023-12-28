from keras_nlp.layers import TransformerDecoder, TokenAndPositionEmbedding
from keras import Sequential, Model
from keras.layers import Embedding, Dense
import keras
import numpy as np
import tensorflow as tf

@keras.saving.register_keras_serializable()
class DecoderModel(Model):
    def __init__(self, intermediate_dim, num_heads, num_layers, dropout, normalize_first, embedding_dim, sequence_length, vocab_size):
            super().__init__()
            self.sequence_length = sequence_length
            self.vocab_size = vocab_size
            self.intermediate_dim = intermediate_dim
            self.num_heads = num_heads
            self.num_layers = num_layers
            self.embedding_dim = embedding_dim
            self.dropout = dropout
            self.activation="relu"
            self.layer_norm_epsilon=1e-05
            self.kernel_initializer="glorot_uniform"
            self.bias_initializer="zeros"
            self.normalize_first=normalize_first
            self.model = [TransformerDecoder(
                self.intermediate_dim,
                self.num_heads,
                self.dropout,
                self.activation,
                self.layer_norm_epsilon,
                self.kernel_initializer,
                self.bias_initializer,
                self.normalize_first
                ) for _ in range(self.num_layers)]
            self.embedding_layer = TokenAndPositionEmbedding(
                vocabulary_size = self.vocab_size,
                sequence_length = self.sequence_length,
                embedding_dim = self.embedding_dim,
                mask_zero = True
            )
            self.output_layer = Dense(units=self.vocab_size, activation="softmax")
        
    def call(self, decoder_sequence):
        embedded_input = self.embedding_layer(decoder_sequence)
        for decoder in self.model:
            embedded_input = decoder(embedded_input)
        output = self.output_layer(embedded_input)
        return output