import numpy as np
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
from tensorflow.data import Dataset
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
import pickle
import os
from model import DecoderModel
import numpy as np
from generate_data import GenerateData
from tokenizers import Tokenizer
strategy = tf.distribute.MirroredStrategy()

checkpoint_filepath = 'vivaldi-v2-large'
strategy = tf.distribute.MirroredStrategy()
print('Number of devices: {}'.format(strategy.num_replicas_in_sync))

BATCH_SIZE = 32
TRAIN_RATIO = 0.8
VOCAB_SIZE = 2048
SEQ_LENGTH = 512
GD = GenerateData(VOCAB_SIZE, BATCH_SIZE, TRAIN_RATIO, True)
train_ds, val_ds, tokenizer = GD.get_tokenizers_and_batched_data(SEQ_LENGTH)
intermediate_dim = 1024
num_heads = 16
num_layers = 10
dropout = 0.0
normalize_first = True
embedding_dim = 256

with strategy.scope():
    model = DecoderModel(
        intermediate_dim = intermediate_dim, 
        num_heads = num_heads, 
        num_layers = num_layers,
        dropout = dropout, 
        normalize_first = normalize_first,
        embedding_dim = embedding_dim,
        sequence_length=SEQ_LENGTH-1,
        vocab_size=VOCAB_SIZE
    )
    optimizer = tf.keras.optimizers.AdamW(learning_rate=5e-4)
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()
    metrics = tf.keras.metrics.SparseCategoricalAccuracy()
    model.compile(optimizer=optimizer, metrics=[metrics], loss=loss_fn, run_eagerly=False)
model.fit(
    train_ds,
    validation_data=val_ds,
    batch_size = BATCH_SIZE,
    epochs=200,
    callbacks=[
    ModelCheckpoint(checkpoint_filepath, save_best_only = True),
    TensorBoard()
    ]
)