import tensorflow as tf
from tensorflow.data import Dataset
import numpy as np
import pickle
from tokenizers import Tokenizer

class GenerateData:
    def __init__(self, vocab_size, batch_size, train_ratio, optimizing = False):
        self.vocab_size = vocab_size
        self.batch_size = batch_size
        self.train_ratio = train_ratio
        self.optimizing=optimizing   
        self.tokenizer = Tokenizer.from_file(f"tokenizer_{self.vocab_size}.json")
        self.list_of_tokenids_per_song = self.tokenize_corpus()
    def tokenize_corpus(self):
        with open(f"token_ids_{self.vocab_size}_small", "rb") as f:
            res = pickle.load(f)
        if self.optimizing:
            no_songs = len(res)
            choices = np.random.choice(no_songs, size = int(no_songs/10), replace = False)
            res = [res[i] for i in choices]
        return res
    def get_tokenizers_and_batched_data(self, seq_length):
        no_songs = len(self.list_of_tokenids_per_song)
        inputs = np.zeros((no_songs*2, seq_length - 1), dtype = np.uint16)
        targets = np.zeros((no_songs*2, seq_length - 1), dtype = np.uint16)
        start = time.time()
        cur_idx = 0
        for song_no, list_of_indices in enumerate(self.list_of_tokenids_per_song):
            for i in range(len(list_of_indices)//seq_length):
                start, end = i*seq_length, (i+1)*seq_length
                inputs[cur_idx] = list_of_indices[start:end][:-1]
                targets[cur_idx] = list_of_indices[start:end][1:]
                cur_idx += 1
            pad_length = seq_length-len(list_of_indices)%seq_length
            if pad_length < seq_length:
                inputs[cur_idx] = list_of_indices[-seq_length+pad_length:][:-1]+pad_length*[0]
                targets[cur_idx] = list_of_indices[-seq_length+pad_length:][1:]+pad_length*[0]
                cur_idx += 1
        inputs = inputs[:cur_idx]
        targets = targets[:cur_idx]
        no_sequence_pairs = inputs.shape[0]

        no_train = int(self.train_ratio*no_sequence_pairs)

        inputs_train, targets_train = inputs[:no_train], targets[:no_train]
        inputs_val, targets_val = inputs[no_train:], targets[no_train:]

        pad_length_train = self.batch_size - inputs_train.shape[0]%self.batch_size
        inputs_train, targets_train  = np.vstack((inputs_train,inputs_train[-pad_length_train:])), np.vstack((targets_train,targets_train[-pad_length_train:]))
        pad_length_val = self.batch_size - inputs_val.shape[0]%self.batch_size
        inputs_val, targets_val  = np.vstack((inputs_val,inputs_val[-pad_length_val:])), np.vstack((targets_val,targets_val[-pad_length_val:]))
        train_ds = Dataset.from_tensor_slices((inputs_train,targets_train))
        train_ds = train_ds.batch(self.batch_size)
        val_ds = Dataset.from_tensor_slices((inputs_val,targets_val))
        val_ds = val_ds.batch(self.batch_size)

        train_ds = train_ds.shuffle(buffer_size = 10000, reshuffle_each_iteration = True)
        val_ds = val_ds.shuffle(buffer_size = 10000, reshuffle_each_iteration = True)
        return train_ds, val_ds, self.tokenizer