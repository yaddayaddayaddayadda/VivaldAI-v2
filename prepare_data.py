import mido
from mido import Message, MidiFile, MidiTrack, MetaMessage
import numpy as np
import glob,os
import matplotlib.pyplot as plt
from tensorflow.data import Dataset
from collections import defaultdict, Counter
import pickle 
import re
import tensorflow as tf 
from keras.models import load_model

SRC_DIRECTORY = 'midi_files'
SILENT_NOTE_VEL = 255
VEL_LOOKUP = {
    "p" : 16,
    "P" : 48,
    "f" : 80,
    "F" : 112
}
class DataHandler:
    def __init__(self):
        pass
    def convert_midi_to_string(self, midi):
        maxi = 0
        chosen_track = None
        for track in midi.tracks:
            if len(track)>maxi:
                maxi = len(track)
                chosen_track = track
        req_tempo = 100000
        orig_time_factor = 960/midi.ticks_per_beat
        time_factor = orig_time_factor
        res = ""
        delta = 0
        prev_delta = 0
        for m in midi.tracks[0]:
            if m.type == 'set_tempo':
                time_factor = orig_time_factor * m.tempo / req_tempo
                break
        for index, m in enumerate(chosen_track):
            if m.type == 'set_tempo':
                time_factor = orig_time_factor * m.tempo / req_tempo
            delta += int(m.time * time_factor)
            if m.type == 'note_on' and m.velocity > 0:
                note = m.note
                octave, note = divmod(note, 12)
                note = chr(note+65)
                vel = m.velocity
                if 0 < vel < 32:
                    vel_quantized = "p"
                elif 32 <= vel < 64:
                    vel_quantized = "P"
                elif 64 <= vel < 96:
                    vel_quantized = "f"
                else:
                    vel_quantized = "F"
                if prev_delta == 0:
                    res += f'{note}_{str(octave)}_{0}_{vel_quantized} '
                else:
                    res += f'{note}_{str(octave)}_{round(delta - prev_delta, -2)}_{vel_quantized} '
                prev_delta = delta
            elif (m.type == 'note_on' and m.velocity == 0) or m.type == "note_off":
                note = m.note
                octave, note = divmod(note, 12)
                note = chr(note+65)
                res += f'{note}_{str(octave)}_{round(delta - prev_delta, -2)}_: '
                prev_delta = delta
        return res[:-1]

    def string_to_midi(self, song, song_name="test"):
        song = song.split(" ")
        mid = MidiFile(ticks_per_beat=960)
        track = MidiTrack()
        mid.tracks.append(track)
        track.append(MetaMessage('set_tempo', tempo = 100000))
        events = []
        time = 0
        for notes in song:
            try:
                note, octave, delta, vel = notes.split("_")
                if vel != ":":
                    note, delta, vel = ord(note) + 12 * int(octave) - ord("A"), int(delta), VEL_LOOKUP[vel]
                else:
                    note, delta, vel = ord(note) + 12 * int(octave) - ord("A"), int(delta), 0
                events.append((delta, note, vel))
            except Exception:
                print(f'{notes} is not a valid event')
        zeros, nonzeros =0,0
        events = events[::-1]
        while events:
            delta, note, vel = events.pop()
            if vel==0:
                track.append(Message('note_off', note=note, velocity=vel, time=delta))
            else:
                track.append(Message('note_on', note=note, velocity=vel, time=delta))
        mid.save(f'{song_name}.mid')


D = DataHandler()
def generate_arrays():
    c = 0
    for midi in glob.glob("midi_files_large/*.mid", recursive=True):
        try:
            arr = D.convert_midi_to_array(MidiFile(midi))
            np.save(f"np_arrays_large/arr_{c}.npy", arr)
            c += 1
        except Exception:
            print(f'Midi {midi} is corrupt')
def generate_corpus():
    res = []
    for arr in glob.glob("np_arrays_large/*.npy", recursive=True):
        arr = np.load(arr)
        res.append(D.array_to_string(arr))
    with open("corpus", "wb") as f:
        pickle.dump(res, f)
def generate_tokenizer():
    with open(f"corpus_large", 'rb') as f:
        corpus = pickle.load(f)
    for vocab_size in [4096]:
        for ngram_max in [8]:
            B = Encoder(vocab_size, pct_bpe=0.95, ngram_min = 1, ngram_max = ngram_max, required_tokens = ["#_"])
            B.fit(corpus)
            pad_idx = B.word_vocab["__pad"]
            if pad_idx != 0:
                zero_token = B.inverse_word_vocab[0]
                B.word_vocab["__pad"], B.word_vocab[zero_token] = 0, pad_idx
                B.inverse_word_vocab[0], B.inverse_word_vocab[pad_idx] = "__pad", zero_token
            assert B.word_vocab["__pad"] == 0
            with open(f"tokenizer_{vocab_size}_{ngram_max}.pkl", 'wb') as f:
                pickle.dump(B, f)
