import pickle
import tensorflow as tf
from keras.models import load_model
import numpy as np
from prepare_data import DataHandler
import re
import sentencepiece as sp
from tokenizers import Tokenizer
from collections import Counter
from mido import Message, MidiFile, MidiTrack, MetaMessage
from collections import defaultdict, deque, Counter
import argparse

def clean_string_input(str_input):
    res = str_input.split(" ")
    res = [l for l in res if Counter(l)["_"]==3]
    clean_str = ""
    possible_notes = "ABCDEFGHIJKL"
    possible_vels = "pPfF:"
    for notes in res:
        note, octave, delta, vel = notes.split("_")
        if note in possible_notes and octave.isdigit() and delta.isdigit() and vel in possible_vels:
            clean_str += notes + " "
    return clean_str[:-1]

def generate_song(midifile, length, song_name):
    DH = DataHandler()

    tokenizer=Tokenizer.from_file("tokenizer_2048.json")

    model = load_model('vivaldi-v2-large-v2')
    SEQ_LENGTH = 511

    song = MidiFile(f"{midifile}")
    song = tokenizer.encode(DH.convert_midi_to_string(song)).ids[:511]
    no_tokens = len(song)
    song = song + [0]*(SEQ_LENGTH - no_tokens)
    prompt = np.array(song)[:SEQ_LENGTH].reshape(1, SEQ_LENGTH)
    res = []
    i = no_tokens
    for _ in range(length):
        i=min(i, SEQ_LENGTH - 1)
        frequency_tokens = Counter(prompt[0, -100:])
        prompt = tf.convert_to_tensor(prompt, dtype = tf.uint16)
        output = model(prompt, training=False).numpy().reshape(SEQ_LENGTH, -1)[i]
        for seen in prompt.numpy()[0,-100:]:
            output[seen] *= np.exp(-frequency_tokens[seen]**2/4)
        output = np.exp(0.85*np.log(output+1e-12))/np.sum(np.exp(0.85*np.log(output+1e-12)), axis=0).reshape(-1,1)
        output = np.squeeze(output)
        max_prob = np.max(output)
        cut_off = 0.15*max_prob
        candidates = np.where(output > cut_off)[0]
        probs = output[candidates]
        probs = probs / np.sum(probs)
        chosen_token = np.random.choice(candidates, size = 1, p = probs)
        res.append(int(chosen_token))
        if i>=SEQ_LENGTH - 1:
            prompt = np.hstack((prompt[0,1:].numpy().reshape(1,-1), np.array(chosen_token).reshape(1,1)))
        else:
            prompt = prompt.numpy()
            prompt[0, i+1] = chosen_token
        i += 1
    res = tokenizer.decode(res)
    res = clean_string_input(res)
    DH.string_to_midi(res, f'{song_name}')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate music using a midi file as prompt"
    )
    parser.add_argument("midi")
    parser.add_argument("--length", type=int, default=1000)
    parser.add_argument("--output", type=str, default="MySong")
    args = parser.parse_args()
    generate_song(args.midi, args.length, args.output)