import glob
import pickle
from prepare_data import DataHandler
import re

with open("corpus_large", "rb") as f:
    corpus = pickle.load(f)

notes = []
for octave in range(10):
    for i in range(12):
        notes.append(f'{chr(i+65)}_{str(octave)}')
dynamics = ["p", "P", "f", "F"]
res = []
for song in corpus:
    song = song.rstrip("\n")
    song_as_list = song.split(" ")
    for note_offset in range(-6,6):
        for dynamic_offset in range(-1,2):
            tmp = ""
            for note_octave_delta_vel in song_as_list:
                vel = note_octave_delta_vel[-1]
                note,octave,delta,velocity = note_octave_delta_vel.split("_")
                note_octave = note_octave_delta_vel[:3]
                idx_notes = max(0, min(len(notes)-1, (notes.index(note_octave) + note_offset)))
                if vel ==":":
                    new_dynamic = ":"
                else:
                    idx_dynamic = max(0, min(len(dynamics)-1, dynamics.index(vel)+dynamic_offset))
                    new_dynamic = dynamics[idx_dynamic]
                new_notes = notes[idx_notes]
                tmp += f'{new_notes}_{delta}_{new_dynamic} '
            tmp = tmp[:-1]
            tmp_list = tmp.split(" ")
            for i in range(len(tmp_list)//512+1):
                partition = ' '.join(tmp_list[i*512:(i+1)*512])
                res.append(partition)
            
with open("corpus_large_augmented", 'wb') as f:
    pickle.dump(res, f)