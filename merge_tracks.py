from mido import Message, MidiFile, MidiTrack, MetaMessage
import numpy as np
import glob,os
import bisect

def merge_tracks(midi, name):
    req_tempo = 100000
    candidates =[]
    for track in midi.tracks:
        for msg in track:
            if msg.type == "program_change":
                if 0<=msg.program<=7 and len(track)>20:
                    candidates.append(track)
            elif msg.type == "set_tempo":
                orig_time_factor = msg.tempo / req_tempo
    msgs = []
    for track in candidates:
        current_time = 0
        time_factor = orig_time_factor
        for msg in track:
            if msg.type == "change_tempo":
                time_factor = m.tempo / req_tempo
            current_time += int(msg.time * time_factor)
            if msg.type == "note_on" or msg.type == "note_off":
                msgs.append([current_time, msg.note, msg.velocity])
    msgs = sorted(msgs)
    merged_midi = MidiFile(ticks_per_beat=midi.ticks_per_beat)
    track = MidiTrack()
    merged_midi.tracks.append(track)
    track.append(MetaMessage('set_tempo', tempo = req_tempo))

    prev_time = 0
    for msg in msgs:
        if prev_time == 0:
            delta = 0
        else:
            delta = min(msg[0] - prev_time, 10000)
        note, vel = msg[1], msg[2]
        track.append(Message('note_on', note=note, velocity=vel, time=delta))
        prev_time = msg[0]
    if candidates:
        merged_midi.save(f"midi_files_merged/{name}.mid")
        
if __name__ == '__main__':
    for mid in glob.glob(f"midi_files_unmerged/*.mid", recursive=True):
        name = mid.split("/")[1]
        name = name[:-4]
        try:
            mid = MidiFile(mid)
            merge_tracks(mid, name)
        except Exception:
            print(f'File {name} is corrupt')