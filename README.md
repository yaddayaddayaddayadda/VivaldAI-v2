# VivaldAI-v2
Generating classical music using the decoder part of a transformer network with 85 million parameters and a context window of 512 tokens. 
The data consists of 1800 classical music pieces ini MIDI format. Each song has been augmented 36 times, transposing the key 12 times and the dynamics 3 times. 

# Usage
Invoke the CLI script to generate music using a MIDI file as prompt. Additional parameters are the number of tokens to generate (length) and the name of the output file (output).
Example: 
python inference.py <YourMidi> --length 1000 --output MySong
