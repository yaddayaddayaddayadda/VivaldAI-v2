# VivaldAI-v2
Generating classical music using the decoder part of a transformer network with 85 million parameters and a context window of 512 tokens. 
The training data consists of 2500 mostly classical music pieces ini MIDI format. Each song has been augmented 36 times, transposing the key in 12 ways and altering the dynamics in 3 ways. 

# Usage
Invoke the CLI script to generate music using a MIDI file as prompt. Additional parameters are the number of tokens to generate (length) and the name of the output file (output).

python inference.py YourMidi --length 1000 --output MySong

# Examples:

Using the first 30 seconds of Chopin's Nocturne Op.27 No.2:

https://github.com/yaddayaddayaddayadda/VivaldAI-v2/assets/45805059/0962e713-1a33-4e6f-8d88-1cd50d0d667d

