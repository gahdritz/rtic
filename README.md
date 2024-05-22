# Modeling Real-Time Interactive Conversations as Timed Diarized Transcripts

Code for the paper "Modeling Real-Time Interactive Conversations as Timed Diarized Transcripts"

Demo videos coming soon!

## Chat application

Code to convert Facebook Messenger chat dumps can be found in `utils/chat_utils.py'. Run `load_chat' to load the dump and then `convert_chat' to compute message prefixes. See `messenger/interactive_messenger.py,' which implements Algorithm 1, for an example.

## Speech application

Code to convert WhisperX speech transcriptions can also be found in `utils/chat_utils.py'. Run `load_transcriptions' to load a directory of transcripts and then `convert_transcriptions' to compute prefixes. See `speech/interactive_speech.py' for a worked demo.
