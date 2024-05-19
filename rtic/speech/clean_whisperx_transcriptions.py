import os

for root, _, f in os.walk("data/transcriptions/scotus_whisperx_transcripts_2"):
    for file in f:
        if file.endswith(".tsv"):
            with open(os.path.join(root, file), "r") as f:
                lines = [l.strip() for l in f.readlines()]

            # Remove column titles
            lines = lines[1:]

            clean_lines = []
            for l in lines:
                l_split = l.split('\t')
                assert(len(l_split) == 4)
                start, end, content, speaker = l_split
                clean_lines.append('\t'.join([start, end, content, speaker]))

            new_filename = file.rsplit('.')[0] + "_clean.tsv"
            with open(os.path.join(root, new_filename), "w") as f:
                f.write('\n'.join(clean_lines))
