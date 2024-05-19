import json
from jsonargparse import CLI
from typing import Optional
import string

from tokenizers import Tokenizer

from ..utils.chat_utils import (
    convert_transcriptions,
    load_transcriptions,
)


""" Measures the minimum generation rate given a chat transcript and a reaction time """


MS_PER_S = 1000


def main(
    chat_dir: str,
    reaction_time_ms: int,
    tokenizer_path: Optional[str] = None,
    tokenizer_name_hf: Optional[str] = None,
):
    """
        Args:
            chat_dir:
                Path to directory full of .tsv transcript files. Transcript
                files contain one line per word formatted as follows:
                "start_time_s<\t>end_time_s<\t>word<\t>speaker_id"
            reaction_time_ms:
                Messages within reaction_time_ms milliseconds of a message do
                not count toward generation rate requirements
            tokenizer_path:
                Local tokenizer file
            tokenizer_name_hf:
                Name of the HuggingFace tokenizer to use
    """
    if(tokenizer_path is not None and tokenizer_name_hf is not None):
        raise ValueError("Only one of tokenizer_path and tokenizer_name_hf can be specified.")
    elif(tokenizer_path is not None):
        tokenizer = Tokenizer.from_file(tokenizer_path)
    elif(tokenizer_name_hf is not None):
        tokenizer = Tokenizer.from_pretrained(tokenizer_name_hf)
    else:
        raise ValueError("Bad conditions")

    # Load chat as (user, timestamp, prefix_len, content) tuples
    raw_chats = load_transcriptions(chat_dir) 
    chats = convert_transcriptions(
        raw_chats,
    )

    users = string.ascii_uppercase

    char_counts = []
    speaker_counts = []
    seconds = []
    windows = []
    token_lengths = []
    token_lengths_prefix = []
    for chat in chats:
        most_recent_timestamps = {user: [] for user in users}
        speakers = set()
        for msg in chat:
            user, start_s, stop_s, prefix_len, content = msg

            if(len(content) == prefix_len):
                continue

            timestamp_ms = start_s * 1000
            user = users[int(user)]

            speakers.add(user)
    
            # Find the timestamp of the most recent message at least {reaction_time}ms ago
            most_recent_timestamp_in_range = -1
            for other_user in most_recent_timestamps:
                idx = -1
                old_msg_timestamps = most_recent_timestamps[other_user]
                for old_msg_timestamp in old_msg_timestamps:
                    with_delay = old_msg_timestamp + reaction_time_ms
                    
                    # Not enough time to respond
                    if(with_delay > timestamp_ms):
                        break
    
                    idx += 1
    
                if(idx >= 0):
                    local_most_recent_timestamp_in_range = old_msg_timestamps[idx]
                    if(local_most_recent_timestamp_in_range > most_recent_timestamp_in_range):
                        most_recent_timestamp_in_range = local_most_recent_timestamp_in_range
                   
                    # Prune redundant times
                    most_recent_timestamps[other_user] = old_msg_timestamps[idx:]
    
            most_recent_timestamps[user].append(timestamp_ms)
    
            if(most_recent_timestamp_in_range >= 0):
                windows.append(timestamp_ms - most_recent_timestamp_in_range)
            else:
                windows.append(-1)
    
            tokenized_content = tokenizer.encode(content, add_special_tokens=False)
            tokenized_content_prefix = tokenizer.encode(content[:prefix_len], add_special_tokens=False) 
            
            token_lengths.append(len(tokenized_content.ids))
            token_lengths_prefix.append(len(tokenized_content_prefix.ids))

            char_counts.append(len(content[prefix_len:]))

        speaker_counts.append(len(speakers))
        seconds.append(stop_s)

    ratios = list(sorted([tl / (tl - tlp) for tl, tlp in zip(token_lengths, token_lengths_prefix)]))
    print(ratios[int(len(ratios) // 2)])
    print(sum(ratios) / len(ratios))
    print(sum(token_lengths))
    print(sum(speaker_counts) / len(speaker_counts))
    print(sum(seconds))
    print(f"char count: {sum(char_counts)}")

    zipped = [t for t in zip(token_lengths, windows) if t[1] >= 0]
    rates = [token_length / window for token_length, window in zipped]
    
    # Convert to s
    rates = [r * MS_PER_S for r in rates]

    rates = sorted(rates)
    percentiles = [0.5, 0.9, 0.95, 0.99, 0.999, 1]
    for p in percentiles:
        idx = int((len(rates) - 1) * p)
        print(f"{p * 100}-th percentile: {rates[idx]}")


if __name__ == "__main__":
    CLI(main)
