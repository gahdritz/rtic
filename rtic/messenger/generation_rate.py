import json
from jsonargparse import CLI
from typing import Optional

from tokenizers import Tokenizer

from ..utils.chat_utils import (
    load_chat,
    convert_chat,
)


""" Measures the minimum generation rate given a chat transcript and a reaction time """


MS_PER_S = 1000


def main(
    chat_file: str,
    reaction_time_ms: int,
    tokenizer_path: Optional[str] = None,
    tokenizer_name_hf: Optional[str] = None,
    msg_delimiter: str = "----",
):
    """
        Args:
            chat_file:
                Path to .json file containing chat data
            reaction_time_ms:
                Messages within reaction_time_ms milliseconds of a message do
                not count toward generation rate requirements
            tokenizer_path:
                Local tokenizer file
            tokenizer_name_hf:
                Name of the HuggingFace tokenizer to use
            msg_delimiter:
                Separator to insert between consecutive messages
    """
    if(tokenizer_path is not None and tokenizer_name_hf is not None):
        raise ValueError("Only one of tokenizer_path and tokenizer_name_hf can be specified.")
    elif(tokenizer_path is not None):
        tokenizer = Tokenizer.from_file(tokenizer_path)
    elif(tokenizer_name_hf is not None):
        tokenizer = Tokenizer.from_pretrained(tokenizer_name_hf)
    else:
        raise ValueError("Bad conditions")

    if(len(tokenizer.encode(msg_delimiter).ids) != 1):
        print(f"WARNING: Delimiter {msg_delimiter} is not a single token")

    # Load chat as (user, timestamp, prefix_len, content) tuples
    chat = load_chat(chat_file) 
    chat = convert_chat(
        chat,
        delimiter=msg_delimiter,
    )

    # Filter
    chat = [msg for msg in chat if msg[2] != len(msg[3])]

    users = set([msg[0] for msg in chat])

    most_recent_timestamps = {user: [] for user in users}
    windows = []
    token_lengths = []
    for msg in chat:
        user, timestamp_s, _, content = msg
        timestamp_ms = timestamp_s * 1000

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
        token_lengths.append(len(tokenized_content.ids))


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
