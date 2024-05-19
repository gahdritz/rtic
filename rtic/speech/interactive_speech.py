import os
import queue
import random
import select
import string
import sys
import threading

import torch
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer,
    StoppingCriteria,
)


QUEUE = queue.Queue()

LLAMA_SPACES = [259, 29871]
GEMMA_SPACES = [235248]


MODELS = [
    ("scratch/llm_foundry/llama_2_hf_scotus", "---------+"),
    ("scratch/llm_foundry/pythia-160m-scotus-hf", "][^"),
    ("scratch/llm_foundry/pythia-1.4b-scotus-hf", "][^"),
    ("scratch/llm_foundry/pythia-12b-hf-scotus", "][^"),
    ("scratch/llm_foundry/gemma-2b_hf_scotus", "________,"),
]

IDX = 4
with open("model_sequence.txt", "r") as fp:
    l = [li.strip() for li in fp.readlines()]

MODEL_DIR, EOM = MODELS[int(l[IDX])]


MAX_LEN = 2048
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
if(DEVICE == "cpu"):
    print("WARNING: Running on CPU.")

# Load the model and its tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)

model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, torch_dtype=torch.bfloat16, attn_implementation="eager")
model.eval()
model.to(DEVICE)

HUMAN_ID = "E"
HUMAN_ID_TOK = tokenizer.encode(f" {HUMAN_ID}", add_special_tokens=False)[-1]

CHAT_WIDTH = 64


def read_lines_into_queue():
    #fifo = os.open(sys.stdin, os.O_RDONLY | os.O_NONBLOCK)
    #fifo_file = os.fdopen(fifo)
    fifo = sys.stdin
    while True:
        select.select([fifo],[],[fifo])
        line = fifo.readline()
        QUEUE.put(line)


def compute_time_since_last_message(timestamp_float, last_timestamp):
    if(timestamp_float > last_timestamp):
        time_since_last_message = timestamp_float - last_timestamp
    else:
        time_since_last_message = 10 + timestamp_float - last_timestamp

    return time_since_last_message


def prune_kv_cache(kv_cache, entries_to_remove):
    pruned = (
        [pk[:, :, :pk.shape[-2] - entries_to_remove, :] for pk in kv_cache[0]],
        [pv[:, :, :pv.shape[-2] - entries_to_remove, :] for pv in kv_cache[0]],
    )

    if(pruned[0][0].shape[-2] == 0):
        return None

    return pruned


def format_lines(lines, start_time):
    prefixes = []
    words = []
    times = []
    for i, line in enumerate(lines):
        idx, ts, word = line.strip().split()

        times.append(float(ts))

        # Skip retconning for now
        if(int(idx) < 0):
            continue

        f = ' '
        if(i == 0):
            f += f"{HUMAN_ID}"
        
        ts = '{:.2f}'.format(float(ts) - start_time)
        ts_decimals = ts.split(".")[1]
        ts_last_digit = str(int(float(ts)) % 10)

        f += f"{ts_last_digit}{ts_decimals}"
        prefixes.append(f)
        words.append(word)

    return prefixes, words, times


def sample_from_allowed_tokens(logits, allowed_tokens):
    """ Sample from the logits, but only allow tokens in allowed_tokens """
    logits = torch.nn.functional.softmax(logits, dim=-1)

    if(allowed_tokens is not None):
        logits = logits[:, allowed_tokens] # no need to normalize
        tok = allowed_tokens[torch.multinomial(logits, num_samples=1)]
    else:
        tok = torch.multinomial(logits, num_samples=1).item()

    return logits.new_tensor([[tok]], dtype=torch.long)


def sample(
    model,
    prompt, 
    start_tokens_speaker,
    start_tokens_digit,
    digit_tokens, 
    max_tokens=30,  
    past_key_values=None,
    user_just_spoke=False,
    stop_tokens=None
):
    """ Sample autoregressively from the model using model.forward() """
    generated = prompt
    if(past_key_values is not None):
        overhang = prompt.shape[-1] - past_key_values[0][0].shape[-2]
        print(f"OVERHANG: {overhang}")
        prompt = prompt[:, -overhang:]
    for i in range(max_tokens):
        out = model.forward(
            prompt,
            past_key_values=past_key_values,
            use_cache=True,
        )

        logits = out.logits[:, -1, :]

        if("llama" in MODEL_DIR):
            if(i == 0):
                allowed_tokens = LLAMA_SPACES # space token
            elif(1 <= i <= 3):
                allowed_tokens = digit_tokens
            elif(i == 4):
                allowed_tokens = start_tokens_speaker
            else:
                allowed_tokens = None
        elif("gemma" in MODEL_DIR):
            if(i == 0):
                allowed_tokens = GEMMA_SPACES
            elif(1 <= i <= 3):
                allowed_tokens = digit_tokens
            elif(i == 4):
                allowed_tokens = start_tokens_speaker
            else:
                allowed_tokens = None
        elif("pythia" in MODEL_DIR):
            if(i == 0):
                allowed_tokens = start_tokens_digit
            elif(1 <= i <= 2):
                allowed_tokens = digit_tokens
            elif(i == 3):
                allowed_tokens = start_tokens_speaker
            else:
                allowed_tokens = None
        else:
            ValueError()

        next_token = sample_from_allowed_tokens(logits, allowed_tokens)

        # Break if we hit a second start token
        if("gemma" in MODEL_DIR):
            if(i != 0 and next_token.item() in GEMMA_SPACES):
                break
        elif("pythia" in MODEL_DIR):
            if i != 0 and next_token.item() in (start_tokens_digit + start_tokens_speaker + stop_tokens):
                break
        elif("llama" in MODEL_DIR):
            if(i != 0 and next_token.item() in LLAMA_SPACES):
                break
            if(i == 4 and next_token.item() == HUMAN_ID_TOK):
                # Remove the space
                past_key_values = prune_kv_cache(past_key_values, 4)
                generated = generated[:, :-4]
                break

        # past_key_values -> (bs, num_heads, seq_len, head_dim)
        past_key_values = out.past_key_values
        
        prompt = next_token
        generated = torch.cat([generated, next_token], dim=-1)

    return generated, past_key_values, speaker_present


# Launch the FIFO reader thread
fifo_reader = threading.Thread(target=read_lines_into_queue, daemon=True)
fifo_reader.start()

# Used for structured decoding
start_tokens_speaker = list(set([
    tokenizer.encode(f" {c}", add_special_tokens=False)[-1]
    for c in [au for au in string.ascii_uppercase if au != HUMAN_ID]
]))
start_tokens_digit = list(set([
    tokenizer.encode(f" {c}", add_special_tokens=False)[-1]
    for c in "0123456789"
]))

stop_tokens = None
if("pythia" in MODEL_DIR):
    stop_tokens = list([
        tokenizer.encode(' ' + '{:03d}'.format(s), add_special_tokens=False)
        for s in range(0, 999)
    ])
    stop_tokens = list(set([st[0] for st in stop_tokens if len(st) == 1]))

digit_tokens = [
    tokenizer.encode(c, add_special_tokens=False)[-1]
    for c in "0123456789"
]

# Start of the conversation (not very principled, but you need to
# kickstart the LM)
history = tokenizer.encode(
    f" 121H alabama",
    return_tensors="pt",
).to(DEVICE)

# Get the start time. All timestamps will be relative to it
print("Waiting for time...")

start_time = QUEUE.get()
print(start_time)
#start_time = 100
start_time = float(start_time)

# Time accounting
cur_time = start_time
last_timestamp = 0
global_times = [] # list of message times

past_key_values = None
user_just_spoke = False
while True:
    with torch.no_grad():
        if(QUEUE.empty()):
            out, past_key_values, speaker_present = sample(
                model,
                history,
                start_tokens_speaker,
                start_tokens_digit,
                digit_tokens,
                max_tokens=30,
                #past_key_values=past_key_values,
                user_just_spoke=user_just_spoke,
                stop_tokens=stop_tokens,
            )
    
            new_tokens = out[:, history.shape[-1]:]

            #print([(t.item(), tokenizer.decode(t)) for t in new_tokens[0]])
    
            # model generated a message as the user
            if(new_tokens.shape[-1] == 0):
                continue
    
            if("llama" in MODEL_DIR):
                new_timestamp = new_tokens[0, 1:4]
            elif("gemma" in MODEL_DIR):
                new_timestamp = new_tokens[0, 1:4]
            elif("pythia" in MODEL_DIR):
                new_timestamp = new_tokens[0, 0:3]
            else:
                raise ValueError()
  
            timestamp_decoded = tokenizer.decode(new_timestamp).strip()
            if(not timestamp_decoded.isnumeric()):
                print(timestamp_decoded)
            assert(timestamp_decoded.isnumeric())
            assert(len(timestamp_decoded) == 3)
    
            s, ds, cs = timestamp_decoded
            timestamp_float = float(f"{s}.{ds}{cs}")
            time_to_wait = compute_time_since_last_message(
                timestamp_float, last_timestamp,
            )
            global_timestamp = cur_time + time_to_wait

        else:
            time_to_wait = 0

        try:
            message = QUEUE.get(block=True, timeout=time_to_wait)

            # User message arrived
            prefixes, words, user_times = format_lines([message], start_time)

            if(len(prefixes) < 1):
                continue # TODO (happens when retcon occurs)

            print((prefixes[0] + words[0]).rjust(CHAT_WIDTH))
            prefix_tokens = tokenizer.encode(prefixes[0], return_tensors='pt', add_special_tokens=False).to(DEVICE)
            word_tokens = tokenizer.encode(words[0], return_tensors='pt', add_special_tokens=False).to(DEVICE)
            history = torch.cat([history, prefix_tokens, word_tokens], dim=-1)
            last_time_digits, last_time_decimal = str(user_times[0] - start_time).split('.')
            last_timestamp = int(last_time_digits) % 10 + float(f"0.{last_time_decimal}")
            user_just_spoke = True
        except (queue.Empty, ValueError) as e:
            # No user message arrived
            print(tokenizer.decode(new_tokens[0]))
            history = torch.cat([history, new_tokens], dim=-1)
            last_timestamp = timestamp_float
            user_just_spoke = False

        if(history.shape[-1] > MAX_LEN):
            print("TOO LONG!")
            exit() # TODO. Need to adjust message boundaries. Kind of a mess
