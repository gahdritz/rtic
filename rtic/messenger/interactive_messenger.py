import datetime
import random
import queue
import re
import string
import sys
import threading
import time

import torch
from torch.distributions import Categorical
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer,
    StoppingCriteria,
)

from ..utils.chat_utils import (
    convert_timestamp,
    load_chat,
    convert_chat_with_datetime,
)


torch.manual_seed(50)

CHAT_DUMP = "data/chat_data/dump2.json"

MODELS = [
    ("scratch/llm_foundry/llama_2_new_hf", "---------+"),
    ("scratch/llm_foundry/pythia-160m-hf", "][^"),
    ("scratch/llm_foundry/pythia-1.4b-hf", "][^"),
    ("scratch/llm_foundry/pythia-12b-hf", "][^"),
    ("scratch/llm_foundry/gemma-2b-hf", "________,"),
]

MODEL_DIR, EOM = MODELS[0]
CHUNK_FILE = "out/short_chunks_idx.txt"
PROMPT_IDX = 4

MAX_LEN = 2048
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
if(DEVICE == "cpu"):
    print("WARNING: Running on CPU.")

SPEAKERS = ['A', 'B']
YEARS = ['2014', '2015', '2016', '2017', '2018', '2019', '2020', '2021', '2022', '2023', '2024']
MONTHS = ['J', 'Fe', 'Ma', 'A', 'May', 'J', 'Jul', 'August', 'S', 'O', 'N', 'D']
WEEKDAYS = ['M', 'Tu', 'W', 'Th', 'F', 'Sa', 'Su']
DAYS = ["{:02d}".format(i) for i in range(1, 32)]
WEEKDAYS = ['M', 'Tu', 'W', 'Th', 'F', 'Sa', 'Su']
HOURS = ["{:02d}".format(i) for i in range(24)]
SIXTIETHS = ["{:02d}".format(i) for i in range(60)]
DIGITS = [str(i) for i in range(10)]

DELIMITERS = [
    ('.', 'decisecond'), # second.decisecond
    (';', 'second'), # minute;second
    (':', 'minute'), # hour:minute
    ('+', 'hour'), # weekday+hour
]

FIRST_MESSAGE = "lol"

CHAT_WIDTH = 64


class EOMCriteria(StoppingCriteria):
    def __init__(self, EOM_TOKS):
        self.EOM_TOKS = set(EOM_TOKS)

    def __call__(self, input_ids, scores, **kwargs):
        if input_ids[:,-1].item() in self.EOM_TOKS:
            return True
        return False  # Continue generation

    def __len__(self):
        return 1

    def __iter__(self):
        yield self


def parse_next_message(next_message_str, prev_dt):
    new_dt = prev_dt
    i = 0
    delimiter = next_message_str[i]
    i += 1

    short_term_delimiters = ['+', ':', ';', '.']
    assert(delimiter in set(short_term_delimiters))
    start_idx = short_term_delimiters.index(delimiter)
    
    lengths = {
        '+': 2,
        ':': 2,
        ';': 2,
        '.': 1,
    }
    delimiter_dict = {k:v for k,v in DELIMITERS}
    while start_idx < len(short_term_delimiters):
        delimiter = short_term_delimiters[start_idx]
        delimiter_name = delimiter_dict[delimiter]
        l = lengths[delimiter]
        timestamp = int(next_message_str[i:i+l])
        
        if(delimiter_name == "decisecond"):
            delimiter_name = "microsecond"
            timestamp *= int(1e5)

        update = {delimiter_name: timestamp}
        new_dt = new_dt.replace(**update)

        start_idx += 1
        i += l
    
    speaker_id = None
    speaker_id = next_message_str[i]
    assert(speaker_id in string.ascii_uppercase)
    next_message_str = next_message_str[i+1:]

    return speaker_id, new_dt, next_message_str


def generate_next_message(history_tokens, model, tokenizer, speaker_id_toks, eom_tok):
    prompt = history_tokens
    generation = model.generate(
        prompt,
        attention_mask=prompt.new_ones(prompt.shape),
        max_new_tokens=100,
        do_sample=True,
        top_k=0, 
        top_p=0.95,
        temperature=1,
        pad_token_id=tokenizer.eos_token_id,
        stopping_criteria=EOMCriteria(speaker_id_toks)
    )

    assert(generation[..., -1].item() in speaker_id_toks)

    generation = model.generate(
        generation,
        attention_mask=generation.new_ones(generation.shape),
        max_new_tokens=100,
        do_sample=True,
        top_k=0, 
        top_p=0.95,
        temperature=1,
        pad_token_id=tokenizer.eos_token_id,
        stopping_criteria=EOMCriteria([eom_tok])
    )

    return generation[:, history_tokens.shape[-1]:]


def to_ms(dt):
    return (dt - datetime.datetime.utcfromtimestamp(0)).total_seconds() * 1000


def output_thread(
    input_queue,
    stop_queue,
    user_id,
    model,
    tokenizer,
    prompt,
    start_dt,
):
    start_dt_str = convert_timestamp(to_ms(start_dt))
    
    converted, start_dt = convert_chat_with_datetime(prompt, delimiter=EOM)
    prefixes = [m[:pl] for (_, _, pl, m) in converted]
    content = [m[pl:] for (_, _, pl, m) in converted]

    prefix_tok = [tokenizer.encode(p, return_tensors='pt', add_special_tokens=False) for p in prefixes]
    content_tok = [tokenizer.encode(c, return_tensors='pt', add_special_tokens=False) for c in content]
    tokenized = torch.cat([v for t in zip(prefix_tok, content_tok) for v in t], dim=-1)

    dummy = tokenizer.encode("dummy", return_tensors='pt', add_special_tokens=True)
    dummy_no_bos = tokenizer.encode("dummy", return_tensors='pt', add_special_tokens=False)
    if(dummy_no_bos[:, 0] != dummy[:, 0]):
        tokenized = torch.cat([dummy[:, :1], tokenized], dim=-1)

    history_tokens = tokenized.cuda()

    print(tokenizer.batch_decode(history_tokens)[0].replace(EOM, " <eom> "))

    model_id = 'B' if user_id == 'A' else 'A'
    model_id_tok = tokenizer.encode(
        f"bla{model_id}",
        add_special_tokens=False,
        return_tensors='pt',
    )[0, -1].to(DEVICE)
    speaker_id_toks = [
        tokenizer.encode(
            f"1234{s}",
            add_special_tokens=False,
            return_tensors='pt',
        )[0, -1].item() for s in ['A', 'B']
    ]


    eom_tok = tokenizer.encode(
        f"bla{EOM}",
        add_special_tokens=False,
        return_tensors='pt',
    )[:, -1:]

    def should_exit():
        try:
            stop = stop_queue.get(block=False)
            return True
        except queue.Empty:
            return False


    real_start_time = datetime.datetime.now()
    prev_dt = start_dt
    last_speaker_id = prompt[-1][0]
    while True:
        next_message = generate_next_message(
            history_tokens, 
            model,
            tokenizer,
            speaker_id_toks,
            eom_tok.item(),
        )
        
        next_message_str = tokenizer.batch_decode(next_message, skip_special_tokens=True)[0]

        speaker_id, new_dt, message_body = parse_next_message(next_message_str, prev_dt)

        time_diff = new_dt - prev_dt
        time_diff_sec = time_diff.total_seconds()

        if(time_diff_sec < 0):
            continue

        try:
            message = input_queue.get(block=True, timeout=time_diff_sec)
            if(should_exit()):
                break

            new_user_dt = start_dt + (datetime.datetime.now() - real_start_time)
            new_user_dt_str = convert_timestamp(to_ms(new_user_dt), prev_dt)
            next_user_message_prefix = f"{new_user_dt_str}{user_id}"
            next_user_message_content = f"{message}{EOM}"
            next_user_message_tokens = torch.cat([
                tokenizer.encode(
                    next_user_message_prefix,
                    return_tensors='pt',
                    add_special_tokens=False,
                ),
                tokenizer.encode(
                    next_user_message_content,
                    return_tensors='pt',
                    add_special_tokens=False,
                ),
            ], dim=-1).to(DEVICE)

            history_tokens = torch.cat([
                history_tokens,
                next_user_message_tokens
            ], dim=-1)
            
            last_speaker_id = user_id
            prev_dt = new_user_dt 
            input_queue.task_done()
        except queue.Empty as e:
            if(should_exit()):
                break

            # Don't speak for the user
            if(speaker_id == user_id):
                continue

            print(message_body.rjust(CHAT_WIDTH).strip(EOM))

            history_tokens = torch.cat([history_tokens, next_message], dim=-1)
            prev_dt = new_dt
            last_speaker_id = model_id


def interact_as(user_id, model, tokenizer, prompt):
    input_queue = queue.Queue()
    stop_queue = queue.Queue()

    start_dt = datetime.datetime.now()

    output_thread_obj = threading.Thread(
        target=output_thread, 
        args=(input_queue, stop_queue, user_id, model, tokenizer, prompt, start_dt)
    )
    output_thread_obj.start()
    
    try:
        while True:
            line = input()
            input_queue.put(line)
    except:
        stop_queue.put(0)



if __name__ == "__main__":
#    saved_stdout = sys.stdout
#    saved_stderr = sys.stderr
#    sys.stdout = open('trash', 'w')
#    sys.stderr = open('trash_err', 'w')
    model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, torch_dtype=torch.bfloat16)
#    sys.stdout = saved_stdout
#    sys.stderr = saved_stderr
    
    model.cuda()
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    with open(CHUNK_FILE, "r") as fp:
        chunk_indices = [c.strip() for c in fp.readlines()]
   
    chunk_indices = [tuple(map(int, c.split(','))) for c in chunk_indices]

    chat = load_chat(CHAT_DUMP)
    chat = chat[int(len(chat) * 0.95):]
    chat = [c for c in chat if c[-1] is not None]

    prompt_s, prompt_e = chunk_indices[PROMPT_IDX]
    prompt = chat[prompt_s: prompt_e]
    prompt = prompt[:int(len(prompt) // 4)]

    msgs = interact_as('A', model, tokenizer, prompt)
