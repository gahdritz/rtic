import random

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from ..utils.chat_utils import (
    convert_chat,
    load_chat,
)


MODELS = [
    ("scratch/llm_foundry/llama_2_hf", "---------+"),
    ("scratch/llm_foundry/pythia-160m-hf", "][^"),
    ("scratch/llm_foundry/pythia-1.4b-hf", "][^"),
    ("scratch/llm_foundry/pythia-12b-hf", "][^"),
    ("scratch/llm_foundry/gemma-2b-hf", "________,"),
]

IDX = 0

with open("model_sequence.txt", "r") as fp:
    sequence = [li.strip() for li in fp.readlines()]

MODEL, DELIMITER = sequence[IDX]

CHAT_DUMP = "data/chat_data/dump2.json"
#CHECKPOINT_DIR = "scratch/llm_foundry/gemma-2b-hf"
#CHECKPOINT_DIR = "scratch/llm_foundry/llama_2_hf"
CHECKPOINT_DIR = "scratch/llm_foundry/pythia-12b-hf"
CHUNK_FILE = "out/short_chunks_idx.txt"
TEMPERATURE = 1
MAX_TOKENS = 500

# Llama 2
#DELIMITER = "---------+"

# Pythia
DELIMITER = "][^"

# Gemma
#DELIMITER = "________,"

OUTPUT = f"scratch/llm_foundry/pythia-12b_hf_conditional_generations_temp_{TEMPERATURE}.txt"


def main():
    model = AutoModelForCausalLM.from_pretrained(CHECKPOINT_DIR, torch_dtype=torch.bfloat16)
    model.cuda()
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT_DIR)

    with open(CHUNK_FILE, "r") as fp:
        chunk_indices = [c.strip() for c in fp.readlines()]
   
    chunk_indices = [tuple(map(int, c.split(','))) for c in chunk_indices]

    chat = load_chat(CHAT_DUMP)
    chat = chat[int(len(chat) * 0.95):]
    chat = [c for c in chat if c[-1] is not None]

    fp = open(OUTPUT, "w", encoding="utf-8")
    
    for i, (s,e) in enumerate(chunk_indices):
        chunk = chat[s:e]
        converted = convert_chat(chunk, delimiter=DELIMITER)

        prefixes = [m[:pl] for (_, _, pl, m) in converted]
        content = [m[pl:] for (_, _, pl, m) in converted]

        prefix_tok = [tokenizer.encode(p, return_tensors='pt', add_special_tokens=False) for p in prefixes]
        content_tok = [tokenizer.encode(c, return_tensors='pt', add_special_tokens=False) for c in content]
        tokenized = torch.cat([v for t in zip(prefix_tok, content_tok) for v in t], dim=-1)

        dummy = tokenizer.encode("dummy", return_tensors='pt', add_special_tokens=True)
        dummy_no_bos = tokenizer.encode("dummy", return_tensors='pt', add_special_tokens=False)
        if(dummy_no_bos[:, 0] != dummy[:, 0]):
            tokenized = torch.cat([dummy[:, :1], tokenized], dim=-1)

        tokenized = tokenized.cuda()

        generation = model.generate(
            tokenized, 
            attention_mask=tokenized.new_ones(tokenized.shape),
            max_new_tokens=MAX_TOKENS, 
            do_sample=True, 
            top_p=0.95, 
            temperature=TEMPERATURE
        )
        generation = generation[:, tokenized.shape[-1]:]

        generation_decoded = tokenizer.batch_decode(generation, skip_special_tokens=True)

        # Prune incomplete messages
        generation_decoded = [DELIMITER.join(g.split(DELIMITER)[:-1]).strip(DELIMITER) for g in generation_decoded]

        for gen in generation_decoded:
            gen = gen.replace('\n', ' ')
            fp.write(f"{gen}\n")
            fp.flush()
        
        print(i)

    fp.close()

if __name__ == "__main__":
    main()
