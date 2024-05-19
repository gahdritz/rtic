import random

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from ..utils.chat_utils import (
    convert_transcriptions,
    load_transcriptions,
)


#CHECKPOINT_DIR = "scratch/llm_foundry/llama_2_hf_scotus"
CHECKPOINT_DIR = "scratch/llm_foundry/pythia-12b-hf-scotus"
#CHECKPOINT_DIR = "scratch/llm_foundry/gemma-2b_hf_scotus"
CHUNK_FILE = "out/short_chunks_scotus.txt"
TEMPERATURE = 1
MAX_TOKENS = 500

TRANSCRIPTION_DIR = "data/transcriptions/scotus_whisperx_transcripts_2/test"

OUTPUT = f"scratch/llm_foundry/pythia-12b_hf_scotus_conditional_generations_temp_{TEMPERATURE}.txt"


def main():
    model = AutoModelForCausalLM.from_pretrained(CHECKPOINT_DIR, torch_dtype=torch.bfloat16)
    model.cuda()
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT_DIR)

    with open(CHUNK_FILE, "r") as fp:
        chunk_indices = [c.strip() for c in fp.readlines()]

    chunk_indices = [tuple(map(int, c.split(','))) for c in chunk_indices]
    
    chats = load_transcriptions(TRANSCRIPTION_DIR)
    converted = convert_transcriptions(chats)

    fp = open(OUTPUT, "w", encoding="utf-8")
   
    skip = set([2, 3, 4, 11, 15, 18, 20, 25, 26])
    converted = [c for i, c in enumerate(converted) if i not in skip]

    for i, (s, e) in enumerate(chunk_indices):
        chunk = converted[i][s:e]

        prefixes = [m[:pl] for (_, _, _, pl, m) in chunk]
        content = [m[pl:] for (_, _, _, pl, m) in chunk]

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

        for gen in generation_decoded:
            gen = gen.replace('\n', ' ')
            fp.write(f"{gen}\n")
            fp.flush()

        if(i == 21):
            break
        
        print(i)

    fp.close()

if __name__ == "__main__":
    main()
