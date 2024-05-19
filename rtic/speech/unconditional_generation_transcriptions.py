import random
import string

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


CHECKPOINT_DIR = "scratch/llm_foundry/llama_2_hf_scotus"
TEMPERATURE = 1
MAX_TOKENS = 1000
NO_GENERATIONS = 512
BATCH_SIZE = 32
OUTPUT = f"scratch/llm_foundry/llama_2_hf_scotus_unconditional_generations_temp_{TEMPERATURE}.txt"

def main():
    model = AutoModelForCausalLM.from_pretrained(CHECKPOINT_DIR, torch_dtype=torch.bfloat16)
    model.cuda()
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT_DIR)

    fp = open(OUTPUT, "w", encoding="utf-8")

    outputs = []
    for i in range(NO_GENERATIONS // BATCH_SIZE):
        inputs = [tokenizer.bos_token] * BATCH_SIZE
        start_time = "{:03d}".format(int(random.random() * 1000))
        inputs = [p + ' ' + f"H{start_time}alabama" for p in inputs]
        inputs = tokenizer(inputs, return_tensors='pt', return_token_type_ids=False, add_special_tokens=False).to("cuda")
        generation = model.generate(**inputs, max_new_tokens=MAX_TOKENS, do_sample=True, top_p=0.95, temperature=TEMPERATURE)
        generation_decoded = tokenizer.batch_decode(generation, skip_special_tokens=True)

        # Prune incomplete messages
        for gen in generation_decoded:
            gen = gen.replace('\n', ' ')
            fp.write(f"{gen}\n")

        print(i)

    fp.close()


if __name__ == "__main__":
    main()
