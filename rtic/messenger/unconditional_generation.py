import random
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


CHECKPOINT_DIR = "scratch/llm_foundry/llama_2_hf"
TEMPERATURE = 1
MAX_TOKENS = 1000
DELIMITER = "---------+" 
NO_GENERATIONS = 512
BATCH_SIZE = 1
#OUTPUT = f"scratch/llm_foundry/llama_2_hf_unconditional_generations_temp_{TEMPERATURE}.txt"
OUTPUT = "test"

SPEAKERS = ["A", "B"]
YEARS = [str(y) for y in range(2016, 2023)]

def main():
    model = AutoModelForCausalLM.from_pretrained(CHECKPOINT_DIR, torch_dtype=torch.bfloat16)
    model.cuda()
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT_DIR)

    outputs = []
    for i in range(NO_GENERATIONS // BATCH_SIZE):
        inputs = [tokenizer.bos_token] * BATCH_SIZE
        inputs = [p + random.sample(SPEAKERS, 1)[0] + random.sample(YEARS, 1)[0] for p in inputs]
        inputs = tokenizer(inputs, return_tensors='pt', return_token_type_ids=False, add_special_tokens=False).to("cuda")

        tok_len_before = inputs["input_ids"].shape[-1]
        t = time.time()
        generation = model.generate(**inputs, max_new_tokens=MAX_TOKENS, do_sample=True, top_p=0.95, temperature=TEMPERATURE)
        time_elapsed = time.time() - t
        tok_len_after = generation[0].shape[-1]

        print((tok_len_after - tok_len_before) / time_elapsed)
        exit()

        generation_decoded = tokenizer.batch_decode(generation, skip_special_tokens=True)

        # Prune incomplete messages
        generation_decoded = [DELIMITER.join(g.split(DELIMITER)[:-1]).strip(DELIMITER) for g in generation_decoded]
        outputs.extend(generation_decoded)
        print(i)

    with open(OUTPUT, "w", encoding="utf-8") as fp:
        for output in outputs:
            fp.write(f"{output}\n")


if __name__ == "__main__":
    main()
