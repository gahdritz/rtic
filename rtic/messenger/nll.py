import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from ..utils.chat_utils import (
    convert_chat,
    load_chat,
)


CHAT_DUMP = "data/chat_data/dump2.json"
MODEL_DIR = "scratch/llm_foundry/llama_2_hf"
CHUNK_SIZE = 64

# Llama 2
DELIMITER = "---------+"

# Pythia
#DELIMITER = "][^"

# Gemma
#DELIMITER = "________,"

print(MODEL_DIR)

chat = load_chat(CHAT_DUMP)
chat = chat[int(len(chat) * 0.95):]
chat = [c for c in chat if c[-1] is not None]

model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, torch_dtype=torch.bfloat16)
model.cuda()
model.eval()

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)

nll_cum = 0
count = 0
for i in range(0, len(chat), CHUNK_SIZE):
    chunk = chat[i: i + CHUNK_SIZE]
    if(len(chunk) != CHUNK_SIZE):
        break

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

    print(tokenized.shape)

    with torch.no_grad():
        logits = model(tokenized).logits

    preds = logits.squeeze(0)[:-1]
    labels = tokenized.squeeze(0)[1:]

    logprobs = torch.log_softmax(preds, dim=-1)
    
    selected_logprobs = torch.take_along_dim(logprobs, labels.unsqueeze(-1), dim=-1)
    selected_logprobs = selected_logprobs.squeeze(-1)

    nll_cum += -1 * torch.sum(selected_logprobs).item()

    count += 1


print(f"NLL: {nll_cum / count}")
