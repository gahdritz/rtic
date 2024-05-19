import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from ..utils.chat_utils import (
    convert_transcriptions,
    load_transcriptions,
)


TRANSCRIPTION_DIR = "data/transcriptions/scotus_whisperx_transcripts_2/test"
MODEL_DIR = "scratch/llm_foundry/pythia-12b-hf-scotus"
CHUNK_SIZE = 256

print(MODEL_DIR)

model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, torch_dtype=torch.bfloat16)
model.cuda()
model.eval()

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)

transcriptions = load_transcriptions(TRANSCRIPTION_DIR)
converted_transcriptions = convert_transcriptions(transcriptions)

nll_cum = 0
#ce_cum = 0
#ppl_cum = 0
count = 0
for j, t in enumerate(converted_transcriptions[:50]):
    print(f"Doc len: {len(t)}")
    print(j)
    for i in range(0, len(t), CHUNK_SIZE):
        msgs = t[i: i + CHUNK_SIZE]

        if(len(msgs) < CHUNK_SIZE):
            continue

        prefixes = [m[:pl] for (_, _, _, pl, m) in msgs]
        content = [m[pl:] for (_, _, _, pl, m) in msgs]

        prefix_tok = [tokenizer.encode(p, return_tensors='pt', add_special_tokens=False) for p in prefixes]
        content_tok = [tokenizer.encode(c, return_tensors='pt', add_special_tokens=False) for c in content]

        tokenized = torch.cat([v for t in zip(prefix_tok, content_tok) for v in t], dim=-1)

        dummy = tokenizer.encode("dummy", return_tensors='pt', add_special_tokens=True)
        dummy_no_bos = tokenizer.encode("dummy", return_tensors='pt', add_special_tokens=False)
        if(dummy_no_bos[:, 0] != dummy[:, 0]):
            tokenized = torch.cat([dummy[:, :1], tokenized], dim=-1)

        tokenized = tokenized.cuda()

        with torch.no_grad():
            logits = model(tokenized).logits

        preds = logits.squeeze(0)[:-1]
        labels = tokenized.squeeze(0)[1:]

        logprobs = torch.log_softmax(preds, dim=-1)
        
        selected_logprobs = torch.take_along_dim(logprobs, labels.unsqueeze(-1), dim=-1)
        selected_logprobs = selected_logprobs.squeeze(-1)

        nll_cum += -1 * torch.sum(selected_logprobs).item()
#        loss = torch.nn.functional.cross_entropy(preds, labels)
#        ppl = torch.exp(loss)
#
#        ce_cum += loss
#        ppl_cum += ppl

        count += 1
        print(i)

print(f"NLL: {nll_cum / count}")
print(f"CE: {ce_cum / count}")
print(f"PPL: {ppl_cum / count}")
