import os
import random
import string

import torch
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer,
    StoppingCriteria,
)

from ..utils.chat_utils import (
    convert_transcriptions,
    load_transcriptions,
)


def tokenize(converted, tokenizer):
    prefixes = [m[:pl] for (_, _, _, pl, m) in converted]
    content = [m[pl:] for (_, _, _, pl, m) in converted]

    prefix_tok = [tokenizer.encode(p, return_tensors='pt', add_special_tokens=False) for p in prefixes]
    content_tok = [tokenizer.encode(c, return_tensors='pt', add_special_tokens=False) for c in content]

    tokenized = torch.cat([v for t in zip(prefix_tok, content_tok) for v in t], dim=-1)

    dummy = tokenizer.encode("dummy", return_tensors='pt', add_special_tokens=True)
    dummy_no_bos = tokenizer.encode("dummy", return_tensors='pt', add_special_tokens=False)
    if(dummy_no_bos[:, 0] != dummy[:, 0]):
        tokenized = torch.cat([dummy[:, :1], tokenized], dim=-1)

    tokenized = tokenized.cuda()

    return tokenized


TRANSCRIPTION_DIR = "data/transcriptions/scotus_whisperx_transcripts_2/test"
MODEL_DIR = "scratch/llm_foundry/llama_2_hf_scotus"
CHUNK_SIZE = 64
LLAMA_INTERNAL_SPACE = 259


class EOMCriteria(StoppingCriteria):
    def __init__(self, EOM_TOK):
        self.EOM_TOK = EOM_TOK

    def __call__(self, input_ids, scores, **kwargs):
        if input_ids[:,-1].item() == self.EOM_TOK:
            return True
        return False  # Continue generation

    def __len__(self):
        return 1

    def __iter__(self):
        yield self


transcriptions = load_transcriptions(TRANSCRIPTION_DIR)
converted_transcriptions = convert_transcriptions(transcriptions)

model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, torch_dtype=torch.bfloat16)
model.cuda()
model.eval()

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)

counts = []
fracs = []
for transcription in transcriptions:
    for j in range(0, len(transcription), CHUNK_SIZE):
        chunk = transcription[j: j + CHUNK_SIZE]
 
        # Search for messages by one user interrupted by another 
        indices = []
        for i, (msg_1, msg_2) in enumerate(zip(chunk[:-1], chunk[1:])):
            if(msg_1[0] != msg_2[0]):
                indices.append(i)
    
        random.shuffle(indices)
        for i in indices[-1:]:
            prompt = chunk[: i + 1]
            real_response = chunk[i + 1]
            model_user = string.ascii_uppercase[int(prompt[-1][0])]
            new_user = string.ascii_uppercase[int(real_response[0])]
            converted = convert_transcriptions([prompt])[0]
    
            toks = tokenize(converted, tokenizer)

            toks = torch.cat([
                toks,
                torch.tensor([[LLAMA_INTERNAL_SPACE]]).cuda(),
            ], dim=-1)

            len_toks = toks.shape[-1]
      
            generation = model.generate(
                toks,
                attention_mask=toks.new_ones(toks.shape),
                max_new_tokens=100, 
                do_sample=True, 
                top_p=0.95, 
                temperature=1,
                stopping_criteria=EOMCriteria(LLAMA_INTERNAL_SPACE),
                output_logits=True,
                return_dict_in_generate=True,
            )

            sequence = generation["sequences"]
            new_tokens = sequence[:, len_toks:][0]

            if(new_tokens[0] == tokenizer.encode(new_user, add_special_tokens=False)[-1]):
                print("as user")
                break
            else:
                print("success!")

            sequence = generation["sequences"]
            new_tokens = sequence[:, len_toks:][0]
    
            draft_logits = torch.cat(generation["logits"], dim=0)
     
            converted_with_real = convert_transcriptions([prompt + [real_response]])[0]
            toks_converted_with_real = tokenize(converted_with_real, tokenizer)
    
            toks_converted_with_real_len = toks_converted_with_real.shape[-1]
            toks_with_draft = torch.cat(
                [
                    toks_converted_with_real,
                    torch.tensor([[LLAMA_INTERNAL_SPACE]]).cuda(),
                    tokenizer.encode(model_user, return_tensors="pt", add_special_tokens=False)[:, -1:].cuda(),
                    new_tokens.unsqueeze(0)
                ],
                dim=-1,
            )
    
            with torch.no_grad():
                logits_with_draft = model(toks_with_draft).logits
    
            relevant_logits = logits_with_draft[0, :-1]
            rlstart = toks_converted_with_real_len + 1
    
            dl = draft_logits
            rl = relevant_logits[rlstart:]
    
            assert(dl.shape == rl.shape)
    
            count = 0
            for l1, l2, tok_id in zip(dl, rl, new_tokens):
                q = l1[tok_id]
                p = l2[tok_id]
    
                if(q < p):
                    count += 1
                    continue
                else:
                    u = random.random()
                    if u > 1 - p/q:
                        count += 1
                        continue
                    else:
                        break

            counts.append(count)
            fracs.append(count / len(new_tokens))

            if(len(counts) == 100):
                print(f"Average: {sum(counts) / len(counts)}")
                print(f"Fractions: {sum(fracs) / len(fracs)}")
                exit()
