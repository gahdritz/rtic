import os
import random

import torch
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer,
    StoppingCriteria,
)

from ..utils.chat_utils import (
    convert_chat,
    load_chat,
)


def tokenize(converted, tokenizer):
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

    return tokenized


CHAT_DUMP = "data/chat_data/dump2.json"
MODEL_DIR = "scratch/llm_foundry/llama_2_new_hf"
CHUNK_SIZE = 64

# Llama 2
DELIMITER = "---------+"

# Pythia
#DELIMITER = "][^"

# Gemma
#DELIMITER = "________,"

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


chat = load_chat(CHAT_DUMP)
chat = chat[int(len(chat) * 0.95):]
chat = [c for c in chat if c[-1] is not None]

model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, torch_dtype=torch.bfloat16)
model.cuda()
model.eval()

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)

eom_tok = tokenizer.encode(f"a{DELIMITER}", add_special_tokens=False)[-1]

delimiters = ['+', ':', ';', '.']
delimiter_toks = [tokenizer.encode(f"bla{d}", add_special_tokens=False)[-1] for d in delimiters]
delimiter_toks.extend([tokenizer.encode(f"{d}", add_special_tokens=False)[-1] for d in delimiters])
delimiter_offsets = {
    '+': 7, 
    ':': 5,
    ';': 3,
    '.': 1,
}

counts = []
fracs = []
print(len(chat))
for j in range(0, len(chat), CHUNK_SIZE):
    chunk = chat[j: j + CHUNK_SIZE]
    if(len(chunk) != CHUNK_SIZE):
        break

    # Search for messages by user B followed by a message by user A
    indices = []
    for i, (msg_1, msg_2) in enumerate(zip(chunk[:-1], chunk[1:])):
        if(msg_1[0] == 'B' and msg_2[0] == 'A'):
            indices.append(i)

    random.shuffle(indices)
    for i in indices[-2:]:
        prompt = chunk[: i + 1]
        converted = convert_chat(prompt, delimiter=DELIMITER)

        toks = tokenize(converted, tokenizer)
        len_toks = toks.shape[-1]
  
        generation = model.generate(
            toks,
            attention_mask=toks.new_ones(toks.shape),
            max_new_tokens=100, 
            do_sample=True, 
            top_p=0.95, 
            temperature=1,
            stopping_criteria=EOMCriteria(eom_tok),
            output_logits=True,
            return_dict_in_generate=True,
        )

        sequence = generation["sequences"]
        new_tokens = sequence[:, len_toks:][0]

        # Find the first delimiter
        del_idx = 0
        delimiter = None
        if(not new_tokens[0] in delimiter_toks):
            continue
        
        delimiter_id = new_tokens[0]
        delimiter = tokenizer.decode([delimiter_id])
       
        uid_token = new_tokens[1 + delimiter_offsets[delimiter]]
        uid_token_decoded = tokenizer.decode([uid_token])
        if(uid_token_decoded == "A"):
            print("as user")
            break
        elif(uid_token_decoded == "B"):
            print("success!")
        else:
            print("bad user")
            break

        draft_logits = torch.cat(generation["logits"], dim=0) 
        real_response = chunk[i + 1]

        converted_with_real = convert_chat(prompt + [real_response], delimiter=DELIMITER)
        toks_converted_with_real = tokenize(converted_with_real, tokenizer)

        new_toks_interruption = toks_converted_with_real[0, len_toks:]
        del_idx_interruption = 0

        if(new_toks_interruption[del_idx_interruption] not in delimiter_toks):
            print("bad")
            break

        # Fix the support of the old logits
        idx = len(new_toks_interruption) - 1
        while idx >= 0 and new_toks_interruption[idx] != delimiter_id:
            idx -= 1

        if(idx < 0):
            pass
        else:
            no_sigfigs = min(delimiter_offsets[delimiter], 2)
            draft_time = tokenizer.decode(new_tokens[1: 1 + no_sigfigs])
            assert(len(draft_time) == no_sigfigs)
            interruption_time = tokenizer.decode(new_toks_interruption[idx + 1: idx + no_sigfigs + 1])
            assert(len(interruption_time) == no_sigfigs)

            draft_time_int = int(draft_time)
            interruption_time_int = int(interruption_time)

            diff = draft_time_int - interruption_time_int
            if(diff <= 0):
                # It would have been output already
                print("Skip! (draft in past)")
                continue
            else:
                if(0 < diff < 10):
                    adjust_idx = no_sigfigs
                else:
                    adjust_idx = 1
    
                interruption_time_digit_at_pos = (delimiter + interruption_time)[adjust_idx] 
                digits_to_remove = range(int(interruption_time_digit_at_pos))
                tokens_to_remove = [tokenizer.encode(f"{delimiter}{d}", add_special_tokens=False)[-1] for d in digits_to_remove]
                for t in tokens_to_remove:
                    draft_logits[adjust_idx][t] = 0
    
                draft_logits[adjust_idx] /= torch.sum(draft_logits[adjust_idx])


#        while(new_tokens[del_idx] == new_toks_interruption[del_idx_interruption]):
#            del_idx += 1
#            del_idx_interruption += 1

        start_idx = del_idx 

        toks_converted_with_real_len = toks_converted_with_real.shape[-1]
        toks_with_draft = torch.cat(
            [
                toks_converted_with_real, 
                new_tokens.unsqueeze(0)
            ], dim=-1
        )

        with torch.no_grad():
            logits_with_draft = model(toks_with_draft).logits

        relevant_logits = logits_with_draft[0, :-1]
        rlstart = toks_converted_with_real_len + start_idx - 1

        dl = draft_logits[start_idx:]
        rl = relevant_logits[rlstart:]

        assert(dl.shape == rl.shape)

        count = 0
        for l1, l2, tok_id in zip(dl, rl, new_tokens[start_idx:]):
            q = l1[tok_id]
            p = l2[tok_id]

            if(q == 0):
                # This token was removed earlier
                break

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
        fracs.append(count / len(new_tokens[start_idx:]))

        if(len(counts) == 300):
            print(f"Average: {sum(counts) / len(counts)}")
            print(f"Fractions: {sum(fracs) / len(fracs)}")
            exit()

print(len(counts))
print(f"Average: {sum(counts) / len(counts)}")
print(f"Fractions: {sum(fracs) / len(fracs)}")
