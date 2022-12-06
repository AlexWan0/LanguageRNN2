import torch
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
from torch import optim
from tqdm.auto import tqdm

import os
device = os.environ.get('device', 'cpu')

def build_sampling(enc, dec, v2, start_id, eos_id, lr=1e-5, temperature=1.0, gen_length=13):
    opt_enc_sampling = optim.Adam(enc.parameters(), lr=lr)
    opt_dec_sampling = optim.Adam(dec.parameters(), lr=lr)

    def wrapped_sampling(input_sit, target_utters, num_samples=500):
        sampling(input_sit,
                 target_utters,
                 enc, 
                 dec,
                 opt_enc_sampling,
                 opt_dec_sampling,
                 v2,
                 start_id,
                 eos_id,
                 temperature,
                 num_samples,
                 gen_length)

    return wrapped_sampling

def sampling(input_sit, target_utters, enc, dec, opt_enc, opt_dec, v2, start_id, eos_id, temperature, num_samples, gen_length):
    distinct_correct = set()
    distinct_incorrect = set()

    for sample_iter in range(num_samples):
        # begin sampling
        enc_out, enc_hidden = enc(input_sit)

        logprob = 0.0
        pred_ids = []
        logits_all = []
        prev = start_id.to(device)
        h = enc_hidden
        for i in range(gen_length):
            logits, h = dec(prev, enc_hidden, h, enc_out)

            probs = Categorical(F.softmax(logits[0] / temperature, dim=0))

            prev = probs.sample()
            pred_ids.append(prev)

            logits_all.append(logits)

            logprob += -F.cross_entropy(logits, prev.unsqueeze(0))

            if prev == eos_id:
                break

        cleaned_pred = ' '.join(v2.decode(pred_ids)).replace('!', '').strip()
        #print('SAMPLE:', cleaned_pred)

        if cleaned_pred in target_utters:
            distinct_correct.add(cleaned_pred)
            #print('SAMPLE CORRECT #%d' % len(distinct_correct))
            loss = -logprob
        else:
            distinct_incorrect.add(cleaned_pred)
            #print('SAMPLE INCORRECT')
            loss = logprob * 1e-2

        loss.backward()

        opt_enc.step()
        opt_dec.step()

        opt_enc.zero_grad()
        opt_dec.zero_grad()
