# environment setup
import os
os.environ["device"] = 'cpu'

device = os.environ.get('device', 'cpu')

import sys
sys.path.append('src/')

# random seed setup
SEED = 0

import torch
torch.manual_seed(SEED)

import random
random.seed(SEED)

import numpy as np
np.random.seed(SEED)

from data import build_data, id_select
from vocab import build_vocab
from net import build_model
from train import wrap_forward
from experiment import Plotter, RunningAvg
from completeness import beam_completeness
from sampling import build_sampling
from torch import nn
from tqdm.auto import tqdm

sit2id, sid2uids, train_sit_id, valid_sit_id, situations, utterances = build_data(
    id_pairs='data/id_pairs_2.txt',
    situations='data/situations_2.txt',
    utterances='data/utterances_2.txt',
    training_set='data/training_set.txt',
    test_set='data/test_set.txt'
)

v1, v2, eos_id, start_id = build_vocab(situations, utterances)

(enc, opt_enc), (dec, opt_dec) = build_model(v1, v2, enc_lr=5e-4, dec_lr=5e-4)

enc = enc.to(device)
dec = dec.to(device)

model_forward = wrap_forward(enc, dec, start_id, eos_id)

train_loss_run = RunningAvg(10)
plot = Plotter()

run_sampling = True

run_sampling = build_sampling(
    enc,
    dec,
    v2,
    start_id,
    eos_id,
    lr=1e-5,
    temperature=1.0,
    gen_length=13
)

ce_loss = nn.CrossEntropyLoss()

random.shuffle(train_sit_id)
pbar = tqdm(train_sit_id)

num_correct = 0

for iter_idx, sit_id in enumerate(pbar):
    utter_ids = sid2uids[sit_id]
    
    target_utters = id_select(utter_ids, utterances)
    
    input_sit_str = '^ ' + id_select([sit_id], situations)[0] + ' !'
    input_sit = v1.tokenize(input_sit_str)
    
    tu = random.choice(target_utters) + ' !'
    tu_target = v2.tokenize(tu)
    
    pred_ids, loss, logits_all = model_forward(
        input_sit.to(device),
        force=True,
        sample=False,
        tu_target=tu_target.to(device),
        loss_func=ce_loss,
        gen_length=None
    )
    
    cleaned_pred = ' '.join(v2.decode(pred_ids)).replace('!', '').strip()
    
    if cleaned_pred in target_utters:
        num_correct += 1
    
    loss.backward()
    opt_dec.step()
    opt_enc.step()
    
    opt_dec.zero_grad()
    opt_enc.zero_grad()
    
    plot.add(loss = train_loss_run(loss.item()/len(pred_ids)), accuracy = num_correct/(iter_idx + 1))
    
    if iter_idx % 500 == 0:
        eta = ''
        if pbar.format_dict['rate'] != None:
            eta = (pbar.format_dict['total'] - iter_idx) / pbar.format_dict['rate']
        plot.output(
            suptitle=(cleaned_pred + " | " + tu.replace('!', '').strip() + " | " + str(eta)),
            subplots=(3, 3),
            figsize=(15, 10)
        )
    
    if iter_idx % 500 == 0 and iter_idx != 0 and run_sampling:        
        print('START SAMPLING:')
        sampling_ids = random.sample(train_sit_id, 100)
        for sampling_iter_idx, sit_id in enumerate(tqdm(sampling_ids)):
            utter_ids = sid2uids[sit_id]
    
            target_utters = id_select(utter_ids, utterances)

            input_sit_str = '^ ' + id_select([sit_id], situations)[0] + ' !'
            input_sit = v1.tokenize(input_sit_str)
            
            '''comp_1 = beam_completeness(
                target_utters,
                input_sit,
                enc,
                dec,
                v2,
                start_id,
                eos_id,
                verbose=False
            )'''
            #print('%d: PRE COMPLETENESS SINGLE SAMPLE: %.2f' % (sampling_iter_idx, comp_1))

            run_sampling(input_sit.to(device), target_utters, num_samples=100)

            '''comp_2 = beam_completeness(
                target_utters,
                input_sit,
                enc,
                dec,
                v2,
                start_id,
                eos_id,
                verbose=False
            )'''
            #print('%d: POST COMPLETENESS SINGLE SAMPLE: %.2f' % (sampling_iter_idx, comp_2))
    
    if iter_idx % 500 == 0 and iter_idx != 0:
        avg_valid_accuracy = 0.0
        avg_valid_completeness = 0.0
        
        print('START VALIDATION:')
        
        for v_iter_idx, sit_id in enumerate(tqdm(valid_sit_id)):
            utter_ids = sid2uids[sit_id]
            
            target_utters = id_select(utter_ids, utterances)
            
            input_sit_str = '^ ' + id_select([sit_id], situations)[0] + ' !'
            input_sit = v1.tokenize(input_sit_str)
            
            with torch.no_grad():
                pred_ids, _, logits_all = model_forward(
                    input_sit.to(device),
                    force=False,
                    sample=False,
                    tu_target=None,
                    loss_func=None,
                    gen_length=13
                )
            
            cleaned_pred = ' '.join(v2.decode(pred_ids)).replace('!', '').strip()
            
            if cleaned_pred in target_utters:
                avg_valid_accuracy += 1
            
            if v_iter_idx < 100:
                comp = beam_completeness(
                    target_utters,
                    input_sit.to(device),
                    enc,
                    dec,
                    v2,
                    start_id,
                    eos_id,
                    verbose=False
                )

                avg_valid_completeness += comp
        
        avg_valid_accuracy /= len(valid_sit_id)
        avg_valid_completeness /= 100
        
        plot.add(valid_accuracy=avg_valid_accuracy, valid_completeness=avg_valid_completeness)