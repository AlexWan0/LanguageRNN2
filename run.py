import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--model', type=str, help='Model to use', choices=['random', 'max', 'feedback', 'closest'], required=True)

parser.add_argument('--cuda', help='Set device to cuda', action='store_true', default=False)

# sampling configs
parser.add_argument('--do_sampling', help='Use sampling', action='store_true', default=False)
parser.add_argument('--sampling_interval', type=int, help='Do sampling every x iterations', default=500)
parser.add_argument('--sampling_iterations', type=int, help='Number of iterations to sample', default=100)
parser.add_argument('--num_samples', type=int, help='Amount of training data to sampling over', default=100)

# property configs
parser.add_argument('--do_property', help='Get property accuracy', action='store_true', default=False)
parser.add_argument('--property_interval', help='Do property evaluation every x iterations', default=500)

# validation configs
parser.add_argument('--valid_completeness_iters', type=int, help='Number of validation samples to use when calculating completeness.', default=100)
parser.add_argument('--validation_interval', help='Do validation every x iterations', default=500)

# plot configs
parser.add_argument('--plot_fp', type=str, help='File path of plot output.', default='results.png')
parser.add_argument('--plot_interval', type=int, help='Output plot every x iterations', default=500)

parser.add_argument('--epochs', type=int, help='Number of epochs', default=1)
parser.add_argument('--limit_iter', type=int, help='Stop each epoch early', default=None)

parser.add_argument('--log_file', type=str, help='Log generations', default=None, required=False)
parser.add_argument('--log_interval', type=int, help='Output to log file every x iterations', default=100, required=False)
args = parser.parse_args()

# environment setup
import os

if args.cuda:
    os.environ["device"] = 'cuda'
else:
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
from comparative import build_comparator
from feedback import feedback_double_forward, feedback_single_forward

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

if args.do_property:
    test_compare = build_comparator(v1, v2, model_forward, 'data/')

if args.do_sampling:
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

train_loss_run = RunningAvg(20)
train_acc_run = RunningAvg(20)
plot = Plotter()

ce_loss = nn.CrossEntropyLoss()

for epoch_idx in range(args.epochs):
    print('STARTING EPOCH %d' % (epoch_idx + 1))

    random.shuffle(train_sit_id)

    if args.limit_iter is not None:
        pbar = tqdm(train_sit_id, total=min(args.limit_iter, len(train_sit_id)))
    else:
        pbar = tqdm(train_sit_id)

    num_correct = 0

    for iter_idx, sit_id in enumerate(pbar):
        if args.limit_iter is not None and iter_idx >= args.limit_iter:
            break

        utter_ids = sid2uids[sit_id]
        
        target_utters = id_select(utter_ids, utterances)
        
        input_sit_str = '^ ' + id_select([sit_id], situations)[0] + ' !'
        input_sit = v1.tokenize(input_sit_str)
        
        # generate feedback based on setting
        if args.model in feedback_double_forward:
            with torch.no_grad():
                pred_ids_feedback, _, _ = model_forward(
                    input_sit.to(device),
                    force=False,
                    sample=False,
                    tu_target=None,
                    loss_func=None,
                    gen_length=13
                )

            cleaned_pred_feedback = ' '.join(v2.decode(pred_ids_feedback)).replace('!', '').strip()

            tu = feedback_double_forward[args.model](cleaned_pred_feedback, target_utters)
        else:
            tu = feedback_single_forward[args.model](target_utters)

        assert tu[-1] == '!', 'Feedback doesn\'t end with EOS token'

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
        
        if args.log_file is not None and iter_idx % args.log_interval == 0:
            with open(args.log_file, 'a') as file_out:
                file_out.write('TRAIN %d: PRED: %s | SIT: %s | FEEDBACK: %s\n' % (iter_idx, cleaned_pred, input_sit_str, tu))

        loss.backward()
        opt_dec.step()
        opt_enc.step()
        
        opt_dec.zero_grad()
        opt_enc.zero_grad()
        
        plot.add(
            loss = train_loss_run(loss.item()/len(pred_ids)),
            accuracy_run = train_acc_run(1 if cleaned_pred in target_utters else 0)
        )

        if iter_idx % args.sampling_interval == 0 and iter_idx != 0 and args.do_sampling:        
            print('START SAMPLING:')
            sampling_ids = random.sample(train_sit_id, args.num_samples)
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

                run_sampling(input_sit.to(device), target_utters, num_samples=args.sampling_iterations)

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
        
        if iter_idx % args.property_interval == 0 and iter_idx != 0 and args.do_property:
            compare_results = test_compare()
            for prop, prop_acc in compare_results.items():
                plot_kwargs = {}
                plot_kwargs[prop] = prop_acc
                plot.add(**plot_kwargs)

        if iter_idx % args.validation_interval == 0 and iter_idx != 0:
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
                
                if args.log_file is not None and v_iter_idx % args.log_interval == 0:
                    with open(args.log_file, 'a') as file_out:
                        file_out.write('VALIDATION %d: PRED: %s| SIT: %s\n' % (v_iter_idx, cleaned_pred, input_sit_str))

                if cleaned_pred in target_utters:
                    avg_valid_accuracy += 1
                
                if v_iter_idx < args.valid_completeness_iters:
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
            avg_valid_completeness /= args.valid_completeness_iters
            
            plot.add(valid_accuracy=avg_valid_accuracy, valid_completeness=avg_valid_completeness)

        if iter_idx % args.plot_interval == 0:
            eta = ''
            if pbar.format_dict['rate'] != None:
                eta = (pbar.format_dict['total'] - iter_idx) / pbar.format_dict['rate']
            plot.output(
                fp=args.plot_fp,
                suptitle=(cleaned_pred + " | " + tu.replace('!', '').strip() + " | " + str(eta)),
                subplots=(3, 3),
                figsize=(15, 10)
            )
