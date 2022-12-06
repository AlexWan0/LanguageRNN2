import os
device = os.environ.get('device', 'cpu')

from torch import nn
import torch
from data import repl_t
from tqdm.auto import tqdm

# https://stackoverflow.com/a/26726185
def argmax(iterable):
    return max(enumerate(iterable), key=lambda x: x[1])[0]

def test_sentence(input_sit, tu_target_compare, model_forward):
    compare_neg_loss = []

    ce_loss = nn.CrossEntropyLoss()
    for tu_target in tu_target_compare:
        with torch.no_grad():
            _, loss, _ = model_forward(
                input_sit.to(device),
                force=True,
                sample=False,
                tu_target=tu_target,
                loss_func=ce_loss,
                gen_length=None
            )

        compare_neg_loss.append(-loss.cpu().item())

    return argmax(compare_neg_loss)

def import_compare_data(prop, split, directory):
    path = os.path.join(directory, '%s_test_set_%d.txt' % (prop, split))

    situations = []
    utterance_pairs = []

    with open(path, 'r') as file_in:
        for line in file_in:
            if len(line) > 0:
                l_spl = line.strip().split('\t')

                situations.append(repl_t(l_spl[0]))
                utterance_pairs.append(l_spl[1:])

    return situations, utterance_pairs

def build_comparator(v1, v2, model_forward, directory, props=['shape', 'relation', 'size', 'color'], split=1):
    def wrapped_compare():
        results = {}
        for p in props:
            results[p] = []

            situations, utterance_pairs = import_compare_data(p, split, directory)

            print('STARTING PROPERTY TEST (%s)' % p)

            for input_sit_str, utt_pair in tqdm(zip(situations, utterance_pairs), total=len(situations)):
                input_sit_str = '^ ' + input_sit_str + ' !'
                input_sit = v1.tokenize(input_sit_str)

                utter_pair_tokenized = (v2.tokenize(u + ' !') for u in utt_pair)

                compare_pred = test_sentence(input_sit, utter_pair_tokenized, model_forward)

                results[p].append(1 if compare_pred == 0 else 0)

            results[p] = sum(results[p]) / len(results[p])

        return results

    return wrapped_compare
