import torch
import torch.nn.functional as F

import os
device = os.environ.get('device', 'cpu')

def check_completeness(model_sentences, utterances):
    """
    This function check if the sentence is valid for the current situation
    :param situation:
    :param model_sentences: List of learner utterances for the current situation
    :param situation_utterance_map:
    :return:
    """

    # Check if the model sentence is within the list of utterances
    correct = 0
    for sentence in utterances:
        if sentence in model_sentences:
            correct += 1

    correct = correct / len(utterances) if len(utterances) > 0 else 0.0

    return correct

def beam_search(input_sit, encoder, decoder, start_id, eos_id, max_length=13, K=10, device='cpu'):
    with torch.no_grad():
        enc_out, enc_hidden = encoder(input_sit)

        pq = [{'h': enc_hidden, 'prev': start_id, 'seq': [], 'prob': 1.0, 'ended': False}]

        for di in range(max_length):
            for beam_idx in range(K):
                if len(pq) == 0:
                    break

                popped = pq.pop(0)
                h = popped['h']
                prev = popped['prev']
                seq = popped['seq']
                prob = popped['prob']
                ended = popped['ended']

                if ended:
                    pq.append(popped)
                    continue

                logits, h = decoder(prev, enc_hidden, h, enc_out)

                _, sorted_indices = torch.sort(logits, dim=1, descending=True)

                for new_pred in sorted_indices[0]:
                    #new_prob = (prob * len(seq) + F.log_softmax(logits, dim=1)[0, new_pred]) / (len(seq) + 1)
                    new_prob = prob + F.log_softmax(logits, dim=1)[0, new_pred]

                    pq.append({'h': h,
                        'prev': new_pred,
                        'seq': seq + [new_pred],
                        'prob': new_prob,
                        'ended': new_pred == eos_id})

            pq = sorted(pq, key=lambda x: -x['prob'])[:K]

        return [(pred_data['seq'], pred_data['prob']) for pred_data in pq]

def beam_completeness(utterances, input_sit, encoder, decoder, vocab, start_id, eos_id, max_length=13, K=40, verbose=False):
    out = beam_search(input_sit,
                      encoder,
                      decoder,
                      start_id.to(device),
                      eos_id,
                      max_length=max_length,
                      K=K)
    
    valid_preds = []
    for seq, prob in out:
        cleaned_pred = ' '.join(vocab.decode(seq)).replace('!', '').strip()
        valid_preds.append(cleaned_pred)
        
        if verbose:
            print(prob.item(), cleaned_pred)

    return check_completeness(valid_preds, utterances)
