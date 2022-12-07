import torch.nn as nn
from torch import optim
import torch
import torch.nn.functional as F

import os
device = os.environ.get('device', 'cpu')

class RNNEnc(nn.Module):
    def __init__(self, vocab, input_size=300, hidden_size=300):
        super().__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.rnn_enc = nn.RNN(input_size=self.input_size, hidden_size=self.hidden_size, num_layers=1)
        
        #self.lin_input = nn.Linear(30, 128)
        self.embed = nn.Embedding(len(vocab), self.input_size)
        
        self.embed.to(device)

        self.relu = nn.ReLU()
    
        self.enc_lin = nn.Linear(self.hidden_size, self.hidden_size)
    
    def forward(self, x): # seq_length
        x = self.embed(x).unsqueeze(1) # seq_length x 1 x 128
        
        h0 = torch.zeros((1, 1, self.hidden_size), device=device)
        
        enc_out, enc_hidden = self.rnn_enc(x, h0)
        # enc_out: seq_length x 1 x 128
        # enc_hidden: 1 x 1 x 128
        
        #enc_out = self.enc_lin(enc_out.squeeze(1))
        #enc_out = self.relu(enc_out) # seq_length x 300
        
        enc_out = enc_out.squeeze(1)
        
        return enc_out, enc_hidden

class RNNDec(nn.Module):
    def __init__(self, vocab, embed, input_size=(300), hidden_size=300, max_length=24):
        super().__init__()
        
        self.max_length = max_length
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.rnn_dec = nn.RNN(input_size=self.input_size, hidden_size=self.hidden_size, num_layers=1)
        
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        
        self.lin_output = nn.Linear(self.hidden_size, len(vocab))
        #self.lin_hidden = nn.Linear(256, 128)
        self.embed = nn.Embedding(len(vocab), 300)
        self.embed.to(device)

        self.softmax = nn.Softmax(dim=1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.1)
        
    def forward(self, x, enc_hidden, h, enc_out):
        # x: 1
        # h: 1 x 1 x 300
        x = self.embed(x).reshape(1, -1) # 1 x 300
        x = self.dropout(x)
        
        #enc_hidden = self.lin_hidden(enc_hidden)
        #enc_hidden = self.relu(enc_hidden)
        
        #print(torch.cat((x[0], enc_hidden[0]), dim=1).shape)
        
        attn_weights = F.softmax(self.attn(torch.cat((x, enc_hidden[0]), dim=1)), dim=1) # 1 x max_length
        
        #seq_length = enc_out.shape[0]
        #attn_weights = attn_weights[:, :seq_length] # 1 x seq_length
        
        attn_applied = torch.mm(attn_weights, enc_out) # (1 x seq_length) x (seq_length x 300) = (1 x 300)
        
        rnn_in = torch.cat((x, attn_applied), dim=1)
        rnn_in = self.attn_combine(rnn_in).unsqueeze(0)
        rnn_in = F.relu(rnn_in)
        
        dec_out, dec_hidden = self.rnn_dec(rnn_in, h)
        # dec_out: 1 x 1 x 300
        # dec_hidden: 1 x 1 x 300
        
        logits = self.lin_output(dec_out[0]) # 1 x 30
        
        #probs = self.softmax(logits)
        
        return logits, dec_hidden

def model_forward(enc,
                  dec,
                  input_sit,
                  start_id,
                  eos_id,
                  gen_length=13,
                  force=False,
                  sample=False,
                  tu_target=None,
                  loss_func=None):

    enc_out, enc_hidden = enc(input_sit)

    # if generation length is None, the generation length is set to the gt length
    assert gen_length is not None or tu_target is not None, 'gen_length = None requires a ground truth to get generation length.'

    if gen_length is None:
        gen_length = len(tu_target)

    # can't do teacher forcing and sampling at the same time
    assert not (force and sample), 'Can\'t do teacher forcing and sampling at the same time.'

    # loss function requires a gt
    assert not (loss_func is not None and tu_target is None), 'Loss function requires a ground truth.'

    pred_ids = []
    logits_all = []
    loss = 0.0

    prev = start_id
    h = enc_hidden
    for i in range(gen_length):
        logits, h = dec(prev.to(device), enc_hidden, h, enc_out)

        if force:
            pred = torch.argmax(logits, dim=1)[0]
            prev = tu_target[i]
        else:
            if not sample:
                pred = torch.argmax(logits, dim=1)[0]
            else:
                probs = Categorical(F.softmax(logits[0], dim=0))
                pred = probs.sample()

            prev = pred

        pred_ids.append(pred)
        logits_all.append(logits)

        if tu_target != None and i < len(tu_target):
            loss += loss_func(logits, tu_target[i].unsqueeze(0))

        if pred == eos_id:
            break

    return pred_ids, loss, logits_all


def build_forcing_schedule(force_p):
    def force_p_schedule(epoch_idx):
        if epoch_idx >= len(force_p):
            return force_p[-1]
        return force_p[epoch_idx]

    return force_p_schedule

def build_model(v1, v2, enc_lr=5e-4, dec_lr=5e-4):
    enc = RNNEnc(v1)
    dec = RNNDec(v2, enc.embed)

    opt_enc = optim.Adam(enc.parameters(), lr=enc_lr)
    opt_dec = optim.Adam(dec.parameters(), lr=dec_lr)

    return (enc, opt_enc), (dec, opt_dec)
