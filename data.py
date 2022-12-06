def load(fn):
    with open(fn, 'r') as file_in:
        return file_in.read().split('\n')

def repl_t(text):
    text = text.replace('(t1)', '(x1)')
    text = text.replace('(t2)', '(x2)')
    text = text.replace('(t1,t2)', '(x1,x2)')
    return text

def id_select(ids, targets):
    result = []
    for target_id in ids:
        result.append(targets[target_id - 1][1])
    return result

def build_data(id_pairs='id_pairs_2.txt', situations='situations_2.txt', utterances='utterances_2.txt', training_set='training_set.txt', test_set='test_set.txt'):
    id_pairs = [[int(x) for x in l.split('\t')] for l in load(id_pairs) if l != '']
    sid2uids = {}

    for s, u in id_pairs:
        if s not in sid2uids:
            sid2uids[s] = []
        sid2uids[s].append(u)

    situations = [[int(l.split('\t')[0]), l.split('\t')[1]] for l in load(situations) if l != '']

    utterances = [[int(l.split('\t')[0]), l.split('\t')[1]] for l in load(utterances) if l != '']

    sit2id = {s[1]: s[0] for s in situations}

    train_sit = [l.split('\t')[0] for l in load(training_set)]
    valid_sit = [l.split('\t')[0] for l in load(test_set)]

    train_sit_id = [sit2id[repl_t(s)] for s in train_sit if s != '']
    valid_sit_id = [sit2id[repl_t(s)] for s in valid_sit if s != '']

    return sit2id, sid2uids, train_sit_id, valid_sit_id, situations, utterances
