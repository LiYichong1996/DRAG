import os
from itertools import product

import torch


def get_dict(dir):
    f = open(dir)
    iter_f = iter(f)

    emb_dict = {}
    for line in iter_f:
        line_list = line.split(' ')
        tok = line_list[0]
        tok = tok.replace("T", "U")
        emb = line_list[1:]
        emb = [float(emb[i]) for i in range(len(emb))]
        emb = torch.tensor(emb, dtype=torch.float)
        emb_dict[tok] = emb

    return emb_dict


def emb_place(seq, i, emb_dict, mer=3):
    l = len(seq)
    start = i
    end = i
    if mer % 2 == 0:
        start = i - (mer // 2)
        end = i + (mer // 2) - 1
    else:
        start = i - (mer // 2)
        end = i + (mer // 2)
    if start < 0:
        start = 0
    if end >= l:
        end = l - 1

    tok = seq[start:end+1]
    emb = emb_dict[tok]

    return emb


def emb_full(seq, emb_dict, mer=3, max_size=None):
    emb_seq = []
    l = len(seq)
    for i in range(l):
        emb = emb_place(seq, i, emb_dict, mer)
        emb_seq.append(emb)

    if max_size is not None:
        emb_padding = emb_dict["<unk>"]
        for i in range(max_size - l):
            emb_seq.append(emb_padding)

    emb_tensor = torch.stack(emb_seq, dim=0)

    return emb_tensor


def find_miss_token(emb_dict, mer=3):
    miss_tokens = []
    for i in range(mer):
        for token in product("AUCG", repeat=i+1):
            s = ''.join(token)
            if s not in emb_dict:
                miss_tokens.append(s)
    for tok in miss_tokens:
        tok_embs = []
        for base in product("AUCG", repeat=1):
            tok1 = tok + str(base[0])
            tok2 = str(base[0]) + tok
            if tok1 in emb_dict:
                tok_embs.append(emb_dict[tok1])
            if tok2 in emb_dict:
                tok_embs.append(emb_dict[tok2])

        sum = tok_embs[0]
        cnt = 0
        for i in range(len(tok_embs)):
            cnt += 1
            if i == 0:
                continue
            sum += tok_embs[i]
        sum /= cnt
        emb_dict[tok] = sum

    return emb_dict


def trim(emb_dict, mer):
    start = 0
    trimed_dict = {}
    trimed_dict["<unk>"] = emb_dict["<unk>"]
    if mer % 2 == 0:
        start = mer // 2
    else:
        start = mer // 2 + 1
    for i in range(start, mer+1):
        for token in product("AUCG", repeat=i):
            s = ''.join(token)
            trimed_dict[s] = emb_dict[s]
    return trimed_dict



