import os

import RNA

import sys

from tqdm import tqdm

sys.path.append('..')
from utils.rna_lib import simple_init_sequence_pair, structure_dotB2Edge


def cheak_dotB(dotB, init_base_order=0, init_pair_order=2, action_space=4, dist_threshold=0):
    edge_index = structure_dotB2Edge(dotB)
    l = len(dotB)
    init_seq_base, _ = simple_init_sequence_pair(dotB, edge_index, init_base_order, init_pair_order, l, action_space)
    real_dotB = RNA.fold(init_seq_base)[0]

    dist = RNA.hamming(dotB, real_dotB)

    if dist <= dist_threshold:
        return False
    else:
        return True


def log_worth_dotBs(dotB_list, init_base_order=0, init_pair_order=2, action_space=4, dist_threshold=0):
    log_list = []
    with tqdm(total=len(dotB_list), desc='screening ...', unit='it') as pbar:
        for dotB in dotB_list:
            if cheak_dotB(dotB, init_base_order, init_pair_order, action_space, dist_threshold):
                log_list.append(dotB)
            pbar.update(1)

    return log_list


if __name__ == '__main__':
    data_dir = './raw/rfam_learn/train/'
    save_dir = './screened/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    file_list = os.listdir(data_dir)
    dotB_list = []
    for file in file_list:
        # file_dir = root + '/data/raw/rfam_learn/train/' + file
        file_dir = data_dir + file
        f = open(file_dir)
        iter_f = iter(f)
        for line in iter_f:
            line = line.replace('\n', '')
            dotB_list.append(line)

    screened_list = log_worth_dotBs(dotB_list, dist_threshold=5)

    l = len(screened_list)

    save_dir += 'rfam_train_5.txt'

    log_f = open(save_dir, "w+")
    for dotB in screened_list:
        log_f.write(dotB + '\n')

    print(l)
