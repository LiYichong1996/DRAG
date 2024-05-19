import os
import torch
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data as Data_g
from torch_geometric.data import DataLoader as DataLoader_g
from torch_geometric.data import Dataset as Dataset_g
from torch_geometric.data import InMemoryDataset
import json
from tqdm import tqdm
from utils.rna_lib import get_graph
import random

def dotB_filter(dotB_list):
    dotB_list_p = []
    with tqdm(total=len(dotB_list), desc='Filtering...', unit='it') as pbar:
        for dotB in dotB_list:
            if dotB not in dotB_list_p:
                dotB_list_p.append(dotB)
            pbar.update(1)
    return dotB_list_p


def generate_rfam_all(data_dir, save_dir, maxsize=None):
    # 获取所有点括号数据
    files = os.listdir(data_dir)

    dotB_list = []

    with tqdm(total=len(files), desc='Reading files ...', unit='it') as pbar:
        for file in files:
            if not os.path.isdir(file):
                f = open(data_dir + '/' + file)
                iter_f = iter(f)

                for line in iter_f:
                    dotB_list.append(line.replace('\n', ''))
            pbar.update(1)

    # 去冗余
    dotB_list = dotB_filter(dotB_list)

    # 生成图
    graph_list = []
    with tqdm(total=len(dotB_list), desc='Generating graphs ...', unit='it') as pbar:
        for dotB in dotB_list:
            graph = get_graph(dotB=dotB)
            graph_list.append(graph)
            pbar.update(1)

    torch.save(graph_list, save_dir, _use_new_zipfile_serialization=False)

    print("Generation Finished!")



