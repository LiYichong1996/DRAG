import torch

# cuda的GPU序号
device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")

num_change_pair = 4
num_change_single = 4

n_gnn = 4
h_s_list = [64 for _ in range(n_gnn-1)]

# backbone的参数
backboneParam_dict = {
    'conv1d_size': 32,
    'in_size': 50,
    'out_size': 16,
    'edge_size': 4,
    'hide_size_list': h_s_list,
    'n_layers': n_gnn,
    'dropout': 0.,
    'alpha': 0.5,
    'lr': 0.000001
}


# single agent 参数
singleParam_dict = {
    'device': device,
    'backboneParam': None,
    # {
    #     'in_size': 16,
    #     'out_size': 16,
    #     'hide_size_list': [32],
    #     'n_layers': 2,
    #     'edge_dim': 4,
    #     'lr': 0.000001
    # },
    'locParam': {
        'actor': {
            'in_size': 16,
            'out_size': 1,
            'hide_size_fc': 4,
            'bias': True,
            'lr': 0.000001
        },
        'critic': {
            'in_size': 16,
            'out_size': 1,
            'hide_size_fc': 4,
            'bias': True,
            'lr': 0.000001
        },
    },
    'mutParam': {
        'actor': {
            'in_size': 16,
            'out_size': num_change_single,
            'hide_size_fc': 8,
            'bias': True,
            'lr': 0.000001
        },
        'critic': {
            'in_size': 16,
            'out_size': 1,
            'hide_size_fc': 4,
            'bias': True,
            'lr': 0.000001
        }
    }
}

# pair agent 参数
pairParam_dict = {
    'device': device,
    'backboneParam': None,
    # {
    #     'in_size': 16,
    #     'out_size': 16,
    #     'hide_size_list': [32],
    #     'n_layers': 2,
    #     'edge_dim': 4,
    #     'lr': 0.000001
    # },
    'locParam': {
        'actor': {
            'in_size': 32,
            'out_size': 1,
            'hide_size_fc': 8,
            'bias': True,
            'lr': 0.000001
        },
        'critic': {
            'in_size': 32,
            'out_size': 1,
            'hide_size_fc': 8,
            'bias': True,
            'lr': 0.000001
        },
    },
    'mutParam': {
        'actor': {
            'in_size': 32,
            'out_size': num_change_pair,
            'hide_size_fc': 8,
            'bias': True,
            'lr': 0.000001
        },
        'critic': {
            'in_size': 32,
            'out_size': 1,
            'hide_size_fc': 8,
            'bias': True,
            'lr': 0.000001
        }
    }
}