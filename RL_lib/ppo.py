import math
import os
from functools import partial
import shutil
import psutil
import torch
import torch.nn as nn
import torch_geometric
from torch import clamp
from torch.autograd import Variable
from torch.autograd.grad_mode import no_grad
from torch.distributions import Categorical
from torch.utils.data import BatchSampler, SubsetRandomSampler
import torch.nn.functional as F
import pickle

from tqdm import tqdm
from networks.RD_GIN import BackboneNet, Locator_Actor, Locator_Critic, Mutator_Actor, Mutator_Critic
from utils.rna_lib import recover_graph, get_single_index_from_graphs, get_pair_index_from_graphs


agent_type_list = ['s', 'p']


def get_location_index(location_list, len_list):
    location_index = []
    start = 0
    for i in range(len(location_list)):
        location_index.append(location_list[i] + start)
        start += len_list[i]
    location_index = torch.tensor(location_index)
    return location_index


class Agent(nn.Module):
    def __init__(
            self, param, 
    ):
        super(Agent, self).__init__()
        self.backboneParam = param['backboneParam']
        self.locParam = param['locParam']
        self.mutParam = param['mutParam']
        self.device = param['device']
        self.backbone = None
        self.optimizer_b = None
        if self.backboneParam is not None:
            self.backbone = BackboneNet(
                self.backboneParam['in_size'], self.backboneParam['out_size'],
                self.backboneParam['edge_dim'], self.backboneParam['hide_size_list'], self.backboneParam['n_layers']
            ).to(self.device)
            self.optimizer_b = torch.optim.Adam(self.backbone.parameters(), lr=self.backboneParam['lr'])

        self.loc_actor = Locator_Actor(
            self.locParam['actor']['in_size'], self.locParam['actor']['out_size'],
            self.locParam['actor']['hide_size_fc'], self.locParam['actor']['bias']
        ).to(self.device)
        self.optimizer_loc_a = torch.optim.Adam(self.loc_actor.parameters(), lr=self.locParam['actor']['lr'])

        self.loc_critic = Locator_Critic(
            self.locParam['critic']['in_size'], self.locParam['critic']['out_size'],
            self.locParam['critic']['hide_size_fc'], self.locParam['critic']['bias']
        ).to(self.device)
        self.optimizer_loc_c = torch.optim.Adam(self.loc_critic.parameters(), self.locParam['critic']['lr'])

        self.mut_actor = Mutator_Actor(
            self.mutParam['actor']['in_size'], self.mutParam['actor']['out_size'],
            self.mutParam['actor']['hide_size_fc'], self.mutParam['actor']['bias']
        ).to(self.device)
        self.optimizer_mut_a = torch.optim.Adam(self.mut_actor.parameters(), lr=self.mutParam['actor']['lr'])

        self.mut_critic = Mutator_Critic(
            self.mutParam['critic']['in_size'], self.mutParam['critic']['out_size'],
            self.mutParam['critic']['hide_size_fc'], self.mutParam['critic']['bias']
        ).to(self.device)
        self.optimizer_mut_c = torch.optim.Adam(self.mut_critic.parameters(), self.mutParam['critic']['lr'])

    def extract(self, x, edge_index, edge_attr):
        if self.backbone is not None:
            x = self.backbone(x, edge_index, edge_attr)
        return x

    def locate(self, x, loc_index, len_list):
        x1 = x[loc_index]
        # shape0 = sum(len_list)
        x2 = x1.view(sum(len_list), -1)

        x3 = self.loc_actor(
            x2, len_list
        )
        return x3, x2

    def mutate(self, x):
        return self.mut_actor(x)

    def loc_value(self, x, loc_index, len_list, loc_batch):
        x1 = x[loc_index]
        x2 = x1.view(sum(len_list), -1)
        values = self.loc_critic(x2, loc_batch)

        return values, x2

    def forward(self, x, edge_index, edge_attr, loc_index, loc_batch, index_len_list, location_list, mutation_list):
        feature = self.extract(x, edge_index, edge_attr)

        location_values, mut_feature_all = self.loc_value(feature, loc_index, index_len_list, loc_batch)
        location_probs, _ = self.locate(feature, loc_index, index_len_list)

        location_index = get_location_index(location_list, index_len_list).to(self.device)
        feature_mut = mut_feature_all.index_select(dim=0, index=location_index)
        mutation_values  = self.mut_critic(feature_mut)
        mutation_probs = self.mut_actor(feature_mut)

        dist_loc = Categorical(location_probs)
        location_log_probs = dist_loc.log_prob(location_list)
        loction_entropy = dist_loc.entropy()

        dist_mut = Categorical(mutation_probs)
        mutation_log_probs = dist_mut.log_prob(mutation_list)
        mutation_entropy = dist_loc.entropy()

        return location_log_probs, location_values, loction_entropy, \
               mutation_log_probs, mutation_values, mutation_entropy

    def actor_freeze(self):
        self.loc_actor.eval()
        self.mut_actor.eval()

    def actor_unfreeze(self):
        self.loc_actor.train()
        self.mut_actor.train()

    def activate(self):
        if self.backbone is not None:
            for param in self.backbone.parameters():
                param.requires_grad = True
            self.backbone.train()

        for param in self.loc_critic.parameters():
            param.requires_grad = True
        for param in self.mut_critic.parameters():
            param.requires_grad = True

        for param in self.loc_actor.parameters():
            param.requires_grad = True
        for param in self.mut_actor.parameters():
            param.requires_grad = True

        self.mut_critic.train()
        self.mut_actor.train()
        self.loc_critic.train()
        self.loc_actor.train()

    def train_step(self, freeze_actor=False):
        if self.optimizer_b is not None:
            self.optimizer_b.step()
        if not freeze_actor:
            self.optimizer_loc_a.step()
            self.optimizer_mut_a.step()
        self.optimizer_loc_c.step()
        self.optimizer_mut_c.step()

    def _zero_grad(self) -> None:
        if self.optimizer_b is not None:
            self.optimizer_b.zero_grad()
        self.optimizer_loc_a.zero_grad()
        self.optimizer_loc_c.zero_grad()
        self.optimizer_mut_a.zero_grad()
        self.optimizer_mut_c.zero_grad()

    def clip_grad_norm(self, max_grad_norm):
        if self.backbone is not None:
            nn.utils.clip_grad_norm_(self.backbone.parameters(), max_grad_norm)
        nn.utils.clip_grad_norm_(self.loc_actor.parameters(), max_grad_norm)
        nn.utils.clip_grad_norm_(self.loc_critic.parameters(), max_grad_norm)
        nn.utils.clip_grad_norm_(self.mut_actor.parameters(), max_grad_norm)
        nn.utils.clip_grad_norm_(self.mut_critic.parameters(), max_grad_norm)

    def save(self, save_dir):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        if self.backbone is not None:
            torch.save(self.backbone.state_dict(), save_dir + '/backbone.pth')
        torch.save(self.loc_actor.state_dict(), save_dir + '/locator_actor.pth')
        torch.save(self.mut_actor.state_dict(), save_dir + '/mutator_actor.pth')
        torch.save(self.loc_critic.state_dict(), save_dir + '/locator_critic.pth')
        torch.save(self.mut_critic.state_dict(), save_dir + '/mutator_critic.pth')

    def load(self, model_dir):
        if not os.path.exists(model_dir):
            raise ValueError('File not exist!')


        if self.backbone is not None:
            self.backbone.load_state_dict(
                torch.load(model_dir + 'backbone.pth', map_location=lambda storage, loc: storage)
            )

        self.loc_actor.load_state_dict(
            torch.load(model_dir + 'locator_actor.pth', map_location=lambda storage, loc: storage)
        )
        self.mut_actor.load_state_dict(
            torch.load(model_dir + 'mutator_actor.pth', map_location=lambda storage, loc: storage)
        )


        self.loc_critic.load_state_dict(
            torch.load(
                model_dir + 'locator_critic.pth', map_location=lambda storage, loc: storage
            )
        )
        self.mut_critic.load_state_dict(
            torch.load(
                model_dir + 'mutator_critic.pth', map_location=lambda storage, loc: storage
            )
        )


def get_element_index(ob_list, index):
    """
    choose some element from the list
    :param ob_list: the orign list
    :param word:
    :return:
    """
    return [ob_list[i] for i in index]


def get_location_sample(pro_list_, freeze_location_list, len):
    pro_list = pro_list_.clone().view(-1, )
    # for action in forbidden_actions:
    #     pro_list[action] = 0
    for forbidden_place in freeze_location_list:
        pro_list[forbidden_place] = 0
    try:
        for i in range(len, pro_list.shape[0]):
            pro_list[i] = 0
        location = torch.multinomial(pro_list, 1).item()
    except:  
        pro_list = torch.ones(pro_list.shape, dtype=pro_list.dtype) / \
                   pro_list.shape[0]
        for i in range(len, pro_list.shape[0]):
            pro_list[i] = 0
        location = torch.multinomial(pro_list, 1).item()

    return location


def get_location_max(pro_list_, freeze_location_list):
    pro_list = pro_list_.clone().view(-1, )

    for action in freeze_location_list:
        pro_list[action] = 0
    location = torch.argmax(pro_list)
    return location.item()


def get_mutation_sample(pro_list_, forbidden_mutation_list):
    pro_list = pro_list_.clone().view(-1, )
    for forbidden_mutation in forbidden_mutation_list:
        pro_list[forbidden_mutation] = 0
    try:
        mutation = torch.multinomial(pro_list, 1).item()
    except:
        pro_list = torch.ones(pro_list.shape, dtype=pro_list.dtype) / \
                   pro_list.shape[0]
        mutation = torch.multinomial(pro_list, 1).item()
    return mutation


def get_mutation_max(pro_list_, forbidden_mutation_list):
    pro_list = pro_list_.clone().view(-1, )
    for forbidden_mutation in forbidden_mutation_list:
        pro_list[forbidden_mutation] = 0
    mutation = torch.argmax(pro_list)
    return mutation.item()


def action_convert(place, base_order, pair_space):
    action = place * pair_space + base_order
    return action


class PPO(nn.Module):
    def __init__(self,
                 backboneParam,
                 singleParam, pairParam,
                 K_epoch=5, train_batch_size=100,
                 backbone_freeze_epi=0,
                 eps_clips=0.2, gamma=0.9,
                 num_graph=100, max_grad_norm=1,
                 pair_space=4, max_loop=1, use_crtic=False, device=torch.device('cpu'),
                 buffer_dir=None
                 ):
        """
        PPO
        :param backboneParam: backbone hyper parameters
        :param criticParam: critic hyper parameters
        :param actorParam: actor hyper parameters
        :param lr_backbone: backbone learning rate
        :param lr_critic: critic learning rate
        :param lr_actor: actor learning rate
        :param K_epoch: training epoches
        :param train_batch_size: batch size
        :param actor_freeze_ep: number of epoches to freeze actor
        :param eps_clips: 
        :param gamma: reward decay
        :param num_graph: number of design rna
        :param max_grad_norm: threshold of gradient
        """
        super(PPO, self).__init__()
        self.backbone_freeze_epi = backbone_freeze_epi
        self.eps_clips = eps_clips
        self.use_cuda = (True if torch.cuda.is_available() else False)
        self.K_epoch = K_epoch
        self.gamma = gamma
        self.num_chain = num_graph
        self.max_grad_norm = max_grad_norm
        self.use_crtic = use_crtic
        self.device = device
        self.buffer_dir = buffer_dir

        self.backbone = BackboneNet(backboneParam['in_size'], backboneParam['out_size'], backboneParam['edge_size'],
                                 backboneParam['hide_size_list'], backboneParam['n_layers']).to(self.device)

        self.single_agent = Agent(singleParam)
        self.pair_agent = Agent(pairParam)

        self.batch_size = train_batch_size
        self.lr_b = backboneParam['lr']

        self.optimizer_backbone = torch.optim.Adam(self.backbone.parameters(), lr=self.lr_b)

        for param in self.backbone.parameters():
            param.requires_grad = True
        self.backbone.train()

        self.single_agent.activate()
        self.pair_agent.activate()

        self.buffer = [[] for i in range(self.num_chain)] 

        self.loss_locator_critic_list = []
        self.loss_locator_actor_list = []
        self.loss_mutator_critic_list = []
        self.loss_mutator_actor_list = []

        self.pair_space = pair_space
        self.buffer_cnt = 0
        self.buffer_loop = max_loop

    def forward(self, data_batch, location_list, mutation_list, loc_index, loc_batch, index_len_list, type='s'):
        data_batch_ = data_batch.clone().to(self.device)
        x = Variable(data_batch_.x.float().to(self.device))
        edge_index = Variable(data_batch_.edge_index.to(self.device))

        if edge_index.dtype is not torch.int64:
            edge_index = torch.tensor(edge_index, dtype=torch.int64)

        edge_attr = Variable(data_batch_.edge_attr.to(self.device))

        feature = self.backbone(x, edge_index, edge_attr)

        if type == 's':
            location_log_probs, location_values, loction_entropy, \
            mutation_log_probs, mutation_values, mutation_entropy = self.single_agent.forward(
                feature, edge_index, edge_attr,
                loc_index, loc_batch, index_len_list,
                location_list, mutation_list
            )
        else:
            location_log_probs, location_values, loction_entropy, \
            mutation_log_probs, mutation_values, mutation_entropy = self.pair_agent.forward(
                feature, edge_index, edge_attr,
                loc_index, loc_batch, index_len_list,
                location_list, mutation_list
            )

        return location_log_probs, location_values, loction_entropy, mutation_log_probs, mutation_values, mutation_entropy

    def work_locate(self, data_batch, loc_index, index_len_list, freeze_location_list, select_type_='selectAction', focus_type='s'):
        """
        choose the location
        :param data_batch: embedding of graphs 
        :param len_list: RNA
        :param type_: 
        :return: locations and the probabilities
        """
        data_batch_ = data_batch.clone().to(self.device)

        x = Variable(data_batch_.x.float().to(self.device))

        edge_index = Variable(data_batch_.edge_index.to(self.device))

        if edge_index.dtype is not torch.int64:
            edge_index = torch.tensor(edge_index, dtype=torch.int64)

        if edge_index.dtype is not torch.int64:
            edge_index = torch.tensor(edge_index, dtype=torch.int64)

        edge_attr = Variable(data_batch_.edge_attr.to(self.device))

        with no_grad():
            feature = self.backbone(x, edge_index, edge_attr)

            if focus_type == 's':
                feature_s = self.single_agent.extract(feature, edge_index, edge_attr)
                location_probs, feature_out = self.single_agent.locate(feature_s, loc_index, index_len_list)
            else:
                feature_p = self.pair_agent.extract(feature, edge_index, edge_attr)
                location_probs, feature_out = self.pair_agent.locate(feature_p, loc_index, index_len_list)
            location_probs = location_probs.cpu()
            location_prob_list = torch.split(location_probs, 1, dim=0)

        if select_type_ == 'selectAction':
            loc_work = partial(get_location_sample)
            locations = list(map(loc_work, location_prob_list, freeze_location_list, index_len_list))
        else:
            loc_work = partial(get_location_max)
            locations = list(map(loc_work, location_prob_list, freeze_location_list))

        locations = torch.tensor(list(locations), dtype=torch.long).view(-1, )
        dist = Categorical(location_probs)
        location_log_probs = dist.log_prob(locations)

        return locations.detach(), location_log_probs.detach(), feature_out

    def work_mutate(self, feature, locations, len_list, forbidden_mutation_list, select_type='selectAction',
                    focus_type='s'):
        with no_grad():
            device = feature.device
            location_index = get_location_index(locations, len_list).to(device)
            feature_mut = torch.index_select(feature, dim=0, index=location_index)

            mutation_probs = self.single_agent.mutate(feature_mut).cpu() if focus_type == 's' \
                else self.pair_agent.mutate(feature_mut).cpu()
            mutation_prob_list = torch.split(mutation_probs, 1, dim=0)

        if select_type == 'selectAction':
            mut_work = partial(get_mutation_sample)
            mutations = list(map(
                mut_work, mutation_prob_list, forbidden_mutation_list
            ))
        else:
            mut_work = partial(get_mutation_max)
            mutations = list(map(
                mut_work, mutation_prob_list, forbidden_mutation_list
            ))

        mutations = torch.tensor(list(mutations), dtype=torch.long).view(-1,)
        dist = Categorical(mutation_probs)
        mutation_log_probs = dist.log_prob(mutations)

        return mutations.detach(), mutation_log_probs.detach()

    def storeTransition(self, transition, id_chain):
        """
        save the experience data
        :param transition:
        :param id_chain: disign id
        :return:
        """
        buffer_tmp = self.buffer[id_chain] + [transition]
        self.buffer[id_chain] = buffer_tmp

    def clean_buffer(self):
        self.buffer_cnt += 1
        if self.buffer_cnt % self.buffer_loop == 0:
            self.buffer_cnt = 0
            for j in range(len(self.buffer)):
                del self.buffer[j][:]

    def save(self, model_dir):
        """
        save model parameters
        :param model_dir: data address
        :param i_episode: current round
        :return:
        """
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        single_dir = model_dir + '/single/'
        if not os.path.exists(single_dir):
            os.makedirs(single_dir)
        pair_dir = model_dir + '/pair/'
        if not os.path.exists(pair_dir):
            os.makedirs(pair_dir)
        torch.save(self.backbone.state_dict(), model_dir + 'backbone.pth')
        self.single_agent.save(single_dir)
        self.pair_agent.save(pair_dir)
        print("Saved agent parameters in " + model_dir + '.')

    def load(self, model_dir):
        """
        load model parameters
        :param model_dir: saving address
        :param i_episode: current round
        :return:
        """
        single_dir = model_dir + '/single/'
        pair_dir = model_dir + '/pair/'
        if not os.path.exists(model_dir):
            raise ValueError('File not exist!')

        self.backbone.load_state_dict(
            torch.load(model_dir + 'backbone.pth', map_location=lambda storage, loc: storage)
        )

        self.single_agent.load(single_dir)
        self.pair_agent.load(pair_dir)

        # print('Load weights from {}'.format(model_dir))

    def load_backbone(self, model_dir):
        if not os.path.exists(model_dir):
            raise ValueError('File not exist!')

        self.backbone.load_state_dict(
            torch.load(model_dir, map_location=lambda storage, loc: storage))

        # print('Load weights from {}.'.format(model_dir))

    def trainStep(self, epi, batchSize=None, buffer_cnt=1, n_buffer_load=0):
        """
        train model 
        :param epi: current round
        :param batchSize: 训练的batch
        :return:
        """
        if batchSize is None:
            batchSize = self.batch_size

        if epi <= self.backbone_freeze_epi:
            print("Backbone freezed.")
            for param in self.backbone.parameters():
                param.requires_grad = False
            self.backbone.eval()
        else:
            print("Backbone unfreezed.")
            for param in self.backbone.parameters():
                param.requires_grad = True
            self.backbone.train()

        graphs = []
        locations = []
        mutations = []
        old_loc_log_prob = []
        old_mut_log_prob = []
        Gt_loc = []
        Gt_mut = []
        type_log = []

        for id_chain in range(self.num_chain):
            # 加载数据
            graphs_tmp = [t.state for t in self.buffer[id_chain]]
            locations_tmp = [t.location for t in self.buffer[id_chain]]
            mutations_tmp = [t.mutation for t in self.buffer[id_chain]]
            loc_reward_tmp = [t.loc_reward for t in self.buffer[id_chain]]
            mut_reward_tmp = [t.mut_reward for t in self.buffer[id_chain]]
            old_loc_log_prob_tmp = [t.loc_log_prob for t in self.buffer[id_chain]]
            old_mut_log_prob_tmp = [t.mut_log_prob for t in self.buffer[id_chain]]
            done_tmp = [t.done for t in self.buffer[id_chain]]
            type_log_tmp = [t.type for t in self.buffer[id_chain]]

            R_loc_s = 0
            R_mut_s = 0
            R_loc_p = 0
            R_mut_p = 0
            Gt_loc_tmp = []
            Gt_mut_tmp = []
            for r_loc, r_mut, done, type in zip(loc_reward_tmp[::-1], mut_reward_tmp[::-1], done_tmp[::-1], type_log_tmp[::-1]):
                if done:
                    if type == 's':
                        R_loc_s = 0
                        R_mut_s = 0
                    else:
                        R_loc_p = 0
                        R_mut_p = 0
                if type == 's':
                    R_loc_s = r_loc + self.gamma * R_loc_s
                    R_mut_s = r_mut + self.gamma * R_mut_s
                    Gt_loc_tmp.insert(0, R_loc_s)
                    Gt_mut_tmp.insert(0, R_mut_s)
                else: 
                    R_loc_p = r_loc + self.gamma * R_loc_p
                    R_mut_p = r_mut + self.gamma * R_mut_p
                    Gt_loc_tmp.insert(0, R_loc_p)
                    Gt_mut_tmp.insert(0, R_mut_p)

            graphs += graphs_tmp
            Gt_loc += Gt_loc_tmp
            Gt_mut += Gt_mut_tmp
            locations += locations_tmp
            mutations += mutations_tmp
            old_loc_log_prob += old_loc_log_prob_tmp
            old_mut_log_prob += old_mut_log_prob_tmp
            type_log += type_log_tmp

        if self.buffer_dir is not None:
            self.save_buffer(buffer_cnt, 
                    graphs,
                    Gt_loc,
                    Gt_mut,
                    locations,
                    mutations,
                    old_loc_log_prob,
                    old_mut_log_prob,
                    type_log)

            graphs_l, Gt_loc_l, \
            Gt_mut_l, locations_l, \
            mutations_l, old_loc_log_prob_l, \
            old_mut_log_prob_l, type_log_l = self.load_buffers(
                buffer_cnt, n_buffer_load
            )

            graphs += graphs_l
            Gt_loc += Gt_loc_l
            Gt_mut += Gt_mut_l
            locations += locations_l
            mutations += mutations_l
            old_loc_log_prob += old_loc_log_prob_l
            old_mut_log_prob += old_mut_log_prob_l
            type_log += type_log_l

            memo_result = psutil.virtual_memory()
            print("Load buffer logs. ")
            print('used {}, total {}, ratio {}'.format(memo_result.used, memo_result.total, memo_result.used/memo_result.total))


        Gt_loc = torch.tensor(Gt_loc, dtype=torch.float).to(self.device)
        Gt_mut = torch.tensor(Gt_mut, dtype=torch.float).to(self.device)
        locations = torch.tensor(locations).view(-1,).to(self.device)
        mutations = torch.tensor(mutations).view(-1,).to(self.device)
        old_loc_log_prob = torch.tensor(old_loc_log_prob, dtype=torch.float).view(-1,).to(self.device)
        old_mut_log_prob = torch.tensor(old_mut_log_prob, dtype=torch.float).view(-1,).to(self.device)
        type_log = torch.tensor(type_log).view(-1,).to(self.device)

        with tqdm(total=self.K_epoch, desc='Training...', unit='it') as pbar:
            for _ in range(1, self.K_epoch + 1):
                loss_loc_a_log_s = 0
                loss_loc_c_log_s = 0
                loss_mut_a_log_s = 0
                loss_mut_c_log_s = 0
                loss_loc_a_log_p = 0
                loss_loc_c_log_p = 0
                loss_mut_a_log_p = 0
                loss_mut_c_log_p = 0
                n_log_s = 0
                n_log_p = 0

                for index in BatchSampler(SubsetRandomSampler(range(len(graphs))), batchSize, False):
                    type_batch = type_log[index]


                    single_index = [index[i] for i in range(len(type_batch)) if type_batch[i] == 0]
                    pair_index = [index[i] for i in range(len(type_batch)) if type_batch[i] == 1]

                    # single_type
                    loss_s = torch.tensor(0, dtype=torch.float).to(self.device)
                    if len(single_index) > 0:
                        Gt_loc_batch_s = Gt_loc[single_index]
                        Gt_mut_batch_s = Gt_mut[single_index]

                        graphs_batch_s = get_element_index(graphs, single_index)
                        graphs_batch_s = list(map(recover_graph, graphs_batch_s))

                        loc_index_s, batch_s, index_len_list_s = get_single_index_from_graphs(graphs_batch_s)

                        graphs_batch_s = torch_geometric.data.Batch.from_data_list(graphs_batch_s).to(self.device)

                        locations_batch_s = locations[single_index]
                        mutations_batch_s = mutations[single_index]

                        batch_s = batch_s.to(self.device)

                        location_log_probs_s, location_values_s, location_entropy_s, \
                        mutation_log_probs_s, mutation_values_s, mutation_entropy_s = self.forward(
                            graphs_batch_s, locations_batch_s, mutations_batch_s,
                            loc_index_s, batch_s, index_len_list_s, type='s'
                        )

                        if self.use_crtic:
                            loc_delta_s = Gt_loc_batch_s.view(-1,) - location_values_s.detach().view(-1,)
                            mut_delta_s = Gt_mut_batch_s.view(-1,) - mutation_values_s.detach().view(-1,)
                        else:
                            loc_delta_s = Gt_loc_batch_s.view(-1, )
                            mut_delta_s = Gt_mut_batch_s.view(-1,)
                        loc_advantage_s = loc_delta_s.view(-1,)
                        mut_advantage_s = mut_delta_s.view(-1,)

                        loc_ratio_s = torch.exp(location_log_probs_s - old_loc_log_prob[single_index].detach())
                        loc_surr1_s = loc_ratio_s * loc_advantage_s
                        loc_surr2_s = clamp(loc_ratio_s, 1 - self.eps_clips, 1 + self.eps_clips) * loc_advantage_s

                        loss_loc_a_s = -torch.min(loc_surr1_s, loc_surr2_s).mean()
                        loss_loc_c_s = F.mse_loss(Gt_loc_batch_s.view(-1,), location_values_s.view(-1,)) \
                            if self.use_crtic else torch.tensor(0)

                        mut_ratio_s = torch.exp(mutation_log_probs_s - old_mut_log_prob[single_index].detach())
                        mut_surr1_s = mut_ratio_s * mut_advantage_s
                        mut_surr2_s = clamp(mut_ratio_s, 1 - self.eps_clips, 1 + self.eps_clips) * mut_advantage_s

                        loss_mut_a_s = -torch.min(mut_surr1_s, mut_surr2_s).mean()
                        loss_mut_c_s = F.mse_loss(Gt_mut_batch_s.view(-1, ), mutation_values_s.view(-1, )) \
                            if self.use_crtic else torch.tensor(0)

                        loss_s = loss_loc_a_s + 0.5 * loss_loc_c_s - 0.01 * location_entropy_s.mean() + \
                                loss_mut_a_s + 0.5 * loss_mut_c_s - 0.01 * mutation_entropy_s.mean()

                    # pair type
                    loss_p = torch.tensor(0, dtype=torch.float).to(self.device)
                    if len(pair_index) > 0:
                        Gt_loc_batch_p = Gt_loc[pair_index]
                        Gt_mut_batch_p = Gt_mut[pair_index]
                        graphs_batch_p = get_element_index(graphs, pair_index)
                        graphs_batch_p = list(map(recover_graph, graphs_batch_p))

                        loc_index_p, batch_p, index_len_list_p = get_pair_index_from_graphs(graphs_batch_p)

                        graphs_batch_p = torch_geometric.data.Batch.from_data_list(graphs_batch_p).to(self.device)

                        locations_batch_p = locations[pair_index]
                        mutations_batch_p = mutations[pair_index]

                        batch_p = batch_p.to(self.device)

                        location_log_probs_p, location_values_p, location_entropy_p, \
                        mutation_log_probs_p, mutation_values_p, mutation_entropy_p = self.forward(
                            graphs_batch_p, locations_batch_p, mutations_batch_p,
                            loc_index_p, batch_p, index_len_list_p, type='p'
                        )

                        if self.use_crtic:
                            loc_delta_p = Gt_loc_batch_p.view(-1, ) - location_values_p.detach().view(-1, )
                            mut_delta_p = Gt_mut_batch_p.view(-1, ) - mutation_values_p.detach().view(-1, )
                        else:
                            loc_delta_p = Gt_loc_batch_p.view(-1, )
                            mut_delta_p = Gt_mut_batch_p.view(-1, )
                        loc_advantage_p = loc_delta_p.view(-1, )
                        mut_advantage_p = mut_delta_p.view(-1, )

                        loc_ratio_p = torch.exp(location_log_probs_p - old_loc_log_prob[pair_index].detach())
                        loc_surr1_p = loc_ratio_p * loc_advantage_p
                        loc_surr2_p = clamp(loc_ratio_p, 1 - self.eps_clips, 1 + self.eps_clips) * loc_advantage_p

                        loss_loc_a_p = -torch.min(loc_surr1_p, loc_surr2_p).mean()
                        loss_loc_c_p = F.mse_loss(Gt_loc_batch_p.view(-1, ), location_values_p.view(-1, )) \
                            if self.use_crtic else torch.tensor(0)

                        mut_ratio_p = torch.exp(mutation_log_probs_p - old_mut_log_prob[pair_index].detach())
                        mut_surr1_p = mut_ratio_p * mut_advantage_p
                        mut_surr2_p = clamp(mut_ratio_p, 1 - self.eps_clips, 1 + self.eps_clips) * mut_advantage_p

                        loss_mut_a_p = -torch.min(mut_surr1_p, mut_surr2_p).mean()
                        loss_mut_c_p = F.mse_loss(Gt_mut_batch_p.view(-1, ), mutation_values_p.view(-1, )) \
                            if self.use_crtic else torch.tensor(0)

                        loss_p = loss_loc_a_p + 0.5 * loss_loc_c_p - 0.01 * location_entropy_p.mean() + \
                                loss_mut_a_p + 0.5 * loss_mut_c_p - 0.01 * mutation_entropy_p.mean()

                    loss_all = loss_s + loss_p

                    l_s = len(single_index)
                    l_p = len(pair_index)
                    n_log_s += l_s
                    n_log_p += l_p

                    loss_loc_a_log_s += loss_loc_a_s.item() * l_s
                    loss_loc_c_log_s += loss_loc_c_s.item() * l_s
                    loss_mut_a_log_s += loss_mut_a_s.item() * l_s
                    loss_mut_c_log_s += loss_mut_c_s.item() * l_s

                    loss_loc_a_log_p += loss_loc_a_p.item() * l_p
                    loss_loc_c_log_p += loss_loc_c_p.item() * l_p
                    loss_mut_a_log_p += loss_mut_a_p.item() * l_p
                    loss_mut_c_log_p += loss_mut_c_p.item() * l_p

                    # if epi > self.backbone_freeze_epi:
                    self.optimizer_backbone.zero_grad()

                    self.single_agent._zero_grad()
                    self.pair_agent._zero_grad()

                    loss_all.backward()

                    nn.utils.clip_grad_norm_(self.backbone.parameters(), self.max_grad_norm)
                    self.single_agent.clip_grad_norm(self.max_grad_norm)
                    self.pair_agent.clip_grad_norm(self.max_grad_norm)

                    if epi > self.backbone_freeze_epi:
                        self.optimizer_backbone.step()

                    self.single_agent.train_step()
                    self.pair_agent.train_step()

                memo_result = psutil.virtual_memory()
                pbar.set_postfix({'used':memo_result.used, 'total': memo_result.total, 'ratio': memo_result.used/memo_result.total})
                # pbar.set_postfix({'RNAfold': args.fold_version})
                pbar.update(1)

                loss_loc_a_log_s = loss_loc_a_log_s / n_log_s
                loss_mut_a_log_s = loss_mut_a_log_s / n_log_s
                loss_loc_c_log_s = loss_loc_c_log_s / n_log_s
                loss_mut_c_log_s = loss_mut_c_log_s / n_log_s

                loss_loc_a_log_p = loss_loc_a_log_p / n_log_p
                loss_mut_a_log_p = loss_mut_a_log_p / n_log_p
                loss_loc_c_log_p = loss_loc_c_log_p / n_log_p
                loss_mut_c_log_p = loss_mut_c_log_p / n_log_p

        return loss_loc_a_log_s, loss_mut_a_log_s, loss_loc_c_log_s, loss_mut_c_log_s, \
               loss_loc_a_log_p, loss_mut_a_log_p, loss_loc_c_log_p, loss_mut_c_log_p

        
    def save_buffer(self, cnt, 
                graphs,
                Gt_loc,
                Gt_mut,
                locations,
                mutations,
                old_loc_log_prob,
                old_mut_log_prob,
                type_log, batch_size=1000):
        
        l = len(graphs)
        n_file = math.ceil(l / batch_size)

        if not os.path.exists(self.buffer_dir):
            os.mkdir(self.buffer_dir)
        save_folder = self.buffer_dir + '/{}/'.format(cnt)
        if not os.path.exists(save_folder):
            os.mkdir(save_folder)
        
        start = 0
        for i in range(1, n_file + 1):
            end = start + batch_size
            buffer_dict = {
                "graphs": graphs[start:end],
                "Gt_loc": Gt_loc[start:end],
                "Gt_mut": Gt_mut[start:end],
                "locations": locations[start:end],
                "mutations": mutations[start:end],
                "old_loc_log_prob": old_loc_log_prob[start:end],
                "old_mut_log_prob": old_mut_log_prob[start:end],
                "type_log": type_log[start:end]
            }
            save_dir = save_folder + '{}.pkl'.format(i)
            torch.save(buffer_dict, save_dir)
            
            start = end
            
    def load_buffers(self, cnt, n_buffer):
        graphs = []
        Gt_loc = []
        Gt_mut = []
        locations = []
        mutations = []
        old_loc_log_prob = []
        old_mut_log_prob = []
        type_log = []
        start_cnt = (cnt-n_buffer) if (cnt-n_buffer) >= 1 else 1 
        for i in range(start_cnt, cnt):
            buffer_folder = self.buffer_dir + '/{}/'.format(i)
            file_list = os.listdir(buffer_folder)
            for file in file_list:
                f_dir = buffer_folder + '/' + file
                buffer_dict = torch.load(f_dir)
                graphs += buffer_dict['graphs']
                Gt_loc += buffer_dict['Gt_loc']
                Gt_mut += buffer_dict['Gt_mut']
                locations += buffer_dict['locations']
                mutations += buffer_dict['mutations']
                old_loc_log_prob += buffer_dict['old_loc_log_prob']
                old_mut_log_prob += buffer_dict['old_mut_log_prob']
                type_log += buffer_dict['type_log']

        print("Buffer loaded.")

        # remove buffer
        buffer_folders = os.listdir(self.buffer_dir)
        for folder_name in buffer_folders:
            if int(folder_name) < start_cnt:
                buffer_folder = self.buffer_dir + '/' + folder_name
                shutil.rmtree(buffer_folder)

        return graphs, Gt_loc, Gt_mut, locations, mutations, old_loc_log_prob, old_mut_log_prob, type_log
    
    def get_act_probs(self, data_batch, loc_index_s, index_s_len_list, loc_index_p, index_p_len_list):
        data_batch_ = data_batch.clone()

        x = Variable(data_batch_.x.float())

        edge_index = Variable(data_batch_.edge_index)

        if edge_index.dtype is not torch.int64:
            edge_index = torch.tensor(edge_index, dtype=torch.int64)

        if edge_index.dtype is not torch.int64:
            edge_index = torch.tensor(edge_index, dtype=torch.int64)

        edge_attr = Variable(data_batch_.edge_attr)

        with no_grad():
            feature = self.backbone(x, edge_index, edge_attr)

            feature_s = self.single_agent.extract(feature, edge_index, edge_attr)
            loc_s_probs, feature_s_out = self.single_agent.locate(feature_s, loc_index_s, index_s_len_list)
            loc_s_prob_list = torch.split(loc_s_probs, 1, dim=0)

            feature_p = self.pair_agent.extract(feature, edge_index, edge_attr)
            loc_p_probs, feature_p_out = self.pair_agent.locate(feature_p, loc_index_p, index_p_len_list)
            loc_p_probs = loc_p_probs.cpu()
            loc_p_prob_list = torch.split(loc_p_probs, 1, dim=0)

            mut_s_probs = self.single_agent.mutate(feature_s_out)

            start_s = 0
            mut_s_prob_list = []
            for l in index_s_len_list:
                end_s = start_s + l
                mut_s_prob_list.append(mut_s_probs[start_s:end_s])
                start_s = end_s

            mut_p_probs = self.pair_agent.mutate(feature_p_out)

            start_p = 0
            mut_p_prob_list = []
            for l in index_p_len_list:
                end_p = start_p + l
                mut_p_prob_list.append(mut_p_probs[start_p:end_p])
                start_p = end_p

        return loc_s_prob_list, mut_s_prob_list, loc_p_prob_list, mut_p_prob_list




    