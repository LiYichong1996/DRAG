import math
import random
from itertools import product
import RNA
import gym
import networkx
import torch
import numpy as np
from collections import namedtuple

from refine_tools.refine import sent_refine_single
from refine_tools.refine_moves import GoodPairsMove, BadPairsMove, MissingPairsMove, BoostMove
from embedding import emb_full
from utils.rna_lib import adjacent_distance_dotB, adjacent_distance_graph, adjacent_distance_graph_adj, random_init_sequence_pair, structure_dotB2Edge, get_graph, simple_init_sequence_pair, \
    seq_base2Onehot, base_pair_list_4, base_pair_list_6, base_list, edge2Adj, get_bp, edge_embedding, get_distance_from_base, rna_act
from utils.rna_tree import RNATree, get_freeze_pos_from_structure_graph, get_variable_pos_from_graph, stemloop_graph, tree2graph


Transition = namedtuple(
    'Transition',
    [
        'state',
        'location', 'mutation',
        'loc_log_prob', 'mut_log_prob',
        'loc_reward', 'mut_reward',
        'next_state', 'done',
        'type'
    ]
)


def sigmoid(input):
    return 1 / (1 + math.exp(-input))


class RNA_Env(gym.Env):
    def __init__(self, 
                # initial infomation
                dotB, rna_id=None, init_seq=None, init_base_order=None, init_pair_order=None, edge_threshold=0.1, max_size=None,
                # action space
                single_action_space=4, pair_action_space=6,
                # refinement
                refine_threshold=5, improve_type='sent', n_traj=300, n_step=30, final_traj=300, final_step=30,
                # subtask 
                use_task_pool=True, slack_threshold=2,
                # solution sequence refresh 
                renew_loop=10,
                # 是否有动作限制 
                use_freeze=False):
        
        super(RNA_Env, self).__init__()
        self.dotB = dotB
        # initial sequence
        self.length = len(dotB)
        self.init_base_order = init_base_order
        self.init_pair_order = init_pair_order
        self.single_action_space = single_action_space
        self.pair_action_space = pair_action_space
        self.rna_id = rna_id
        self.edge_threshold = edge_threshold
        self.init_seq = init_seq
        if max_size is None:
            self.max_size = self.length
        else:
            self.max_size = max_size
        self.dist_show = self.length
        self.slack_threshold = slack_threshold
        self.refine_threshold = refine_threshold
        # initial sequence
        self.edge_index = structure_dotB2Edge(self.dotB)
        if init_seq is not None:
            self.init_seq_base = init_seq
            self.init_seq_onehot = seq_base2Onehot(init_seq)
        elif self.init_base_order is not None:
            self.init_seq_base, self.init_seq_onehot = \
                simple_init_sequence_pair(self.dotB, self.edge_index, self.init_base_order, self.init_pair_order,
                                          self.max_size, self.pair_action_space)
        else:
            self.init_seq_base, self.init_seq_onehot = random_init_sequence_pair(self.dotB, self.edge_index,
                                          self.max_size, self.pair_action_space)
            
        # create Hierarchical Tree
        self.use_task_pool=use_task_pool

        self.tree = RNATree(dotB)
        self.tree.external_loop_create()
        self.tree_level = len(self.tree.branch_log.items()) - 1
        self.branchs = [[] for i in range(self.tree_level)]
        # initial aim
        self.aim_scope = None
        self.aim_graph = None
        self.aim_dotB = None
        self.aim_level = -1
        self.settle_nodes = []
        self.aim_branchs = []
        self.aim_branch = None
        # self.settled_place = []
        self.max_level = self.tree_level - 1

        self.last_energy = 1e6
        self.last_distance = self.length
        self.last_score = 0
        self.forbidden_actions = []
        self.freeze_actions = []
        self.base_pair_list = base_pair_list_4 if pair_action_space == 4 else base_pair_list_6
        self.base_lsit = base_list
        self.settle_seq = list(self.init_seq_base)

        self.last_dist_score = None
        self.last_bp_score = None
        self.renew_loop = renew_loop

        self.done = 0

        self.best_seq = None
        self.best_dist = self.length
        self.best_seq_log = None
        self.best_dist_log = self.length

        self.move_set = None
        self.improve_type = improve_type
        if self.improve_type == 'sent':
            self.move_set = [GoodPairsMove(), BadPairsMove(), MissingPairsMove(), BoostMove()]
        self.n_traj = n_traj
        self.n_step = n_step
        self.final_traj = final_traj
        self.final_step = final_step
        self.loop = 0

        self.aim_structure_graph = None
        self.use_freeze = use_freeze

    def __len__(self):
        return len(self.aim_dotB)

    class settle_branch:
        def __init__(self, scope, seq_base, seq_onehot):
            self.scope = scope
            self.seq_base = seq_base
            self.seq_onehot = seq_onehot

    def merge_sub_branch_to_loop(self, sub_branch_seqs, branch):
        loop = branch.get_loop()
        if type(loop[0]) == int:
            start = loop[0]
        else:
            start = loop[0].get_start()
        if type(loop[-1]) == int:
            end = loop[-1]
        else:
            end = loop[-1].get_end()

        self.aim_dotB = self.dotB[start:end+1] if end < self.length else self.dotB[start:]
        self.aim_graph = get_graph(self.aim_dotB, len(self.aim_dotB))
        self.aim_level = branch.get_level()

    def reset(self):
        self.log_seq = []
        self.log_forbid = []
        
        if self.init_seq is not None:
            self.init_seq_base = self.init_seq
            self.init_seq_onehot = seq_base2Onehot(self.init_seq)
        elif self.init_base_order is None:
            self.init_seq_base, self.init_seq_onehot = \
                random_init_sequence_pair(self.dotB, self.edge_index, self.max_size, self.pair_action_space)
        if self.loop % self.renew_loop != 0:
            self.settle_seq = list(self.best_seq)
        else:
            self.settle_seq = list(self.init_seq_base)
        
        self.tree.reset()
        # self.settled_place = []
        self.aim_level = self.tree_level - 1
        self.aim_branchs = self.tree.branch_log[str(self.aim_level)].copy()
        self.last_distance = 0
        self.last_aim_energy = 0
        self.last_real_energy = 0
        self.last_dist_score = 0
        self.last_energy_score = 0
        self.last_bp_score = 0
        self.last_mut_score = 0
        self.last_loc_score = 0

        self.done = 0

        if not self.use_task_pool:
            self.aim_level = 0
            self.aim_branchs = []

        self.switch_aim_branch()

        slack_skip = self.aim_level > -1 and self.last_distance <= self.slack_threshold
        
        while (slack_skip or self.last_distance == 0) and self.aim_level >-1:
            self.switch_aim_branch()
            slack_skip = self.aim_level > -1 and self.last_distance <= self.slack_threshold

        self.log_seq.append(self.aim_graph.y['seq_base'])

        self.last_bp = self.aim_graph.y['bp']
        self.last_real_energy = self.aim_graph.y['real_energy']
        self.last_aim_energy = self.aim_graph.y['aim_energy']

        delta_dist_score, distance, dist_score = self.reward_whole_distance()

        delta_energy_score, energy_score, old_energy_score = self.reward_whole_energy()

        delta_bp_score, bp_score = self.reward_whole_bp()

        if self.aim_level == -1 and self.last_distance == 0:
            self.done = True

        self.loop += 1
        
        return self.aim_graph#.clone()

    def check_settle(self, branch_scope):

        seq_base = self.settle_seq[branch_scope[0]: branch_scope[1]+1]

        seq_base = ''.join(seq_base)
        return seq_base
    
    def _stemloop_graph(self, node_cnt, branch, start):
        graph = networkx.Graph()
        first_node = node_cnt
        
        helices = []
        for helix in branch.helice:
            helix_new = [i - start for i in helix]
            helices.append(helix_new)
        graph.add_nodes_from(
            [(node_cnt,
            {
                "type": "stem",
                "pos": helices,
                "color": "red"
            })]
        )
        node_cnt += 1
        loop_node = []
        final_node = first_node
        for i, node in zip(range(len(branch.loop)), branch.loop):
            if type(node) is int:
                loop_node.append(node-start)
                if i == len(branch.loop) - 1:
                    graph.add_nodes_from(
                        [(node_cnt, {
                            "type": "loop",
                            "pos": loop_node,
                            "color": "blue"
                        })]
                    )
                    # if final_node > 0:
                    graph.add_edge(final_node, node_cnt)
                    final_node = node_cnt
                    loop_node = []
                    node_cnt += 1
            if type(node) is not int:
                if len(loop_node) > 0:
                    graph.add_nodes_from(
                        [(node_cnt, {
                            "type": "loop",
                            "pos": loop_node,
                            "color": "blue"
                        })]
                    )
                    # if final_node > 0:
                    graph.add_edge(final_node, node_cnt)
                    final_node = node_cnt
                    loop_node = []
                    node_cnt += 1
                last_node_cnt = node_cnt
                sub_graph, node_cnt = self._stemloop_graph(node_cnt, node, start)
                graph.add_nodes_from(sub_graph.nodes(data=True))
                graph.add_edges_from(sub_graph.edges)
                graph.add_edge(last_node_cnt-1, last_node_cnt)
                final_node = last_node_cnt
        graph.add_edge(first_node, final_node)
        return graph, node_cnt

    def switch_aim_branch(self):
        if len(self.aim_branchs) > 0:
            self.aim_branch = self.aim_branchs.pop()
            start, _ = self.aim_branch.scope()
            self.aim_structure_graph, _ = self._stemloop_graph(0, self.aim_branch, start)

            if self.aim_level == -1: 
                self.aim_structure_graph = tree2graph(self.tree) 

        else:
            if self.aim_level > 0:
                self.aim_level -= 1
                self.aim_branchs = self.tree.branch_log[str(self.aim_level)].copy()
                random.shuffle(self.aim_branchs)
                self.aim_branch = self.aim_branchs.pop()
                start, _ = self.aim_branch.scope()
                self.aim_structure_graph, _ = self._stemloop_graph(0, self.aim_branch, start)
            if self.aim_level == 0:
                self.aim_level -= 1
                self.aim_branch = RNATree.Base_Branch()
                self.aim_branch.set_start(0)
                self.aim_branch.set_end(self.length-1)
                self.aim_structure_graph = tree2graph(self.tree)


        self.aim_scope = self.aim_branch.scope()
        self.aim_dotB = self.dotB[self.aim_scope[0]:] if self.aim_scope[1] >= self.length - 1 \
            else self.dotB[self.aim_scope[0]:self.aim_scope[1] + 1]
        
        seq_base = self.check_settle(self.aim_scope)
        seq_onehot = seq_base2Onehot(seq_base)
        self.aim_graph = get_graph(self.aim_dotB, max_size=None, seq_base=seq_base, seq_onehot=seq_onehot,
                                                 edge_threshold=self.edge_threshold)

        # distance
        self.last_distance = adjacent_distance_graph(self.aim_graph)
        self.dist_show = self.last_distance
        if self.last_distance == 0:
            self.renew_settle_seq()
            # self.renew_settle_place()

        self.aim_l_seq = len(self.aim_graph.y['seq_base'])
        self.aim_l_dotB = len(self.aim_graph.y['dotB'])
        self.last_bp = self.aim_graph.y['bp']

        return self.aim_graph #.clone()

    def renew_settle_seq(self):
        seq = list(self.aim_graph.y['seq_base'])
        for i, j in zip(range(self.aim_scope[0], self.aim_scope[1] + 1), range(len(seq))):
            self.settle_seq[i] = seq[j]

    def renew_settle_place(self):
        seq = list(self.aim_graph.y['seq_base'])
        for i, j in zip(range(self.aim_scope[0], self.aim_scope[1]+1), range(len(seq))):
            if i not in self.settled_place:
                self.settled_place.append(i)

    def _freeze_locations(self, single_free=False, pair_free=False, type='s'):
        l = len(self.aim_graph.y['dotB'])

        diff_adj = self.aim_graph.y['aim_adj'] - self.aim_graph.y['real_adj']
        same_index = [i for i in range(l) if np.all(diff_adj[i] == 0)]

        if single_free:
            single_index = [i for i in range(l) if self.aim_graph.y['dotB'][i] == '.']
        else:
            single_index = []

        pair_free_list = []
        for index in same_index:
            if pair_free:
                if self.aim_graph.y['dotB'][index] != '.':
                    pair_free_list.append(index)

        freeze_loacations = []
        for index in same_index:
            if index not in single_index and index not in pair_free_list:
                freeze_loacations.append(index)

        freeze_out = []
        if type == 's':
            for freeze_loc in freeze_loacations:
                if self.aim_graph.y['dotB'][freeze_loc] == '.':
                    loc = torch.nonzero(torch.tensor(self.aim_graph.y['single_index']) == freeze_loc).item()
                    freeze_out.append(loc)
            # print(len(self.aim_graph.y['single_index']))
            if len(freeze_out) == len(self.aim_graph.y['single_index']):
                freeze_out = []
        else:
            for freeze_loc in freeze_loacations:
                pair_index_0 = torch.tensor(self.aim_graph.y['pair_index'])[:, 0].view(-1,)
                max_loc = torch.max(pair_index_0)
                if self.aim_graph.y['dotB'][freeze_loc] != '.' and freeze_loc <= max_loc and freeze_loc in pair_index_0:
                    loc = torch.nonzero(pair_index_0 == freeze_loc).item()
                    freeze_out.append(loc)

        return freeze_out

    def get_freeze_locations(self, type):
        if not self.use_freeze:
            return []

        freeze_loc_s, freeze_loc_p = get_freeze_pos_from_structure_graph(self.aim_graph, self.aim_structure_graph)

        if type == 's':
            freeze_locations = freeze_loc_s
        else:
            freeze_locations = freeze_loc_p

        return freeze_locations

    def get_forbidden_mutation(self, location, action_space):
        base_pair_list = base_pair_list_4 if action_space == 4 else base_pair_list_6

        forbidden_mutation_list = []
        seq_base = self.aim_graph.y['seq_base']

        if location.view(-1, ).shape[0] == 1:
            base = seq_base[location]
            forbidden_mutation = base_list.index(base)
            forbidden_mutation_list.append(forbidden_mutation)
        else:
            base = seq_base[location[0]]
            base_pair = seq_base[location[1]]
            if [base, base_pair] in base_pair_list:
                forbidden_mutation_list.append(base_pair_list.index([base, base_pair]))
        return forbidden_mutation_list


    def _step(self, location, action_mutation, final_step=False, type='s'):
        action_space = self.single_action_space
        if type == 's':
            # action_location = torch.tensor(self.aim_graph.y['single_index'])[loc_agent]
            action_location = location
        else:
            # action_location = torch.tensor(self.aim_graph.y['pair_index'])[loc_agent][0]
            action_location = location
            action_space = self.pair_action_space
        self.aim_graph = rna_act(
            self.aim_graph, action_location, action_mutation, action_space, type
        )

        real_dotB, real_energy = RNA.fold(self.aim_graph.y['seq_base'])
        self.aim_graph.y['real_energy'] = real_energy

        real_edge_index = structure_dotB2Edge(real_dotB)
        self.aim_graph.y['real_dotB'] = real_dotB
        self.aim_graph.y['real_edge_index'] = real_edge_index
        self.aim_graph.y['real_adj'] = edge2Adj(real_edge_index, len(self.aim_dotB))
        dist_tmp = adjacent_distance_graph(self.aim_graph)

        # local improvement
        is_improved = False
        if self.n_step > 0:
            self.aim_graph, is_improved = self._local_improve_for_graph(self.aim_graph, dist_tmp)

        if is_improved:
            real_dotB, real_energy = RNA.fold(self.aim_graph.y['seq_base'])
            # dist_tmp = RNA.hamming(real_dotB, self.aim_dotB)
            self.aim_graph.y['real_energy'] = real_energy
            real_edge_index = structure_dotB2Edge(real_dotB)
            self.aim_graph.y['real_dotB'] = real_dotB
            self.aim_graph.y['real_edge_index'] = real_edge_index
            self.aim_graph.y['real_adj'] = edge2Adj(real_edge_index, len(self.aim_dotB))
            dist_tmp = adjacent_distance_graph(self.aim_graph)

        aim_energy = RNA.energy_of_struct(self.aim_graph.y['seq_base'], self.aim_dotB)

        self.aim_graph.y['aim_energy'] = aim_energy

        self.aim_graph.y['bp'] = get_bp(self.aim_graph.y['seq_base'], len(self.aim_dotB))

        self.renew_settle_seq()

        delta_dist_score, distance, dist_score = self.reward_whole_distance()

        delta_energy_score, energy_score, old_energy_score = self.reward_whole_energy()

        delta_bp_score, bp_score = self.reward_whole_bp()

        reward_loc_self = self.reward_locator(action_location, action_mutation, action_space=action_space)
        # reward_loc = self.reward_locator(dist_score, bp_score)

        reward_mut_self = self.reward_mutator(action_location)

        reward_whole = 0.5 * delta_dist_score + 0.2 * delta_energy_score + 0.3 * delta_bp_score

        reward_loc = 0.7 * reward_loc_self + 0.3 * reward_whole
        reward_mut = 0.7 * reward_mut_self + 0.3 * reward_whole

        self.dist_show = distance
        self.last_distance = distance

        is_terminal = 0
        switched = False

        slack_skip = self.aim_level > -1 and self.last_distance <= self.slack_threshold

        if distance == 0 or slack_skip:
            # self.renew_settle_place()

            while (self.last_distance == 0 or slack_skip) and self.aim_level > -1:
                self.switch_aim_branch()
                switched = True
                slack_skip = self.aim_level and self.last_distance <= self.slack_threshold
        
        if self.aim_level == -1 and self.last_distance == 0:
            self.done = True
            is_terminal = 1
            reward_loc += 1
            reward_mut += 1

        self.log_seq.append(self.aim_graph.y['seq_base'])

        if switched:
            delta_dist_score, distance, dist_score = self.reward_whole_distance()

            delta_energy_score, energy_score, old_energy_score = self.reward_whole_energy()

            delta_bp_score, bp_score = self.reward_whole_bp()

            # reward_loc = self.reward_locator(action_location, action_mutation, action_space)
            reward_loc = self.reward_locator(dist_score, bp_score)

            reward_mut = self.reward_mutator(action_location, energy_score, old_energy_score)

        self.last_bp = self.aim_graph.y['bp']
        self.last_real_energy = self.aim_graph.y['real_energy']
        self.last_aim_energy = self.aim_graph.y['aim_energy']

        edge_index, edge_attr = edge_embedding(self.aim_graph.y['aim_edge_index'], self.aim_graph.y['bp'],
                                               self.edge_threshold)

        self.aim_graph.edge_index = edge_index
        self.aim_graph.edge_attr = edge_attr

        is_improved = False
        if final_step and (self.final_step > 0):
            dist, seq, is_improved = self._final_improve_for_graph(''.join(self.settle_seq), self.dotB)
            if is_improved:
                self.settle_seq = seq
            if dist == 0:
                is_terminal = 1

        self.renew_best_seq(switched, dist_tmp)

        return self.aim_graph, reward_loc, reward_mut, is_terminal

    def _step_design(self, location, action_mutation, final_step=False, type='s'):
        action_space = self.single_action_space
        if type == 's':
            action_location = location
        else:
            action_location = location
            action_space = self.pair_action_space
        self.aim_graph = rna_act(
            self.aim_graph, action_location, action_mutation, action_space, type
        )

        real_dotB, real_energy = RNA.fold(self.aim_graph.y['seq_base'])
        # dist_tmp = RNA.hamming(real_dotB, self.aim_dotB)
        self.aim_graph.y['real_energy'] = real_energy

        real_edge_index = structure_dotB2Edge(real_dotB)
        self.aim_graph.y['real_dotB'] = real_dotB
        self.aim_graph.y['real_edge_index'] = real_edge_index
        self.aim_graph.y['real_adj'] = edge2Adj(real_edge_index, len(self.aim_dotB))
        dist_tmp = adjacent_distance_graph(self.aim_graph)

        # local improvement
        is_improved = False
        if self.n_step > 0:
            self.aim_graph, is_improved = self._local_improve_for_graph(self.aim_graph, dist_tmp)

        if is_improved:
            real_dotB, real_energy = RNA.fold(self.aim_graph.y['seq_base'])
            # dist_tmp = RNA.hamming(real_dotB, self.aim_dotB)
            self.aim_graph.y['real_energy'] = real_energy
            real_edge_index = structure_dotB2Edge(real_dotB)
            self.aim_graph.y['real_dotB'] = real_dotB
            self.aim_graph.y['real_edge_index'] = real_edge_index
            self.aim_graph.y['real_adj'] = edge2Adj(real_edge_index, len(self.aim_dotB))
            dist_tmp = adjacent_distance_graph(self.aim_graph)

        self.aim_graph.y['bp'] = get_bp(self.aim_graph.y['seq_base'], len(self.aim_dotB))

        self.renew_settle_seq()

        _, distance, _ = self.reward_whole_distance()

        self.dist_show = distance
        self.last_distance = distance

        is_terminal = 0
        switched = False

        slack_skip = self.aim_level > -1 and self.last_distance <= self.slack_threshold

        if distance == 0 or slack_skip:
            # self.renew_settle_place()
            while (self.last_distance == 0 or slack_skip) and self.aim_level > -1:
                self.switch_aim_branch()
                switched = True
                slack_skip = self.aim_level and self.last_distance <= self.slack_threshold

        if self.aim_level == -1 and self.last_distance == 0:
            self.done = True
            is_terminal = 1

        self.log_seq.append(self.aim_graph.y['seq_base'])

        if switched:
            self.reward_whole_distance()

        self.last_bp = self.aim_graph.y['bp']

        edge_index, edge_attr = edge_embedding(self.aim_graph.y['aim_edge_index'], self.aim_graph.y['bp'],
                                               self.edge_threshold)

        self.aim_graph.edge_index = edge_index
        self.aim_graph.edge_attr = edge_attr

        is_improved = False
        if final_step and (self.final_step > 0):
            dist, seq, is_improved = self._final_improve_for_graph(''.join(self.settle_seq), self.dotB)
            if is_improved:
                self.settle_seq = seq
            if dist == 0:
                is_terminal = 1

        self.renew_best_seq(switched, dist_tmp)

        return self.aim_graph, is_terminal

    def _local_improve_for_graph(self, graph, dist):
        # dist = adjacent_distance_sim(graph, normalize=False)
        is_improved = False
        if dist <= self.refine_threshold and dist > 0:
            if self.improve_type == 'sent':
                dist, graph, is_improved = self._local_improve_sent(graph, self.n_traj, self.n_step)
            else:
                dist, graph, is_improved = self._local_improve(graph)
        return graph, is_improved

    def _final_improve_for_graph(self, seq_base, dotB):
        imporoved_seq, improved_dist, is_improved = sent_refine_single(
            dotB, seq_base, self.final_traj, self.final_step, self.move_set
        )
        return improved_dist, imporoved_seq, is_improved

    def _get_mutation(self, seq_list, mutations_s, mutations_p, index_s, index_p):
        for site_s, mutation_s in zip(index_s, mutations_s):
            seq_list[site_s] = base_list[int(mutation_s)]
        for site_p, mutation_p in zip(index_p, mutations_p):
            seq_list[site_p[0]] = self.base_pair_list[int(mutation_p)][0]
            seq_list[site_p[1]] = self.base_pair_list[int(mutation_p)][1]
        return seq_list

    def _local_improve(self, graph):
        is_improved = False
        max_size = graph.x.shape[0]
        dotB = graph.y['dotB']

        adj = graph.y['aim_adj'] - graph.y['real_adj']
        diff_index = [i for i in range(len(graph.y['dotB'])) if not np.all(adj[i] == 0)]
        diff_single_index = [i for i in diff_index if dotB[i] == '.']
        diff_pair_index_s = [i for i in diff_index if dotB[i] == '(']
        diff_pair_index = []
        for i in diff_pair_index_s:
            indexs = [j for j in range(len(graph.y['aim_edge_index'][0])) if graph.y['aim_edge_index'][0][j] == i]
            for index in indexs:
                if i < graph.y['aim_edge_index'][1][index] - 1:
                    diff_pair_index.append([i, int(graph.y['aim_edge_index'][1][index])])

        improved_dist = []
        old_seq_base = graph.y['seq_base']
        seq_base_list = list(old_seq_base)

        improved_seq_list = []
        improved_dotB_list = []

        bases_order_str = [str(i) for i in range(self.pair_action_space)]
        bases_order_str = ''.join(bases_order_str)

        for mutations_s in product("0123", repeat=len(diff_single_index)):
            for mutations_p in product(bases_order_str, repeat=len(diff_pair_index)):
                tmp_seq_list = seq_base_list.copy()
                tmp_seq_list = self._get_mutation(tmp_seq_list, mutations_s, mutations_p, diff_single_index,
                                                      diff_pair_index)
                improved_seq_list.append(tmp_seq_list)
                tmp_seq = ''.join(tmp_seq_list)
                changed_dotB = RNA.fold(tmp_seq)[0]
                improved_dotB_list.append(changed_dotB)
                changed_edge = structure_dotB2Edge(changed_dotB)
                changed_adj = edge2Adj(changed_edge, len(self.aim_dotB))
                changed_dist = adjacent_distance_graph_adj(graph, changed_adj)
                improved_dist.append(changed_dist)
                if changed_dist == 0:
                    is_improved = True
                    graph.x = seq_base2Onehot(tmp_seq, max_size)
                    graph.y['seq_base'] = tmp_seq
                    graph.y['real_dotB'] = changed_dotB
                    graph.y['real_edge_index'] = structure_dotB2Edge(changed_dotB)
                    graph.y['real_adj'] = changed_adj
                    return 0, graph, is_improved

        min_dist = min(improved_dist)
        min_places = [i for i in range(len(improved_dist)) if min_dist == improved_dist[i]]
        min_index = random.choice(min_places)
        if min_index > 0:
            is_improved = True
            min_seq_list = improved_seq_list[min_index]
            min_dotB = improved_dotB_list[min_index]

            min_seq = ''.join(min_seq_list)
            graph.x = seq_base2Onehot(min_seq, max_size)
            graph.y['seq_base'] = min_seq
            graph.y['real_dotB'] = min_dotB
            graph.y['real_edge_index'] = structure_dotB2Edge(min_dotB)
            graph.y['real_adj'] = edge2Adj(edge_index=graph.y['real_edge_index'], max_size=len(self.aim_dotB))

        return min_dist, graph, is_improved

    def get_emb(self, emb_dict, mer):
        seq_emb = emb_full(self.aim_graph.y['seq_base'], emb_dict, mer, max_size=len(self.aim_dotB))
        graph = self.aim_graph.clone()
        graph.x = seq_emb
        return graph

    def reward_whole_distance(self):
        distance = adjacent_distance_graph(self.aim_graph)
        dist_score = 1 / (1 + distance)
        delta_dist_score = dist_score - self.last_dist_score
        self.last_dist_score = dist_score
        return delta_dist_score, distance, dist_score

    def reward_whole_energy(self):
        # energy_score = 1 - (self.aim_graph.y['aim_energy'] - self.aim_graph.y['real_energy']) / (abs(self.aim_graph.y['aim_energy']) + abs(self.aim_graph.y['real_energy'])) # - self.aim_graph.y['real_energy']
        energy_score = 1 - (abs(self.aim_graph.y['aim_energy'] - self.aim_graph.y['real_energy'])) / (abs(self.aim_graph.y['aim_energy']) + abs(self.aim_graph.y['real_energy']) + 1)
        delta_energy_score = (energy_score - self.last_energy_score) # / (abs(self.last_energy_score) + 1)
        old_energy_score = self.last_energy_score
        self.last_energy_score = energy_score
        return delta_energy_score, energy_score, old_energy_score

    def reward_whole_bp(self):
        score = np.dot(self.aim_graph.y['aim_adj'], self.aim_graph.y['bp'])
        score = score.sum().sum()
        delta_bp_score = score - self.last_bp_score
        self.last_bp_score = score
        return delta_bp_score, score

    def get_mutation_seq(self, action):
        place = torch.div(action, self.pair_action_space, rounding_mode='trunc')
        base_pair = self.base_pair_list[action % self.pair_action_space]

        seq_base_list = list(self.aim_graph.y['seq_base'])


        seq_base_list[place] = base_pair[0]

        index = (self.aim_graph.y['aim_edge_index'][0, :] == place).nonzero()
        pair_places = self.aim_graph.y['aim_edge_index'][1, index]

        for pair_place in pair_places:
            if place < pair_place - 1 or place > pair_place + 1:
                seq_base_list[pair_place] = base_pair[1]
        seq_mutation = ''.join(seq_base_list)
        return seq_mutation

    def get_near_seq(self, location_, mutation_, action_space):
        location = location_.cpu()
        mutation = mutation_.cpu()
        near_mutation_base_order = [i for i in range(action_space) if i != mutation]

        if location.view(-1, ).shape[0] > 1:
            location = location[0]

        near_action_list = [order + action_space * location for order in near_mutation_base_order]

        near_seq_list = list(map(self.get_mutation_seq, near_action_list))

        return near_seq_list

    def seq_scores_locator(self, seq):
        l = len(seq)
        dotB_real, energy_real = RNA.fold(seq)
        real_edge = structure_dotB2Edge(dotB_real)
        real_adj = edge2Adj(real_edge, l)
        # distance = RNA.hamming(dotB_real, self.aim_dotB)
        distance = adjacent_distance_graph_adj(self.aim_graph, real_adj)
        dist_score = 1 / (1 + distance)
        return dist_score

    def reward_locator(self, location, mutation, alpha=1, beta=1, gamma=1, action_space=4):
        near_seq_list = self.get_near_seq(location, mutation, action_space)
        dist_score_list = list(map(self.seq_scores_locator, near_seq_list))

        dist_score_list.append(self.last_dist_score)

        dist_score_locator = max(dist_score_list)

        dist_reward = dist_score_locator - self.last_dist_score

        _reward_locator = alpha * dist_reward

        return _reward_locator

    def reward_mutator(self, location):
        delta_bp = self.aim_graph.y['bp'][location] - self.last_bp[location]
        adj_loc = 2 * (self.aim_graph.y['aim_adj'][location] - 0.5)

        mutator_reward = delta_bp * adj_loc
        mutator_reward = mutator_reward.sum()

        return mutator_reward

    def _local_improve_sent(self, graph, n_traj, n_step):
        seq_base = graph.y['seq_base']
        dotB = graph.y['dotB']
        imporoved_seq, improved_dist, is_improved = sent_refine_single(
            dotB, seq_base, n_traj, n_step, self.move_set
        )
        if is_improved:
            result_graph = get_graph(
                dotB=dotB, max_size=None, seq_base=imporoved_seq, edge_threshold=self.edge_threshold
            )
        else:
            result_graph = graph
        return improved_dist, result_graph, is_improved

    def renew_best_seq(self, switched, dist):
        seq_base = ''.join(self.settle_seq)
        if switched:
            real_dotB = RNA.fold(seq_base)[0]
            dist = adjacent_distance_dotB(real_dotB, self.dotB)
        if self.best_dist > dist:
            self.best_dist = dist
            self.best_seq = seq_base
        if self.best_dist_log > dist:
            self.best_dist_log = dist
            self.best_seq_log = seq_base










































