import itertools
from datetime import datetime
import multiprocessing
from multiprocessing import get_context
import objgraph

import gym
import numpy as np
import torch
import pathos
import pathos.multiprocessing as pathos_mp
import pathos.pools as pp
from contextlib import closing

from refine_tools.refine import sent_refine_single
from refine_tools.refine_moves import GoodPairsMove, BadPairsMove, MissingPairsMove, BoostMove
from embedding.embedding import get_dict, find_miss_token, trim, emb_full
from functools import partial

from RL_lib.environment import RNA_Env
from utils.rna_lib import get_single_index_from_graphs, get_pair_index_from_graphs, get_pair_index_from_list, \
    get_single_index_from_list


def time_sum(time_list):
    time = datetime.now()
    time -= time
    for time_ in time_list:
        time += time_
    return time


def reset_s(env):
    graph = env.reset()
    return env, graph


def get_emb(graph, emb_dict, mer):
    seq_emb = emb_full(graph.y['seq_base'], emb_dict, mer, max_size=len(graph.y['dotB']))
    # graph = env.aim_graph.clone()
    graph.x = seq_emb
    return graph
    # graph = env.get_emb(emb_dict, mer)
    # return env, graph


def step_s(args, final_step=False, type='s'):
    (env, location, mutation) = args
    graph, loc_reward, mut_reward, isterminal = env._step(location, mutation, final_step, type)
    return env, graph, loc_reward, mut_reward, isterminal

def step_s_design(args, final_step=False, type='s'):
    (env, location, mutation) = args
    graph, isterminal = env._step_design(location, mutation, final_step, type)
    return env, graph, isterminal


def single_refine(args, final_traj, final_step, move_list):
    (dotB, seq) = args
    imporoved_seq, improved_dist, is_improved = sent_refine_single(
        dotB, seq, final_traj, final_step, move_list
    )
    return improved_dist, imporoved_seq, is_improved


class Multi_RNA_Env(gym.Env):
    def __init__(self, dotB_list, id_list=None, init_seq_list=None, init_base_order=None, init_pair_order=None, edge_threshold=None, max_size=None, 
                 single_action_space=4, pair_action_space=6,
                 refine_threshold=5, improve_type='sent', n_traj=300, n_step=30, final_traj=300, final_step=30, 
                 use_task_pool=True, slack_threshold=2, 
                 use_freeze=True,
                renew_loop=10,
                mer=3, dict_dir=None, 
                use_mp=False):
        super(Multi_RNA_Env, self).__init__()

        def create_env(args,
                       init_base_order, init_pair_order, edge_threshold, max_size,
                        single_action_space, pair_action_space,
                       refine_threshold, improve_type, n_traj, n_step, final_traj, final_step,
                       use_task_pool, slack_threshold,
                       renew_loop,
                       use_freeze):
            (dotB, rna_id, init_seq) = args
            return RNA_Env(
                # initial infomation
                dotB=dotB, rna_id=rna_id, init_seq=init_seq, init_base_order=init_base_order, init_pair_order=init_pair_order, edge_threshold=edge_threshold, max_size=max_size,
                # action space
                single_action_space=single_action_space, pair_action_space=pair_action_space,
                # refinement
                refine_threshold=refine_threshold, improve_type=improve_type, n_traj=n_traj, n_step=n_step, final_traj=final_traj, final_step=final_step,
                # subtask 
                use_task_pool=use_task_pool, slack_threshold=slack_threshold,
                # solution sequence refresh 
                renew_loop=renew_loop,
                # 是否有动作限制 
                use_freeze=use_freeze
            )

        self.init_seq_list = init_seq_list

        self.dotB_list = dotB_list
        if id_list is None:
            print('no initial ids')
            self.id_list = list(range(len(self.dotB_list)))
        else:
            self.id_list = id_list
        self.index_list = list(range(len(dotB_list)))

        self.init_base_order = init_base_order
        self.init_pair_order = init_pair_order
        self.edge_threshold = edge_threshold
        self.max_size = max_size
        
        self.single_action_space = single_action_space
        self.pair_action_space = pair_action_space
        
        self.refine_threshold = refine_threshold
        self.improve_type = improve_type
        self.n_traj = n_traj
        self.n_step = n_step
        self.final_traj = final_traj
        self.final_step = final_step
        self.move_list = [GoodPairsMove(), BadPairsMove(), MissingPairsMove(), BoostMove()]

        self.use_task_pool = use_task_pool
        self.slack_threshold=slack_threshold

        self.use_freeze = use_freeze
        
        self.renew_loop = renew_loop

        self.use_mp = use_mp

        create_work = partial(
            create_env, 
            init_base_order=self.init_base_order, init_pair_order=self.init_pair_order, edge_threshold=self.edge_threshold, max_size=self.max_size,
            single_action_space=self.single_action_space, pair_action_space=self.pair_action_space,
            refine_threshold=self.refine_threshold, improve_type=self.improve_type, n_traj=self.n_traj, n_step=self.n_step, final_traj=self.final_traj, final_step=self.final_step,
            use_task_pool=self.use_task_pool, slack_threshold=self.slack_threshold,
            renew_loop=self.renew_loop,
            use_freeze=self.use_freeze
        )
        arg_list = []
        for i in range(len(self.dotB_list)):
            arg_list.append(
                (self.dotB_list[i], self.id_list[i], self.init_seq_list[i])
            )

        if use_mp:
            with pp.ProcessPool(multiprocessing.cpu_count()) as pool:
                self.RNA_list = pool.map(create_work, arg_list)
                pool.close()
                pool.join()
                pool.clear()
        else:
            self.RNA_list = list(map(create_work, arg_list))
        print('Env established.')

        self.emb_dict = {}
        self.use_emb = False

        self.mer = mer

        if dict_dir is not None:
            self.emb_dict = get_dict(dict_dir)
            self.emb_dict = find_miss_token(self.emb_dict, self.mer)
            self.emb_dict = trim(self.emb_dict, self.mer)
            self.use_emb = True

    def _get_emb(self, env):
        graph = env.get_emb(self.emb_dict, self.mer)
        return env, graph

    def get_len_list(self):
        len_list = [len(rna.aim_dotB) for rna in self.RNA_list]
        return len_list

    def get_forbidden_list(self):
        forbidden_list = [rna.forbidden_actions for rna in self.RNA_list]
        return forbidden_list

    def get_freeze_list(self):
        freeze_list = [rna.freeze_actions for rna in self.RNA_list]
        return freeze_list

    def get_energy_list(self):
        energy_list = [rna.last_energy for rna in self.RNA_list]
        return energy_list

    def get_distance_list(self):
        distance_list = [rna.last_distance for rna in self.RNA_list]
        return distance_list

    def get_aim_graph_list(self):
        aim_graph_list = [rna.aim_graph for rna in self.RNA_list]
        return aim_graph_list

    def _reset_s(self, env):
        graph = env.reset()
        return env, graph

    def get_done_list(self):
        done_list = [rna.done for rna in self.RNA_list]
        return done_list

    def get_dist_show_list(self):
        dist_list = [rna.dist_show for rna in self.RNA_list]
        return dist_list

    def get_freeze_locations(self, env, type):
        return env.get_freeze_locations(type)

    def get_freeze_location_list(self, type):
        freeze_work = partial(self.get_freeze_locations, type=type)
        freeze_location_list = list(map(
          freeze_work, self.RNA_list
        ))
        return freeze_location_list

    def get_forbidden_mutation(self, env, location, action_space):
        return env.get_forbidden_mutation(location, action_space)

    def get_forbidden_mutation_list(self, location_list, action_space):
        forbid_work = partial(self.get_forbidden_mutation, action_space=action_space)
        forbidden_mutation_list = list(
            map(
                forbid_work, self.RNA_list, location_list
            )
        )

        return forbidden_mutation_list

    def reset(self):
        if self.use_mp:
            with closing(get_context("spawn").Pool()) as pool:
                result = pool.map(reset_s, self.RNA_list)
                pool.close()
                pool.join()
            # pool.clear()

        else:
            result = map(reset_s, self.RNA_list)
        result = list(result)
        result = list(zip(*result))
        self.RNA_list = result[0]

        print('Env rest.')

        result.clear()

        graph_list = [rna.aim_graph.clone() for rna in self.RNA_list]

        if self.use_emb:
            emb_work = partial(get_emb, emb_dict=self.emb_dict, mer=self.mer)
            result = map(emb_work, graph_list)
            result = list(result)

        return result

    def get_graph_list(self):
        return [rna.aim_graph for rna in self.RNA_list]

    def _step_s(self, args, final_step=False, type='s'):
        (env, location, mutation) = args
        graph, loc_reward, mut_reward, isterminal = env._step(location, mutation, final_step, type)
        return env, graph, loc_reward, mut_reward, isterminal
    
    def _step_s_design(self, args, final_step=False, type='s'):
        (env, location, mutation) = args
        graph, isterminal = env._step_design(location, mutation, final_step, type)
        return env, graph, isterminal

    def _step(self, locations, mutations, final_step=False, type='s'):
        step_work = partial(step_s, final_step=final_step, type=type)

        arg_list = [(self.RNA_list[i], locations[i], mutations[i]) for i in range(len(self.RNA_list))]
        if self.use_mp:

            with closing(get_context("spawn").Pool(multiprocessing.cpu_count())) as pool:
                result = pool.map(step_work, arg_list)
                pool.close()
                pool.join()

        else:
            result = map(step_work, arg_list)

        result = list(result)
        result = list(zip(*result))
        self.RNA_list = result[0]

        loc_reward_list = result[2]
        mut_reward_list = result[3]
        done_list = result[4]

        is_terminal = False
        if np.all(np.array(done_list) == 1):
            is_terminal = True

        graph_list = [rna.aim_graph.clone() for rna in self.RNA_list]

        result.clear()

        if self.use_emb:
            emb_work = partial(get_emb, emb_dict=self.emb_dict, mer=self.mer)
            result = map(emb_work, graph_list)
            result = list(result)
            graph_list = result

        return graph_list, loc_reward_list, mut_reward_list, is_terminal, done_list, self.id_list.copy(), self.index_list.copy() #, time_base_emb

    def _step_design(self, locations, mutations, final_step=False, type='s'):
        step_work = partial(step_s_design, final_step=final_step, type=type)
        arg_list = [(self.RNA_list[i], locations[i], mutations[i]) for i in range(len(self.RNA_list))]
        if self.use_mp:
            with closing(get_context("spawn").Pool(multiprocessing.cpu_count())) as pool:
                result = pool.map(step_work, arg_list)
                pool.close()
                pool.join()
        else:
            result = map(step_work, arg_list)

        result = list(result)
        result = list(zip(*result))
        self.RNA_list = result[0]

        done_list = result[2]

        is_terminal = False
        if np.all(np.array(done_list) == 1):
            is_terminal = True

        # time_1 = datetime.now()
        graph_list = [rna.aim_graph.clone() for rna in self.RNA_list]

        result.clear()

        if self.use_emb:
            emb_work = partial(get_emb, emb_dict=self.emb_dict, mer=self.mer)
            result = map(emb_work, graph_list)
            result = list(result)
            graph_list = result


        return graph_list, is_terminal, done_list, self.id_list.copy(), self.index_list.copy()

    def final_refine(self,):

        id_list, dotB_list, seq_list, dist_list = self.get_best_solutions()

        arg_list = [(dotB_list[i], seq_list[i]) for i in range(len(dotB_list))]
        refine_work = partial(single_refine, final_traj=self.final_traj, final_step=self.final_step, move_list=self.move_list)
        if self.use_mp:
            with get_context("spawn").Pool() as pool:
                result = pool.map(refine_work, arg_list)
                pool.close()
                pool.join()
        else:
            result = map(refine_work, arg_list)
        result = list(result)
        result = list(zip(*result))
        dist_list = result[0]
        imp_seq_list = result[1]
        is_imp_list = result[2]

        is_terminal_list = []
        for i in range(len(id_list)):
            if is_imp_list[i]:
                self.RNA_list[i].settle_seq = list(imp_seq_list[i])
                self.RNA_list[i].best_seq = imp_seq_list[i]
                self.RNA_list[i].best_dist = dist_list[i]
            is_terminal_list.append(dist_list[i] == 0)

        result.clear()
        return id_list, dotB_list, imp_seq_list, is_terminal_list

    def _single_refine(self, args):
        (dotB, seq) = args
        imporoved_seq, improved_dist, is_improved = sent_refine_single(
            dotB, seq, self.final_traj, self.final_step, self.move_list
        )
        return improved_dist, imporoved_seq, is_improved


    def get_best_solutions(self):
        seq_list = []
        dotB_list = []
        id_list = []
        dist_list = []
        for i in range(len(self.RNA_list)):
            seq_list.append(self.RNA_list[i].best_seq)
            dotB_list.append(self.RNA_list[i].dotB)
            dist_list.append(self.RNA_list[i].best_dist)
            id_list.append(self.RNA_list[i].rna_id)
        return id_list, dotB_list, seq_list, dist_list

    def get_best_solutions_log(self):
        seq_list = []
        dotB_list = []
        id_list = []
        dist_list = []
        for i in range(len(self.RNA_list)):
            seq_list.append(self.RNA_list[i].best_seq_log)
            dotB_list.append(self.RNA_list[i].dotB)
            dist_list.append(self.RNA_list[i].best_dist_log)
            id_list.append(self.RNA_list[i].rna_id)
        return id_list, dotB_list, seq_list, dist_list

    def remove_env(self, orders):
        remove_id_list = [self.id_list[i] for i in range(0, len(self.id_list), 1) if i in orders]
        remove_index_list = [self.index_list[i] for i in range(0, len(self.index_list), 1) if i in orders]
        remove_RNA_list = [self.RNA_list[i] for i in range(0, len(self.RNA_list), 1) if i in orders]

        remove_seq_list = [rna.aim_graph.y['seq_base'] for rna in remove_RNA_list]
        remove_distance_list = [rna.last_distance for rna in remove_RNA_list]
        remove_energy_list = [rna.last_energy for rna in remove_RNA_list]
        remove_dotB_list = [rna.dotB for rna in remove_RNA_list]

        self.id_list = [self.id_list[i] for i in range(0, len(self.id_list), 1) if i not in orders]
        self.index_list = [self.index_list[i] for i in range(0, len(self.index_list), 1) if i not in orders]
        self.RNA_list = [self.RNA_list[i] for i in range(0, len(self.RNA_list), 1) if i not in orders]

        return remove_id_list, remove_dotB_list, remove_seq_list, remove_energy_list, remove_distance_list, remove_index_list

    def loop_zero(self):
        for env in self.RNA_list:
            env.loop = 0

    def _get_single_index_list(self):
        single_index_list = [torch.tensor(rna.aim_graph.y['single_index']) for rna in self.RNA_list]
        return single_index_list

    def _get_pair_index_list(self):
        pair_index_list = [torch.tensor(rna.aim_graph.y['pair_index']) for rna in self.RNA_list]
        return pair_index_list

    def _get_single_index_from_graphs(self):
        len_list = self.get_len_list()
        single_index_list = self._get_single_index_list()
        single_index, batch, index_len_list = get_single_index_from_list(len_list, single_index_list)
        return single_index, batch, index_len_list

    def _get_pair_index_from_graphs(self):
        len_list = self.get_len_list()
        pair_index_list = self._get_pair_index_list()
        pair_index, batch, index_len_list = get_pair_index_from_list(len_list, pair_index_list)
        return pair_index, batch, index_len_list
    
    def _clean(self):
        for env in self.RNA_list:
            env.close()
            del env



