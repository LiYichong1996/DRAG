from argparse import SUPPRESS, ArgumentDefaultsHelpFormatter, ArgumentParser
from functools import partial
import random
import time
import psutil
import argparse
import sys
import os
import torch_geometric
from tqdm import tqdm
import numpy as np
from pathlib import Path
import torch
import RNA
import wandb
from RL_lib import Multi_RNA_Env
# from test import get_graph
from utils.config import backboneParam_dict, singleParam_dict, pairParam_dict, num_change_single, num_change_pair
from RL_lib.ppo import PPO, agent_type_list, get_element_index
from utils.rna_lib import seq_onehot2Base, get_energy_from_onehot, get_energy_from_base, \
get_distance_from_base, get_topology_distance, get_pair_ratio, get_real_graph, simplify_graph, \
get_pair_index_from_graphs, get_single_index_from_graphs, location_convert_from_agent
from RL_lib.environment import Transition


DOTB_DIR = "data/eterna_v2/"

def load_dotB_single(dotB_root, best_dir, id, load_best):

    dotB_dir = dotB_root + '/{}.rna'.format(id)
    f = open(dotB_dir)
    iter_f = iter(f)
    line = list(iter_f)[-1]
    dotB = line.replace('\n', '')
    f.close()

    best_seq_list = []
    if load_best:
        # print('Load solutions.')
        design_dir = best_dir + '/' + str(id) + '.csv'
        if os.path.exists(design_dir):
            f = open(design_dir)
            iter_f = iter(f)
            for line in iter_f:
                init_seq = line.replace('\n', '')
                best_seq_list.append(init_seq)
            f.close()
        
    return dotB, best_seq_list


def dict2str(ori_dict, split_str=' '):
    out_str = ''
    for key in ori_dict.keys():
        tmp_str = ''
        tmp_str += key
        tmp_str += ':'
        tmp_str += str(ori_dict[key])
        tmp_str += split_str
        out_str += tmp_str
    split_l = len(split_str)

    out_str = out_str[:-split_l]
    return out_str




if __name__ == "__main__":
    
    
    # sys.argv = [
    #     "train_epi", 
    #     "--dotB_dir", "./data/eterna_v2/", 
    #     "--best_dir", "/amax/data/liyichong/RNA_Split_2_data/2023_08_14_09_24_26_V2/best_design/",
    #     "--id", "50",
    #     "--done_dir", "/amax/data/liyichong/RNA_Split_2_data/2023_08_14_09_24_26_V2/done_log.csv",
    #     "--loss_dir", "/amax/data/liyichong/RNA_Split_2_data/2023_08_14_09_24_26_V2/loss.txt",
    #     "--reward_dir", "/amax/data/liyichong/RNA_Split_2_data/2023_08_14_09_24_26_V2/reward.txt",
    #     "--model_save_dir", "/amax/data/liyichong/RNA_Split_2_data/2023_08_14_09_24_26_V2/models/",
    #     "--episode", "1", 
    #     "--epoch", "1",
    #     "--buffer_dir", "/amax/data/liyichong/RNA_Split_2_data/2023_08_14_09_24_26_V2/buffers/",
    #     "--buffer_cnt", "1" ,
    #     "--n_buffer_load", "20"
    # ]
    fold_version = 2
    gpu_order = 2
    id = 50
    n_gen = 100
    max_seq_log = 300
    restart_epo = 10
    max_epo = 9999
    max_step = 100

    if fold_version == 1:
        RNA.params_load_RNA_Turner1999()
        print("Use RNAfold 1.")

    device = torch.device("cuda:{}".format(gpu_order) if torch.cuda.is_available() else "cpu")

    root = "/amax/data/liyichong/RNA_Split_2_data/"
    loc_time = str(time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime()))
    wandb.init(project='Drag_Train_v{}'.format(fold_version), name=loc_time, resume='allow')

    log_root = root + "/" + loc_time + "_v{}/".format(fold_version)
    if not os.path.exists(log_root):
        os.makedirs(log_root)

    

    ########################### Log ###########################

    done_dir = log_root + '/done_log.csv'
    loss_dir = log_root + '/loss.txt'
    reward_dir = log_root + '/reward.txt'

    done_f = open(done_dir, 'a+')
    loss_f = open(loss_dir, 'a+')
    reward_f = open(reward_dir, 'a+')

    ########################### Create Agent ###########################

    # 每次训练的epoch
    K_epochs = 4

    batch_size = 2048

    # ppo的更新限制
    eps_clip = 0.1

    # actor参数冻结轮次
    actor_freeze_ep = 0

    # 学习率衰减率
    lr_decay = 0.999

    # 奖励衰减
    gamma = 0.8

    # 是否使用critic
    use_critic = True

    buffer_vol = 1

    # 学习率
    lr_backbone = 0.00001
    lr_actor = 0.000001  # learning rate for actor network
    lr_critic = 0.000001  # learning rate for critic network

    backbone_freeze_epi = 0

    n_dotB = n_gen

    backboneParam_dict['device'] = device
    singleParam_dict['device'] = device
    pairParam_dict['device'] = device

    buffer_dir = log_root + "/buffers/"

    agent = PPO(
        backboneParam_dict,
        singleParam_dict,
        pairParam_dict,
        K_epoch=K_epochs, train_batch_size=batch_size,
        eps_clips=eps_clip, gamma=gamma,
        num_graph=n_dotB,
        max_loop=buffer_vol, use_crtic=use_critic,
        backbone_freeze_epi=backbone_freeze_epi,
        device=device,
        buffer_dir=buffer_dir
    ).to(device)

    agent_dir = log_root + '/models/'
    if not os.path.exists(agent_dir):
        os.makedirs(agent_dir)

    # Load Model
    
    # if args.agent_dir is not None:
    #     if os.path.exists(args.agent_dir):
    #         agent.load(args.agent_dir)
    #         print('Load agent.')
    #     else:
    #         print("No aim parameters for agent!")

    # if args.backbone_dir is not None:
    #     # print(args.backbone_dir)
    #     if os.path.exists(args.backbone_dir):
    #         agent.load_backbone(args.backbone_dir)
    #         print('Load backbone.')
    #     else:
    #         print("No aim parameters for backbone!")
    
    
    ########################### Environment Setting ###########################

    init_base_order = None
    init_pair_order = None
    edge_threshold = 0.001
    max_size = None
    # 动作空间
    single_action_space = num_change_single
    pair_action_space = num_change_pair
    # 本地优化
    refine_threshold = 0
    improve_type = 'sent'
    n_traj = 0
    n_step = 0
    final_traj = 0
    final_step = 0
    # 任务池
    use_task_pool = True
    slack_threshold = 450
    # 动作限制
    use_freeze = True
    # embeding
    dict_dit = "./embedding/7MerVectors_2.txt"
    mer = 7
    # 多进程
    use_mp = False

    renew_loop = restart_epo

    # if not args.load_best:
    #     print('Renew the best solution.')

    # RNAfold 设置
    
    # 初始化

    ########################### Playing ###########################
    best_cnt = 1
    loss_loc_a_log_s_sum = 0
    loss_mut_a_log_s_sum = 0
    loss_loc_c_log_s_sum = 0
    loss_mut_c_log_s_sum = 0
    loss_loc_a_log_p_sum = 0
    loss_mut_a_log_p_sum = 0
    loss_loc_c_log_p_sum = 0
    loss_mut_c_log_p_sum = 0
    loc_current_ep_reward_p_sum = 0
    mut_current_ep_reward_p_sum = 0
    loc_current_ep_reward_s_sum = 0
    mut_current_ep_reward_s_sum = 0
    for epo in range(1, max_epo+1):
        print("================== epoch {} ==================".format(epo))
        # 初始化环境
        best_dir = log_root + "/best_design_{}/".format(best_cnt)
        if not os.path.exists(best_dir):
            os.makedirs(best_dir)

        dotB, old_best_seq_list = load_dotB_single(DOTB_DIR, best_dir, id, load_best=True)

        if len(old_best_seq_list) == 0:
            init_seq_list = [None for i in range(n_gen)] #* n_gen
        else:
            init_seq_list = random.choices(old_best_seq_list, k=n_gen)

        id_list = [id for i in range(n_gen)] #* n_gen
        dotB_list = [dotB for i in range(n_gen)] #* n_gen

        print("Load {} puzzels.".format(len(dotB_list)))

        env_m = Multi_RNA_Env(
                    dotB_list=dotB_list, id_list=id_list, init_seq_list=init_seq_list,
                    init_base_order=init_base_order, init_pair_order=init_pair_order,
                    edge_threshold=edge_threshold, max_size=max_size,
                    single_action_space=single_action_space, pair_action_space=pair_action_space,
                    refine_threshold=refine_threshold, improve_type=improve_type, n_traj=n_traj, n_step=n_step, final_traj=final_traj, final_step=final_step, 
                    use_task_pool=use_task_pool, slack_threshold=slack_threshold,
                    use_freeze=use_freeze,
                    renew_loop=renew_loop,
                    mer=mer, dict_dir=dict_dit,
                    use_mp=use_mp
                )

        action_type = 'selectAction'

        state = env_m.reset()
        done_list = env_m.get_done_list()
        done_index = [i for i in range(len(done_list)) if done_list[i] == 1]
        if len(done_index) > 0:

            remove_ids, dotBs, sequences, energys, distances, remove_index_list = env_m.remove_env(done_index)

            state = [state[i] for i in range(len(state)) if i not in done_index]

            for remove_id, dotB, sequence, energy, distance in zip(remove_ids, dotBs, sequences,
                                                                    energys, distances):
                done_f.write(
                    'Episode_{} Graph_{} is removed! Struct: {} | Sequence: {} | Energy: {} | Distance: {}'.format(
                        0, remove_id,
                        dotB,
                        sequence,
                        energy,
                        distance))
                done_f.write('\n')
                done_f.flush()

        if len(env_m.RNA_list) == 0:
            print("All done!")
            pass

        else:
            # 重置reward记录
            loc_current_ep_reward_s = 0
            mut_current_ep_reward_s = 0
            loc_current_ep_reward_p = 0
            mut_current_ep_reward_p = 0

            state_ = torch_geometric.data.Batch.from_data_list(state).clone()
            with tqdm(total=max_step, desc='Playing...', unit='it') as pbar:
                for t in range(1, max_step + 1):
                    type = t % 2
                    type_word = agent_type_list[type]
                    # print("step: {}".format(t))
                    final_step = (t == max_step)
                    # 智能体产生动作
                    freeze_location_list = env_m.get_freeze_location_list(type_word)

                    loc_index, batch, index_len_list = get_pair_index_from_graphs(state) if type \
                        else get_single_index_from_graphs(state)

                    loc_index_list = [torch.tensor(graph.y['pair_index']) for graph in state] if type \
                        else [torch.tensor(graph.y['single_index']) for graph in state]
                    action_space = env_m.pair_action_space if type else env_m.single_action_space

                    locations_agent, location_log_probs, feature_mut = agent.work_locate(
                        state_, loc_index, index_len_list,
                        freeze_location_list, select_type_= action_type, focus_type=type_word
                    )

                    locations = location_convert_from_agent(locations_agent, loc_index_list)

                    forbidden_mutation_list = env_m.get_forbidden_mutation_list(locations, action_space)

                    mutations, mutation_log_probs = agent.work_mutate(
                        feature_mut, locations_agent, index_len_list, forbidden_mutation_list, select_type=action_type,
                        focus_type=type_word
                    )

                    # 环境执行动作
                    next_state, loc_reward_list, mut_reward_list, is_termial, done_list, ids, exist_index_list= env_m._step(
                        locations, mutations, type=type_word
                    )

                    # 数据放入经验池
                    locations_agent = locations_agent.split(1, dim=0)
                    mutations = mutations.split(1, dim=0)
                    location_log_probs = location_log_probs.split(1, dim=0)
                    mutation_log_probs = mutation_log_probs.split(1, dim=0)

                    for graph, location, mutation, loc_prob, mut_prob, loc_reward, mut_reward, next_graph, id, done, exist_index in zip(
                            state_.clone().to_data_list(), locations_agent, mutations, location_log_probs, mutation_log_probs,
                            list(loc_reward_list).copy(), list(mut_reward_list).copy(), next_state, ids,
                            done_list, exist_index_list
                    ):
                        sim_graph = simplify_graph(graph)
                        sim_graph_next = simplify_graph(next_graph)
                        trans = Transition(
                            sim_graph,
                            location.item(), mutation.item(),
                            loc_prob.item(), mut_prob.item(),
                            loc_reward, mut_reward,
                            sim_graph_next, done, type
                        )
                        agent.storeTransition(trans, exist_index)

                    done_index = [i for i in range(len(done_list)) if done_list[i] == 1]
                    if len(done_index) > 0:
                        remove_ids, dotBs, sequences, energys, distances, remove_index_list = env_m.remove_env(done_index)

                        next_state = [next_state[i] for i in range(len(next_state)) if i not in done_index]

                        for remove_id, dotB, sequence, energy, distance, remove_index in zip(remove_ids, dotBs, sequences,
                                                                                        energys, distances, remove_index_list):
                            done_f.write(
                                'Episode_{} Graph_{} is removed! Struct: {} | Sequence: {} | Energy: {} | Distance: {}'.format(
                                    epo, remove_id,
                                    dotB,
                                    sequence,
                                    energy,
                                    distance))
                            done_f.write('\n')
                            done_f.flush()

                    if final_step:
                        final_id_list, final_dotB_list, \
                        final_imp_seq_list, final_done_list = env_m.final_refine()
                        final_done_index = [i for i in range(len(final_done_list)) if final_done_list[i]]
                        if len(final_done_index) > 0:
                            remove_ids, dotBs, sequences, energys, distances, remove_index_list = env_m.remove_env(final_done_index)

                            for remove_id, dotB, sequence, energy, distance in zip(remove_ids, dotBs, sequences,
                                                                                    energys, distances):
                                done_f.write(
                                    'Episode_{} Graph_{} is removed! Struct: {} | Sequence: {} | Energy: {} | Distance: {}'.format(
                                        epo, remove_id,
                                        dotB,
                                        sequence,
                                        energy,
                                        distance))
                                done_f.write('\n')
                                done_f.flush()

                        ids, _, best_seq_list, best_dist_list = env_m.get_best_solutions_log()
                        # for id, best_dist, best_seq in zip(ids, best_dist_list, best_seq_list):
                        #     best_design_seq_dir = args.best_dir + '/' + str(id) + '.csv'
                        #     with open(best_design_seq_dir, 'a+') as f:
                        #         f.write(best_seq + '\n')
                        #         f.close()

                        new_best_seq_list = old_best_seq_list + best_seq_list
                        if len(new_best_seq_list) > max_seq_log:
                            new_best_seq_list = random.choices(new_best_seq_list, k=max_seq_log)

                        best_design_seq_dir = best_dir + '/' + str(id) + '.csv'
                        with open(best_design_seq_dir, 'w+') as f:
                            for seq in new_best_seq_list:
                                f.write(seq + '\n')
                            f.close()

                    # 如果有序列设计完成，则直接训练
                    if is_termial:
                        # # 训练
                        break

                    # 记录当前的平均reward
                    # current_ep_reward_np += np.array(reward_list)
                    if type_word == 's':
                        loc_current_ep_reward_s += np.array(loc_reward_list).mean()
                        mut_current_ep_reward_s += np.array(mut_reward_list).mean()
                    else:
                        loc_current_ep_reward_p += np.array(loc_reward_list).mean()
                        mut_current_ep_reward_p += np.array(mut_reward_list).mean()

                    # 更新state

                    state = next_state
                    state_ = torch_geometric.data.Batch.from_data_list(state)

                        # tqdm更新显示
                    memo_result = psutil.virtual_memory()
                    pbar.set_postfix({'used':memo_result.used, 'total': memo_result.total, 'ratio': memo_result.used/memo_result.total})
                    # pbar.set_postfix({'RNAfold': args.fold_version})
                    pbar.update(1) 

    ########################### Train ###########################
        max_buffer = 2
        n_buffer_load = (epo-1) % restart_epo
        if n_buffer_load > max_buffer:
            n_buffer_load = max_buffer
        loss_loc_a_log_s, loss_mut_a_log_s, \
        loss_loc_c_log_s, loss_mut_c_log_s, \
        loss_loc_a_log_p, loss_mut_a_log_p, \
        loss_loc_c_log_p, loss_mut_c_log_p = agent.trainStep(epo, 
                                                             buffer_cnt=epo,
                                                             n_buffer_load=n_buffer_load)

    ########################### Saving ###########################
        model_save_dir = agent_dir + "/{}/".format(epo)
        agent.save(model_save_dir)

    ########################### Loging ###########################

        loss_loc_a_log_s = abs(loss_loc_a_log_s)
        loss_loc_a_log_p = abs(loss_loc_a_log_p)
        loss_mut_a_log_s = abs(loss_mut_a_log_s)
        loss_mut_a_log_p = abs(loss_mut_a_log_p)

        loss_loc_a_log_s_sum += loss_loc_a_log_s
        loss_mut_a_log_s_sum += loss_mut_a_log_s
        loss_loc_c_log_s_sum += loss_loc_c_log_s
        loss_mut_c_log_s_sum += loss_mut_c_log_s
        loss_loc_a_log_p_sum += loss_loc_a_log_p
        loss_mut_a_log_p_sum += loss_mut_a_log_p
        loss_loc_c_log_p_sum += loss_loc_c_log_p
        loss_mut_c_log_p_sum += loss_mut_c_log_p
        loc_current_ep_reward_p_sum += loc_current_ep_reward_p
        mut_current_ep_reward_p_sum += mut_current_ep_reward_p
        loc_current_ep_reward_s_sum += loc_current_ep_reward_s
        mut_current_ep_reward_s_sum += mut_current_ep_reward_s

        if epo % restart_epo == 0:
            print("Restart!")
            best_cnt += 1
            log_dict = {
                'loss_loc_a_s': loss_loc_a_log_s_sum/restart_epo,
                'loss_mut_a_s': loss_mut_a_log_s_sum/restart_epo,
                'loss_loc_c_s': loss_loc_c_log_s_sum/restart_epo,
                'loss_mut_c_s': loss_mut_c_log_s_sum/restart_epo,
                'loss_loc_a_p': loss_loc_a_log_p_sum/restart_epo,
                'loss_mut_a_p': loss_mut_a_log_p_sum/restart_epo,
                'loss_loc_c_p': loss_loc_c_log_p_sum/restart_epo,
                'loss_mut_c_p': loss_mut_c_log_p_sum/restart_epo,
                'reward_loc_p': loc_current_ep_reward_p_sum/restart_epo,
                'reward_mut_p': mut_current_ep_reward_p_sum/restart_epo,
                'reward_loc_s': loc_current_ep_reward_s_sum/restart_epo,
                'reward_mut_s': mut_current_ep_reward_s_sum/restart_epo
            }

            wandb.log(log_dict)

            loss_loc_a_log_s_sum = 0
            loss_mut_a_log_s_sum = 0
            loss_loc_c_log_s_sum = 0
            loss_mut_c_log_s_sum = 0
            loss_loc_a_log_p_sum = 0
            loss_mut_a_log_p_sum = 0
            loss_loc_c_log_p_sum = 0
            loss_mut_c_log_p_sum = 0
            loc_current_ep_reward_p_sum = 0
            mut_current_ep_reward_p_sum = 0
            loc_current_ep_reward_s_sum = 0
            mut_current_ep_reward_s_sum = 0

        # loss_log_dict = {
        #     'loss_loc_a_s': loss_loc_a_log_s,
        #     'loss_mut_a_s': loss_mut_a_log_s,
        #     'loss_loc_c_s': loss_loc_c_log_s,
        #     'loss_mut_c_s': loss_mut_c_log_s,
        #     'loss_loc_a_p': loss_loc_a_log_p,
        #     'loss_mut_a_p': loss_mut_a_log_p,
        #     'loss_loc_c_p': loss_loc_c_log_p,
        #     'loss_mut_c_p': loss_mut_c_log_p,
        # }  
        
        # print("Log losses in " + loss_dir + '.')
        # loss_log_str = dict2str(loss_log_dict)

        # loss_f.write(loss_log_str)
        # loss_f.write('\n')
        # loss_f.flush()
        # loss_f.close()

        # reward_log_dict = {
        #     'reward_loc_p': loc_current_ep_reward_s,
        #     'reward_mut_p': mut_current_ep_reward_p,
        #     'reward_loc_s': loc_current_ep_reward_s,
        #     'reward_mut_s': mut_current_ep_reward_s
        # }

        
        # print("Log rewards in " + reward_dir + '.')
        # reward_log_str = dict2str(reward_log_dict)

        # reward_f.write(reward_log_str)
        # reward_f.write('\n')
        # reward_f.flush()
        # reward_f.close()
        

    


