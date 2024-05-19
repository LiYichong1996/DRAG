import csv
import math
import RNA
import psutil
from torch.utils.data import BatchSampler, SubsetRandomSampler

from RL_lib import Multi_RNA_Env
from RL_lib.environment import Transition
from RL_lib.ppo import PPO, agent_type_list, get_element_index
# from utils.done_analyze import done_analyze
from utils.config import device, backboneParam_dict, singleParam_dict, pairParam_dict, num_change_single, num_change_pair
import os
import torch
import torch_geometric
import time
from datetime import datetime
# from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np
from utils import seq_onehot2Base, get_energy_from_onehot, get_energy_from_base, \
    get_distance_from_base, get_topology_distance, get_pair_ratio, get_real_graph, simplify_graph, \
    get_pair_index_from_graphs, get_single_index_from_graphs, location_convert_from_agent
import wandb

from utils.get_ids import rest_ids


def main():

    ################################### Training ###################################

    print("===============================train===============================")

    ####### initialize global hyperparameters ######

    os.environ['WANDB_START_METHOD'] = 'thread'
    torch.backends.cudnn.enable = False

    RNAfold_version = 1
    if RNAfold_version == 1:
        RNA.params_load_RNA_Turner1999()

    # 根目录
    root = os.path.dirname(os.path.realpath(__file__))
    # 当前时间
    local_time = str(time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime()))
    # wandb
    wandb.init(project='Drag_Train_v{}'.format(RNAfold_version), name=local_time, resume='allow')
    config = wandb.config

    # 设置进程池
    #####################################################

    ###################### logging ######################

    # 目录
    # 记录总目录
    log_dir_root = "/amax/data/liyichong/RNA_Split_2_data/"
    log_dir = log_dir_root + "/" + local_time + "_v{}/".format(RNAfold_version)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # 模型保存目录
    model_dir = log_dir + '/Model/'
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    done_dir = log_dir + "/done.csv"
    done_log_f = open(done_dir, "w+")

    best_design_dir = log_dir + '/best_design/'
    if not os.path.exists(best_design_dir):
        os.makedirs(best_design_dir)

    #####################################################

    ####### initialize  ######

    # 交互设置
    # 总步数
    max_epoisode = 40

    # 每episode的步数
    max_step = 4

    # 模型训练间隔
    update_timestep = 1

    # 模型保存间隔
    save_model_freq = 1

    # 动作选择模式
    action_type = 'selectAction'

    # 选择多步计算频率
    # cal_freq_start = 1 # max_ep_len
    # cal_freq_end = 1 # max_ep_len
    # cal_freq_decay = 20000


    #####################################################

    ####### initialize environment hyperparameters ######

    data_dir = root + '/data/raw/rfam_learn/train'
    file_list = os.listdir(data_dir)
    rna_id_list = list(range(1, len(file_list) + 1))
    
    # 初始化
    init_base_order = None
    init_pair_order = None
    edge_threshold = 0.001
    max_size = None
    # 动作空间
    single_action_space = num_change_single
    pair_action_space = num_change_pair
    # 本地优化
    refine_threshold = 10
    improve_type = 'sent'
    n_traj = 0
    n_step = 0
    final_traj = 0
    final_step = 0
    renew_loop = 10
    # 任务池
    use_task_pool = True
    slack_threshold = 10
    # 动作限制
    use_freeze = True
    # embeding
    dict_dit = "./embedding/7MerVectors_2.txt"
    mer = 7
    # 多进程
    n_dotB = 1000
    use_mp = True

    # best_solution_renew
    best_renew_freq = renew_loop
    
    #####################################################

    ################ PPO hyperparameters ################

    # backbone、ctritc、actor的参数详见 ./utilities/config

    # 每次训练的epoch
    K_epochs = 4

    batch_size = 1024

    # ppo的更新限制
    eps_clip = 0.1

    # actor参数冻结轮次
    actor_freeze_ep = 0

    # 学习率衰减率
    lr_decay = 0.99

    # 奖励衰减
    gamma = 0.4

    # 是否使用critic
    use_critic = True

    buffer_vol = 1

    n_buffer_load = 10

    # 学习率
    lr_backbone = 0.00001
    lr_actor = 0.000001  # learning rate for actor network
    lr_critic = 0.000001  # learning rate for critic network

    backbone_freeze_ep = 1

    # buffer_dir = log_dir_root + "/buffers/"

    backboneParam_dict['device'] = device
    singleParam_dict['device'] = device
    pairParam_dict['device'] = device

    agent = PPO(
        backboneParam_dict,
        singleParam_dict,
        pairParam_dict,
        K_epoch=K_epochs, train_batch_size=batch_size,
        eps_clips=eps_clip, gamma=gamma,
        num_graph=n_dotB,
        max_loop=buffer_vol, use_crtic=use_critic,
        backbone_freeze_epi=backbone_freeze_ep,
        device=device,
        buffer_dir=None
        
    ).to(device)

    agent.load_backbone('./pre_train/logs/PPO_logs_2023_03_25_23_56_20/module/backbone_40.pth')

    # scheduler_b = torch.optim.lr_scheduler.ExponentialLR(agent.optimizer_backbone, lr_decay)
    # if use_critic:
    #     loc_scheduler_c = torch.optim.lr_scheduler.ExponentialLR(agent.optimizer_locator_critic, lr_decay)
    #     mut_scheduler_c = torch.optim.lr_scheduler.ExponentialLR(agent.optimizer_mutator_critic, lr_decay)
    # loc_scheduler_a = torch.optim.lr_scheduler.ExponentialLR(agent.optimizer_locator_actor, lr_decay)
    # mut_scheduler_a = torch.optim.lr_scheduler.ExponentialLR(agent.optimizer_mutator_actor, lr_decay)

    #####################################################

    ################# training procedure ################

    # track total training time
    start_time = datetime.now().replace(microsecond=0)
    print("Started training at (GMT) : ", start_time)
    print("============================================================================================")

    # logging file

    # 初始化变量
    print_running_reward_loc = 0
    print_running_reward_mut = 0
    print_running_episodes = 0

    log_running_reward_loc = 0
    log_running_reward_mut = 0
    log_running_episodes = 0

    time_step = 0
    i_episode = 0

    while i_episode < max_epoisode:
        rna_id_list_, _ = rest_ids(done_dir, len(rna_id_list))
        epoch = 0
        i_episode += 1

        for index in BatchSampler(SubsetRandomSampler(range(len(rna_id_list_))), n_dotB, False):
            epoch += 1
            # load aim structures
            load_best = (i_episode % best_renew_freq == 0)
            id_list = get_element_index(rna_id_list_, index)
            dotB_list = []
            init_seq_list = []
            for rna_id in id_list:
                rna_dir = data_dir + '/' + str(rna_id) + '.rna'
                f = open(rna_dir)
                iter_f = iter(f)
                line = list(iter_f)[-1]
                dotB = line.replace('\n', '')
                dotB_list.append(dotB)

                design_dir = best_design_dir + '/' + str(rna_id) + '.csv'
                init_seq = None
                if os.path.exists(design_dir) and load_best:
                    f = open(design_dir)
                    iter_f = iter(f)
                    line = list(iter_f)[-1]
                    init_seq = line.replace('\n', '')
                init_seq_list.append(init_seq)


            # create environment
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

            state = env_m.reset()

            done_list = env_m.get_done_list()
            done_index = [i for i in range(len(done_list)) if done_list[i] == 1]
            if len(done_index) > 0:

                remove_ids, dotBs, sequences, energys, distances, remove_index_list = env_m.remove_env(done_index)

                state = [state[i] for i in range(len(state)) if i not in done_index]

                for remove_id, dotB, sequence, energy, distance in zip(remove_ids, dotBs, sequences,
                                                                       energys, distances):
                    done_log_f.write(
                        'Episode_{} Graph_{} is removed! Struct: {} | Sequence: {} | Energy: {} | Distance: {}'.format(
                            0, remove_id,
                            dotB,
                            sequence,
                            energy,
                            distance))
                    done_log_f.write('\n')
                    done_log_f.flush()

                    rna_id_list.remove(remove_id)

            if len(env_m.RNA_list) == 0 and i_episode > 1:
                print("All done!")
                break

            print("========= episode_" + str(i_episode) + " / epcoc_" + str(epoch) + " =========")
            loc_current_ep_reward = 0
            mut_current_ep_reward = 0

            if len(env_m.RNA_list) == 0:
                print("All done!")
                break
            else:
                loc_current_ep_reward_s = 0
                mut_current_ep_reward_s = 0
                loc_current_ep_reward_p = 0
                mut_current_ep_reward_p = 0



            state_ = torch_geometric.data.Batch.from_data_list(state).clone()

            with tqdm(total=max_step, desc='Playing...', unit='it') as pbar:

                for t in range(1, max_step + 1):
                    type = t % 2
                    type_word = agent_type_list[type]
                    final_step = (t == max_step)

                    freeze_location_list = env_m.get_freeze_location_list(type_word)

                    loc_index, batch, index_len_list = get_pair_index_from_graphs(state) if type \
                        else get_single_index_from_graphs(state)

                    loc_index_list = [torch.tensor(graph.y['pair_index']) for graph in state] if type \
                        else [torch.tensor(graph.y['single_index']) for graph in state]
                    action_space = env_m.pair_action_space if type else env_m.single_action_space


                    locations_agent, location_log_probs, feature_mut = agent.work_locate(
                        state_, loc_index, index_len_list,
                        freeze_location_list, select_type_=action_type, focus_type=type_word
                    )

                    locations = location_convert_from_agent(locations_agent, loc_index_list)


                    forbidden_mutation_list = env_m.get_forbidden_mutation_list(locations, action_space)

                    mutations, mutation_log_probs = agent.work_mutate(
                        feature_mut, locations_agent, index_len_list, forbidden_mutation_list, select_type=action_type,
                        focus_type=type_word
                    )


                    next_state, loc_reward_list, mut_reward_list, is_termial, done_list, ids, exist_index_list= env_m._step(
                        locations, mutations, type=type_word
                    )

                    locations_agent = locations_agent.split(1, dim=0)
                    mutations = mutations.split(1, dim=0)
                    location_log_probs = location_log_probs.split(1, dim=0)
                    mutation_log_probs = mutation_log_probs.split(1, dim=0)

                    for graph, location, mutation, loc_prob, mut_prob, loc_reward, mut_reward, next_graph, rna_id, done, exist_index in zip(
                            state_.clone().to_data_list(), locations_agent, mutations, location_log_probs, mutation_log_probs,
                            list(loc_reward_list).copy(), list(mut_reward_list).copy(), next_state, ids,
                            done_list, exist_index_list
                    ):
                        sim_graph = simplify_graph(graph)
                        sim_graph_next = simplify_graph(next_graph)
                        # trans = Transition(graph.to(device), action.item(), prob.item(), reward, next_graph, done)
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
                            done_log_f.write(
                                'Episode_{} Graph_{} is removed! Struct: {} | Sequence: {} | Energy: {} | Distance: {}'.format(
                                    i_episode, remove_id,
                                    dotB,
                                    sequence,
                                    energy,
                                    distance))
                            done_log_f.write('\n')
                            done_log_f.flush()
                            rna_id_list.remove(remove_id)

                    if final_step:
                        final_id_list, final_dotB_list, \
                        final_imp_seq_list, final_done_list = env_m.final_refine()
                        final_done_index = [i for i in range(len(final_done_list)) if final_done_list[i]]
                        if len(final_done_index) > 0:
                            remove_ids, dotBs, sequences, energys, distances, remove_index_list = env_m.remove_env(final_done_index)

                            for remove_id, dotB, sequence, energy, distance in zip(remove_ids, dotBs, sequences,
                                                                                energys, distances):
                                done_log_f.write(
                                    'Episode_{} Graph_{} is removed! Struct: {} | Sequence: {} | Energy: {} | Distance: {}'.format(
                                        i_episode, remove_id,
                                        dotB,
                                        sequence,
                                        energy,
                                        distance))
                                done_log_f.write('\n')
                                done_log_f.flush()
                                rna_id_list.remove(remove_id)

                        ids, _, best_seq_list, best_dist_list = env_m.get_best_solutions_log()
                        for rna_id, best_dist, best_seq in zip(ids, best_dist_list, best_seq_list):
                            best_design_seq_dir = best_design_dir + '/' + str(rna_id) + '.csv'
                            with open(best_design_seq_dir, 'w+') as f:
                                f.write(best_seq)

                    if is_termial:
                        break

                    if type_word == 's':
                        loc_current_ep_reward_s += np.array(loc_reward_list).mean()
                        mut_current_ep_reward_s += np.array(mut_reward_list).mean()
                    else:
                        loc_current_ep_reward_p += np.array(loc_reward_list).mean()
                        mut_current_ep_reward_p += np.array(mut_reward_list).mean()

                    state = next_state
                    state_ = torch_geometric.data.Batch.from_data_list(state)

                    memo_result = psutil.virtual_memory()
                    pbar.set_postfix({'used':memo_result.used, 'total': memo_result.total, 'ratio': memo_result.used/memo_result.total})
                    pbar.update(1)

            time_step += 1

            print_running_reward_loc += loc_current_ep_reward
            print_running_reward_mut += mut_current_ep_reward
            print_running_episodes += 1

            log_running_reward_loc += loc_current_ep_reward
            log_running_reward_mut += mut_current_ep_reward
            log_running_episodes += 1

            if time_step % update_timestep == 0:


                loss_loc_a_log_s, loss_mut_a_log_s, \
                loss_loc_c_log_s, loss_mut_c_log_s, \
                loss_loc_a_log_p, loss_mut_a_log_p, \
                loss_loc_c_log_p, loss_mut_c_log_p = agent.trainStep(i_episode, 
                                                                     buffer_cnt=epoch,
                                                                    n_buffer_load=n_buffer_load)
                loss_loc_a_log_s = abs(loss_loc_a_log_s)
                loss_loc_a_log_p = abs(loss_loc_a_log_p)
                loss_mut_a_log_s = abs(loss_mut_a_log_s)
                loss_mut_a_log_p = abs(loss_mut_a_log_p)

                log_dict = {
                    'loss_loc_a_log_s': loss_loc_a_log_s,
                    'loss_mut_a_log_s': loss_mut_a_log_s,
                    'loss_loc_c_log_s': loss_loc_c_log_s,
                    'loss_mut_c_log_s': loss_mut_c_log_s,
                    'loss_loc_a_log_p': loss_loc_a_log_p,
                    'loss_mut_a_log_p': loss_mut_a_log_p,
                    'loss_loc_c_log_p': loss_loc_c_log_p,
                    'loss_mut_c_log_p': loss_mut_c_log_p
                }

                wandb.log(log_dict)


                agent.clean_buffer()

            if time_step % save_model_freq == 0:
                save_model_dir = model_dir + '/epi_{}_epo_{}.pth'.format(i_episode, epoch)
                print("--------------------------------------------------------------------------------------------")
                print("saving model at : " + save_model_dir)
                agent.save(save_model_dir)
                print("model saved")
                print("Elapsed Time  : ", datetime.now().replace(microsecond=0) - start_time)
                print("--------------------------------------------------------------------------------------------")


    done_log_f.close()
    env_m.close()

    # print total training time
    print("============================================================================================")
    end_time = datetime.now().replace(microsecond=0)
    print("Started training at (GMT) : ", start_time)
    print("Finished training at (GMT) : ", end_time)
    print("Total training time  : ", end_time - start_time)
    print("============================================================================================")


if __name__ == "__main__":
    main()