import psutil


def load_dotB(dotB_root, best_dir, id_list, load_best):
    dotB_list = []
    init_seq_list = []
    for rna_id in id_list:
        dotB_dir = dotB_root + '/' + rna_id + '.rna'
        f = open(dotB_dir)
        iter_f = iter(f)
        line = list(iter_f)[-1]
        dotB = line.replace('\n', '')
        dotB_list.append(dotB)
        f.close()

        init_seq = None
        if load_best:
            # print('Load solutions.')
            design_dir = best_dir + '/' + str(rna_id) + '.csv'
            init_seq = None
            if os.path.exists(design_dir):
                f = open(design_dir)
                iter_f = iter(f)
                line = list(iter_f)[-1]
                init_seq = line.replace('\n', '')
                f.close()
        init_seq_list.append(init_seq)
    return dotB_list, init_seq_list


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

def get_lr_c(epi):
    if epi <= 10:
        return 2.4e-5
    elif epi <= 100:
        return 2.4e-5 - 1.2e-5 / 90 * (epi - 10)
    else:
        return 1.2e-5
    
def get_lr_a(epi):
    if epi <= 10:
        return 0
    elif epi <= 100:
        return 1.2e-5 / 90 * (epi - 10)
    else:
        return 1.2e-5
    

def adjust_learning_rate(optimizer, lr):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


if __name__ == "__main__":
    import argparse
    import sys
    import os
    import torch_geometric
    from tqdm import tqdm
    import numpy as np
    from pathlib import Path
    import torch
    import RNA

    from RL_lib import Multi_RNA_Env
    from utils.config import backboneParam_dict, singleParam_dict, pairParam_dict, num_change_single, num_change_pair
    from RL_lib.ppo import PPO, agent_type_list, get_element_index
    from utils.rna_lib import seq_onehot2Base, get_energy_from_onehot, get_energy_from_base, \
    get_distance_from_base, get_topology_distance, get_pair_ratio, get_real_graph, simplify_graph, \
    get_pair_index_from_graphs, get_single_index_from_graphs, location_convert_from_agent
    from RL_lib.environment import Transition

    parser = argparse.ArgumentParser()

    # Data
    parser.add_argument(
        "--dotB_dir", help="Path to folder of secondary sturcure.", default=None
    )
    parser.add_argument(
        "--best_dir", type=str, help="Path to best solutions.", default=None
    )
    parser.add_argument(
        "--id_batch_str",
        type=str,
        help="List of target structure ids to run on", default=None
    )

    # Log
    parser.add_argument(
        '--done_dir', type=str, help='Url of sturctures done and the soulutions.', default=None
    )
    parser.add_argument(
        '--loss_dir', type=str, help='The file logging the losses.', default=None
    )
    parser.add_argument(
        '--reward_dir', type=str, help='The file logging the reward.', default=None
    )

    # Model
    parser.add_argument("--backbone_dir", type=str, help="Model weights of the backbone.", default=None)
    parser.add_argument("--agent_dir", type=str, help="Model weights of the whole agent", default=None)
    # parser.add_argument("--freeze_backbone", action="store_true", help="Freeze the backbone's weights.")
    parser.add_argument("--model_save_dir", type=str, help="The folder to save the trained model's weights.", default=None)

    # Episode and epoch
    parser.add_argument("--episode", type=int, help="Playing episode.", default=1)
    parser.add_argument("--epoch", type=int, help="Playing epoch.", default=1)
    parser.add_argument("--step", type=int, help="Number of steps playing in one epoch.", default=40)
    parser.add_argument("--load_best", action='store_true', help='Load the best solution or not')

    # buffer
    parser.add_argument("--buffer_dir", type=str, help="Address for buffer storing.", default=None)
    parser.add_argument("--buffer_cnt", type=int, help="The order of this buffer.", default=1)
    parser.add_argument("--n_buffer_load", type=int, help="Number of buffers to load.", default=0)

    # device
    parser.add_argument("--gpu_order", type=int, help="The order of gup.", default=0)

    # loocal_refine
    parser.add_argument("--refine_threshold", type=int, help="The order of gup.", default=0)
    parser.add_argument("--improve_type", default='sent')
    parser.add_argument("--n_traj", type=int, help="The order of gup.", default=0)
    parser.add_argument("--n_step", type=int, help="The order of gup.", default=0)
    parser.add_argument("--final_traj", type=int, help="The order of gup.", default=0)
    parser.add_argument("--final_step", type=int, help="The order of gup.", default=0)

    # subtask pool
    parser.add_argument("--use_task_pool", action='store_true', help='Load the best solution or not')
    parser.add_argument("--slack_threshold", type=int, help="The order of gup.", default=10)

    # freeze
    parser.add_argument("--use_freeze", action='store_true', help='Load the best solution or not')

    # use multiprocess
    parser.add_argument("--use_mp", action='store_true', help='Load the best solution or not')

    # RNAFold version
    parser.add_argument("--fold_version", type=int, default=2)

    args = parser.parse_args()

    device = torch.device("cuda:{}".format(args.gpu_order) if torch.cuda.is_available() else "cpu")

    ########################### Create Environment ###########################
    id_list = args.id_batch_str.split(',')[:-1]
    dotB_list, init_seq_list = load_dotB(args.dotB_dir, args.best_dir, id_list, args.load_best)

    if args.fold_version == 1:
        RNA.params_load_RNA_Turner1999()
        print("Use RNAfold 1.")

    init_base_order = 0
    init_pair_order = 2
    edge_threshold = 0.001
    max_size = None

    single_action_space = num_change_single
    pair_action_space = num_change_pair

    refine_threshold = args.refine_threshold
    improve_type = args.improve_type
    n_traj = args.n_traj
    n_step = args.n_step
    final_traj = args.final_traj
    final_step = args.final_step

    use_task_pool = args.use_task_pool
    slack_threshold = args.slack_threshold

    use_freeze = args.use_freeze
    # embeding
    dict_dit = "./embedding/7MerVectors_2.txt"
    mer = 7

    use_mp = args.use_mp

    renew_loop = 10

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
    
    ########################### Create Agent ###########################

    K_epochs = 4
    batch_size = 2048
    eps_clip = 0.1
    actor_freeze_ep = 0
    lr_decay = 0.999
    gamma = 0.4
    use_critic = True
    buffer_vol = 1

    lr_backbone = 0.00001
    lr_actor = 0.000001  # learning rate for actor network
    lr_critic = 0.000001  # learning rate for critic network

    backbone_freeze_epi = 10

    n_dotB = len(id_list)

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
        backbone_freeze_epi=backbone_freeze_epi,
        device=device,
        buffer_dir=args.buffer_dir
    ).to(device)

    # Load Model
    
    if args.agent_dir is not None:
        if os.path.exists(args.agent_dir):
            agent.load(args.agent_dir)
            print('Load agent.')
        else:
            print("No aim parameters for agent!")

    if args.backbone_dir is not None:
        # print(args.backbone_dir)
        if os.path.exists(args.backbone_dir):
            agent.load_backbone(args.backbone_dir)
            print('Load backbone.')
        else:
            print("No aim parameters for backbone!")

    ########################### Log ###########################

    done_f = open(args.done_dir, 'a+')
    loss_f = open(args.loss_dir, 'a+')
    reward_f = open(args.reward_dir, 'a+')

    ########################### Playing ###########################

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
        loc_current_ep_reward_s = 0
        mut_current_ep_reward_s = 0
        loc_current_ep_reward_p = 0
        mut_current_ep_reward_p = 0

        state_ = torch_geometric.data.Batch.from_data_list(state).clone()
        with tqdm(total=args.step, desc='Playing...', unit='it') as pbar:
            for t in range(1, args.step + 1):
                type = t % 2
                type_word = agent_type_list[type]
                # print("step: {}".format(t))
                final_step = (t == args.step)
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

                next_state, loc_reward_list, mut_reward_list, is_termial, done_list, ids, exist_index_list= env_m._step(
                    locations, mutations, type=type_word
                )

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
                                args.episode, remove_id,
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
                                    args.episode, remove_id,
                                    dotB,
                                    sequence,
                                    energy,
                                    distance))
                            done_f.write('\n')
                            done_f.flush()

                    ids, _, best_seq_list, best_dist_list = env_m.get_best_solutions_log()
                    for rna_id, best_dist, best_seq in zip(ids, best_dist_list, best_seq_list):
                        best_design_seq_dir = args.best_dir + '/' + str(rna_id) + '.csv'
                        with open(best_design_seq_dir, 'w+') as f:
                            f.write(best_seq + '\n')
                            f.close()

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

    ########################### Train ###########################
        lr_c = get_lr_c(args.episode)
        lr_a = get_lr_a(args.episode)
        
        adjust_learning_rate(agent.optimizer_backbone, lr_a)
        if singleParam_dict["backboneParam"] is not None:
            adjust_learning_rate(agent.single_agent.optimizer_b, lr_a)
        adjust_learning_rate(agent.single_agent.optimizer_loc_a, lr_a)
        adjust_learning_rate(agent.single_agent.optimizer_mut_a, lr_a)
        adjust_learning_rate(agent.single_agent.optimizer_loc_c, lr_c)
        adjust_learning_rate(agent.single_agent.optimizer_mut_c, lr_c)
        
        if pairParam_dict["backboneParam"] is not None:
            adjust_learning_rate(agent.pair_agent.optimizer_b, lr_a)
        adjust_learning_rate(agent.pair_agent.optimizer_loc_a, lr_a)
        adjust_learning_rate(agent.pair_agent.optimizer_mut_a, lr_a)
        adjust_learning_rate(agent.pair_agent.optimizer_loc_c, lr_c)
        adjust_learning_rate(agent.pair_agent.optimizer_mut_c, lr_c)
        
        loss_loc_a_log_s, loss_mut_a_log_s, \
        loss_loc_c_log_s, loss_mut_c_log_s, \
        loss_loc_a_log_p, loss_mut_a_log_p, \
        loss_loc_c_log_p, loss_mut_c_log_p = agent.trainStep(args.episode, 
                                                             buffer_cnt=args.buffer_cnt,
                                                             n_buffer_load=args.n_buffer_load)

    ########################### Saving ###########################

        agent.save(args.model_save_dir)

    ########################### Loging ###########################

        loss_loc_a_log_s = abs(loss_loc_a_log_s)
        loss_loc_a_log_p = abs(loss_loc_a_log_p)
        loss_mut_a_log_s = abs(loss_mut_a_log_s)
        loss_mut_a_log_p = abs(loss_mut_a_log_p)

        loss_log_dict = {
            'loss_loc_a_s': loss_loc_a_log_s,
            'loss_mut_a_s': loss_mut_a_log_s,
            'loss_loc_c_s': loss_loc_c_log_s,
            'loss_mut_c_s': loss_mut_c_log_s,
            'loss_loc_a_p': loss_loc_a_log_p,
            'loss_mut_a_p': loss_mut_a_log_p,
            'loss_loc_c_p': loss_loc_c_log_p,
            'loss_mut_c_p': loss_mut_c_log_p,
        }  
        
        print("Log losses in " + args.loss_dir + '.')
        loss_log_str = dict2str(loss_log_dict)

        loss_f.write(loss_log_str)
        loss_f.write('\n')
        loss_f.flush()
        loss_f.close()

        reward_log_dict = {
            'reward_loc_p': loc_current_ep_reward_s,
            'reward_mut_p': mut_current_ep_reward_p,
            'reward_loc_s': loc_current_ep_reward_s,
            'reward_mut_s': mut_current_ep_reward_s
        }

        
        print("Log rewards in " + args.reward_dir + '.')
        reward_log_str = dict2str(reward_log_dict)

        reward_f.write(reward_log_str)
        reward_f.write('\n')
        reward_f.flush()
        reward_f.close()

    


