import os
import matplotlib.pyplot as plt

def check_done_id(done_dir):
    data_dir = '/home/liyichong/Drag/data/eterna_v2/'
    file_list = os.listdir(data_dir)
    dotB_list = []
    for file in file_list:
        file_dir = data_dir + file
        f = open(file_dir)
        iter_f = iter(f)
        for line in iter_f:
            line = line.replace('\n', '')
            dotB_list.append(line)
    
    done_dotB_list = []
    done_id_list = []
    done_dotB_dir = done_dir
    done_f = open(done_dotB_dir)
    done_iter = iter(done_f)
    for done_str in done_iter:
        words = done_str.split(' ')
        # done_dotB = words[5]
        # done_dotB_list.append(done_dotB)
        done_id = words[1].split('_')[1]
        done_id_list.append(int(done_id))


    # done_id_list = []
    # for done_dotB in done_dotB_list:
    #     id = dotB_list.index(done_dotB)
    #     done_id_list.append(id)

    done_id_list = list(
        set(done_id_list)
    )

    done_id_list.sort()

    print(done_id_list)
    print(len(done_id_list))
    print('final')


def check_id_batch():
    id_batch_dir = "/amax/data/liyichong/RNA_Split_2_data/design_2023_05_05_02_03_51/id_batch_1.txt"
    f = open(id_batch_dir)
    id_list = []
    iter_f = iter(f)
    for line in iter_f:
        words = line.split(',')
        id_list_tmp = [words[i].replace(' ', '') for i in range(len(words)-1)]
        id_list += id_list_tmp
    id_list.sort()
    print(id_list)
    print(1)


def check_done_step(done_dir):
    data_dir = '/home/liyichong/Drag/data/eterna_v2/'
    file_list = os.listdir(data_dir)
    dotB_list = []
    for file in file_list:
        file_dir = data_dir + file
        f = open(file_dir)
        iter_f = iter(f)
        for line in iter_f:
            line = line.replace('\n', '')
            dotB_list.append(line)
    
    done_dotB_list = []
    done_id_list = []
    done_dotB_dir = done_dir
    done_f = open(done_dotB_dir)
    done_iter = iter(done_f)
    step_cnt = 1
    last_step = -1

    # max_step = 0
    done_dict = {}
    for done_str in done_iter:
        words = done_str.split(' ')
        # done_dotB = words[5]
        # done_dotB_list.append(done_dotB)
        done_step = int(words[0].split('_')[1])

        if done_step == 0 and last_step != 0 and last_step >= 0:
            done_dict[str(step_cnt)] = list(set(done_id_list))
            step_cnt += 1
            done_id_list = []
            
        done_id = words[1].split('_')[1]
        done_id_list.append(int(done_id))

        last_step = done_step


    last_done_ids = []
    acc_done_dict = {}
    for key in done_dict:
        last_done_ids += done_dict[key]
        acc_done_dict[key] = list(set(last_done_ids))

    return acc_done_dict


def done_folder_check(done_folder):
    all_f_name_list = os.listdir(done_folder)

    f_list = [name for name in all_f_name_list if 'done_log' in name]

    id_list = []
    for name in f_list:
        f_dir = done_folder + name
        f = open(f_dir)
        f_iter = iter(f)
        for line in f_iter:
            words = line.split(' ')
            dotB_id = int(
                words[1].split("_")[-1]
            )
            id_list.append(dotB_id)
        f.close()
    
    
    id_list = list(
        set(id_list)
    )
    id_list.sort()
    # print(id_list)
    # print(len(id_list))
    return id_list


if __name__ == '__main__':
    # done_dir_full = '/amax/data/liyichong/RNA_Split_2_data/2023_08_14_09_35_10_V2/done_log.csv'
    # done_dir_naive = './result/done_log.csv'
    # # check_done_id(done_dir)
    # acc_done_dict_f = check_done_step(done_dir_full)
    # acc_done_dict_n = check_done_step(done_dir_naive)

    # train_step = list(range(1, len(acc_done_dict_n)))

    # y_f = [len(acc_done_dict_f[key]) for key in acc_done_dict_f][:12]
    # y_n = [len(acc_done_dict_n[key]) for key in acc_done_dict_n][:12]

    # y_f = [43, 51, 52, 56, 56, 57, 58, 58, 59, 59, 59, 59]
    # y_n = [47, 52, 52, 55, 55, 57, 57, 57, 57, 57, 57, 57]

    # plt.plot(train_step, y_f, label='Intricate graphs')
    # plt.plot(train_step, y_n, label='Simlpe graphs')

    # plt.xlabel("Training epoch")
    # plt.ylabel("Number of solved puzzles")

    # plt.legend()

    # plt.show()

    # plt.savefig("./result/grap_ab.jpg")
    root = '/amax/data/liyichong/rest_try/2024_01_11_17_14_48_v2/'
    folder_list = os.listdir(root)
    # folder_list = [folder for folder in folder_list if 'v2' in folder or 'V2' in folder]
    done_id_list = []
    for folder in folder_list:
        id_list = done_folder_check(
            root + folder + '/'
        )
        done_id_list += id_list
    done_id_list = list(
        set(done_id_list)
    )
    done_id_list.sort()
    print(done_id_list)
    print(len(done_id_list))
    