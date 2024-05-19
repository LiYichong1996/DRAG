import math
import wandb

def log_analysis(loss_dir, project_name, name, period=1):

    wandb.init(project=project_name, name=name, resume='allow')

    f = open(loss_dir)
    f_iter = iter(f)

    data_dict = {}
    for line in f_iter:
        
        words = line.replace('\n', '').split(' ')
        if len(words) == 1:
            pass
        else:
            for word in words:
                word_list = word.split(':')
                key = word_list[0]
                value = float(word_list[1])
                if key in data_dict:
                    data_dict[key] = data_dict[key] + [value]
                else:
                    data_dict[key] = [value]

    for key in data_dict:
        data_dict[key] = cal_period_avg(data_dict[key], period)
        l = len(data_dict[key])

    for i in range(l):
        log_dict = {}
        for key in data_dict:
            log_dict[key] = data_dict[key][i]
        wandb.log(log_dict)


def cal_period_avg(data_list, period=1):
    if period == 1:
        return data_list
    else:
        new_data_list = []
        start = 0
        for i in range(math.ceil(len(data_list) / period)):
            end = start + period
            if end > len(data_list):
                break
            data_perid = data_list[start:end]
            data_avg = sum(data_perid) / period
            new_data_list.append(data_avg)
            start = end
    return new_data_list


if __name__ == '__main__':
    loss_dir = '/amax/data/liyichong/RNA_Split_2_data/2023_08_09_22_49_33_V2_pre/loss.txt'
    reward_dir = '/amax/data/liyichong/RNA_Split_2_data/2023_08_09_22_49_33_V2_pre/reward.txt'

    project_name = 'train_analysis'

    log_analysis(reward_dir, project_name, 'reward_with_pre', period=10)



