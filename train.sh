MAX_EPISODE=1000
LOCAL_DIR=$(cd $(dirname $0); pwd)
echo "Local ${LOCAL_DIR}"
DATA_DIR="${LOCAL_DIR}/data/eterna_v2/"
LOG_DIR_ROOT="/amax/data/liyichong/RNA_Split_2_data/"
PLAY_BATCH_SIZE=100
N_BUFFER_LOAD=10
STEP=40
N_TRAJ=0
N_STEP=0
F_TRAJ=0
F_STEP=0
SLACK_THRESHOLD=450
GPU=1
RWNEW_FRE=4
RNAfold_Version=2

time_now=`date +%Y_%m_%d_%H_%M_%S`
# time_now='eterna_v2'
epi=0
cd ${DATA_DIR}
n_rna=`ls -l |grep "^-"|wc -l`
cd ${LOCAL_DIR}
echo "Train on ${n_rna} structures of RNA."
log_dir="${LOG_DIR_ROOT}${time_now}_V${RNAfold_Version}/"
mkdir ${log_dir}
echo "Log in ${log_dir}"
model_folder_root="${log_dir}models"
mkdir ${model_folder_root}
echo "Save models in ${model_folder_root}"
done_dir="${log_dir}done_log"

best_dir="${log_dir}best_design/"
mkdir ${best_dir}
echo "Best designs of RNAs in ${best_dir}"
loss_dir="${log_dir}loss.txt"
touch ${loss_dir}
echo "Losses of training in ${loss_dir}"
reward_dir="${log_dir}reward.txt"
touch ${log_dir}
echo "Rewards of training in ${reward_dir}"
buffer_dir="${log_dir}buffers/"
mkdir ${buffer_dir}
echo "Save buffers at ${buffer_dir}"

buffer_cnt=0
init_seed=0

echo "++++++++++++++++++ Train ++++++++++++++++++"
while [ "$epi" -le "$MAX_EPISODE" ]
do
    epi_f=`expr $epi % 10`
    epi=`expr $epi + 1`
    echo "================== Episode ${epi} =================="
    epo=0
    id_batch_dir="${log_dir}id_batch_${epi}.txt"
    echo "Save id batchs in ${id_batch_dir}"

    if [ "$epi_f" -eq 0 ]
    then
        done_dir_f="${done_dir}_${epi}.csv"
        touch ${done_dir_f}
        echo "Maintain sloved puzzles."
        python ./utils/get_ids.py --done_dir ${done_dir_f} --n_rna ${n_rna} --batch_size ${PLAY_BATCH_SIZE} --save_dir ${id_batch_dir} --forbid_id 22 --maintain_solved --set_init_ids
    else
        echo "Remove solved puzzles."
        python ./utils/get_ids.py --done_dir ${done_dir_f} --n_rna ${n_rna} --batch_size ${PLAY_BATCH_SIZE} --save_dir ${id_batch_dir} --forbid_id 22 --set_init_ids
    fi

    for id_batch_str in `cat ${id_batch_dir}`
    do
        buffer_cnt=`expr $buffer_cnt + 1`
        epo_show=`expr $epo + 1`
        echo "------------------ Epoch ${epo_show} ------------------"
        echo $id_batch_str
        if [ "$epi" -eq 1 ] # && [ "$epo" -eq 0 ]
        then
            backbone_dir="${LOCAL_DIR}/pre_train/logs/2023_08_09_10_32_32/module/backbone_36.pth"
            # backbone_dir=none
        else
            backbone_dir=none
        fi
        echo "Load backbone from ${backbone_dir}"

        if [ "$epi" -gt 1 ] && [ "$epo" -eq 0 ]
        then
            last_epi=`expr $epi - 1`
            load_model_dir="${model_folder_root}/${last_epi}_${last_epo}/"
        else
            load_model_dir="${model_folder_root}/${epi}_${epo}/"
        fi

        # if [ "$epi" -eq 1 ] && [ "$epo" -eq 0 ]
        # then
        #     load_model_dir="/amax/data/liyichong/RNA_Split_2_data/2023_10_16_17_08_34_V2/models/335_1/"
        # fi

        echo "Load model from ${load_model_dir}"

        epo=`expr $epo + 1`
        save_model_dir="${model_folder_root}/${epi}_${epo}/"
        echo "Save model to ${save_model_dir}"

        if [ "$epi_f" -eq 0 ]
        then
            echo "Renew best solutions."
            python ${LOCAL_DIR}/train_epi.py --dotB_dir ${DATA_DIR} --best_dir ${best_dir} --id_batch_str ${id_batch_str} --done_dir ${done_dir_f} --loss_dir ${loss_dir} --reward_dir ${reward_dir} --backbone_dir ${backbone_dir} --agent_dir ${load_model_dir} --model_save_dir ${save_model_dir} --episode ${epi} --epoch ${epo} --step ${STEP} --gpu_order ${GPU} --n_traj ${N_TRAJ} --n_step ${N_STEP} --final_traj ${F_TRAJ} --final_step ${F_STEP} --slack_threshold ${SLACK_THRESHOLD} --fold_version ${RNAfold_Version} --use_freeze --buffer_dir ${buffer_dir} --buffer_cnt ${buffer_cnt} --n_buffer_load ${N_BUFFER_LOAD} --use_task_pool #--init_seed ${init_seed} #--use_mp  
        else
            echo "Load best solutions."
            python ${LOCAL_DIR}/train_epi.py --dotB_dir ${DATA_DIR} --best_dir ${best_dir} --id_batch_str ${id_batch_str} --done_dir ${done_dir_f} --loss_dir ${loss_dir} --reward_dir ${reward_dir} --backbone_dir ${backbone_dir} --agent_dir ${load_model_dir} --model_save_dir ${save_model_dir} --episode ${epi} --epoch ${epo} --step ${STEP} --load_best --gpu_order ${GPU} --n_traj ${N_TRAJ} --n_step ${N_STEP} --final_traj ${F_TRAJ} --final_step ${F_STEP} --slack_threshold ${SLACK_THRESHOLD} --fold_version ${RNAfold_Version} --use_freeze --buffer_dir ${buffer_dir} --buffer_cnt ${buffer_cnt} --n_buffer_load ${N_BUFFER_LOAD} --use_task_pool #--use_mp
        fi
        last_epo=$epo
    done
done