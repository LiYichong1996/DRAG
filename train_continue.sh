MAX_EPISODE=99
LOCAL_DIR="/home/liyichong/RNA_Split_2/"
DATA_DIR="/home/liyichong/RNA_Split_2/data/raw/rfam_learn/train/"
LOG_DIR_ROOT="/amax/data/liyichong/RNA_Split_2_data//"
PLAY_BATCH_SIZE=1000
STEP=40
N_TRAJ=0
N_STEP=0
F_TRAJ=0
F_STEP=0
GPU=2

# time_now=`date +%Y_%m_%d_%H_%M_%S`
time_now='2023_04_14_15_35_18'
epi=5
cd ${DATA_DIR}
n_rna=`ls -l |grep "^-"|wc -l`
cd ${LOCAL_DIR}
echo "Train on ${n_rna} structures of RNA."
log_dir="${LOG_DIR_ROOT}${time_now}/"
mkdir ${log_dir}
echo "Log in ${log_dir}"
model_folder_root="${log_dir}models"
mkdir ${model_folder_root}
echo "Save models in ${model_folder_root}"
done_dir="${log_dir}done_log.csv"
# touch ${done_dir}
echo "Save solutions in ${done_dir}"
best_dir="${log_dir}best_design/"
mkdir ${best_dir}
echo "Best designs of RNAs in ${best_dir}"
loss_dir="${log_dir}loss.txt"
# touch ${loss_dir}
echo "Losses of training in ${loss_dir}"
reward_dir="${log_dir}reward.txt"
# touch ${log_dir}
echo "Rewards of training in ${reward_dir}"
last_epo=46

echo "++++++++++++++++++ Train ++++++++++++++++++"
while [ "$epi" -le "$MAX_EPISODE" ]
do
    epi_f=`expr $epi % 4`
    epi=`expr $epi + 1`
    echo "================== Episode ${epi} =================="
    epo=0
    id_batch_dir="${log_dir}id_batch_${epi}.txt"
    echo "Save id batchs in ${id_batch_dir}"
    python ./utils/get_ids.py --done_dir ${done_dir} --n_rna ${n_rna} --batch_size ${PLAY_BATCH_SIZE} --save_dir ${id_batch_dir}

    for id_batch_str in `cat ${id_batch_dir}`
    do
        epo_show=`expr $epo + 1`
        echo "------------------ Epoch ${epo_show} ------------------"
        # echo $id_batch_str
        if [ "$epi" -eq 1 ] && [ "$epo" -eq 0 ]
        then
            backbone_dir="${LOCAL_DIR}/pre_train/logs/PPO_logs_2023_03_25_23_56_20/module/backbone_40.pth"
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
        echo "Load model from ${load_model_dir}"

        epo=`expr $epo + 1`
        save_model_dir="${model_folder_root}/${epi}_${epo}/"
        echo "Save model to ${save_model_dir}"

        if [ "$epi_f" -eq 0 ]
        then
            python ${LOCAL_DIR}/train_epi.py --dotB_dir ${DATA_DIR} --best_dir ${best_dir} --id_batch_str ${id_batch_str} --done_dir ${done_dir} --loss_dir ${loss_dir} --reward_dir ${reward_dir} --backbone_dir ${backbone_dir} --agent_dir ${load_model_dir} --model_save_dir ${save_model_dir} --episode ${epi} --epoch ${epo} --step ${STEP} --gpu_order ${GPU} --n_traj ${N_TRAJ} --n_step ${N_STEP} --final_traj ${F_TRAJ} --final_step ${F_STEP} #--buffer_dir ${buffer_dir} --buffer_cnt ${buffer_cnt} --n_buffer_load ${N_BUFFER_LOAD}
        else
            python ${LOCAL_DIR}/train_epi.py --dotB_dir ${DATA_DIR} --best_dir ${best_dir} --id_batch_str ${id_batch_str} --done_dir ${done_dir} --loss_dir ${loss_dir} --reward_dir ${reward_dir} --backbone_dir ${backbone_dir} --agent_dir ${load_model_dir} --model_save_dir ${save_model_dir} --episode ${epi} --epoch ${epo} --step ${STEP} --load_best --gpu_order ${GPU} --n_traj ${N_TRAJ} --n_step ${N_STEP} --final_traj ${F_TRAJ} --final_step ${F_STEP}  #--buffer_dir ${buffer_dir} --buffer_cnt ${buffer_cnt} --n_buffer_load ${N_BUFFER_LOAD}
        fi

        last_epo=$epo
    done
done