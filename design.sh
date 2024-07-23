MAX_EPISODE=2
MAX_ROUND=1
LOCAL_DIR=$(cd $(dirname $0); pwd)
# echo "Local ${LOCAL_DIR}"
DATA_DIR="${LOCAL_DIR}/data/eterna_v2/"
# echo "Puzzles at ${DATA_DIR}"
LOG_DIR_ROOT="${LOCAL_DIR}/Design_Log/"
# echo "Log at ${LOG_DIR_ROOT}"
LOAD_MODEL_DIR="${LOCAL_DIR}/model_param/1001_1/"
# echo "Model parameters at ${LOAD_MODEL_DIR}"
PLAY_BATCH_SIZE=100
STEP=100
N_TRAJ=0
N_STEP=0
F_TRAJ=0
F_STEP=0
SLACK_THRESHOLD=450
GPU=2
RWNEW_FRE=4
n_gen=10
# RNAfold_Version=2

echo "Enter the target strcucture:"
read dotB
echo "Your target structure: $dotB"

echo "Enter the version of RNAfold(1 or 2):"
read RNAfold_Version
echo "Chose RNAfold $RNAfold_version"

time_now=`date +%Y_%m_%d_%H_%M_%S`
epi=0
cd ${DATA_DIR}
n_rna=`ls -l |grep "^-"|wc -l`
cd ${LOCAL_DIR}
echo "Design for ${n_rna} structures of RNA."
log_dir="${LOG_DIR_ROOT}design_${time_now}/"
mkdir ${log_dir}
echo "Log in ${log_dir}"
done_dir="${log_dir}done_log.csv"
touch ${done_dir}
echo "Save solutions in ${done_dir}"
best_dir="${log_dir}best_design/"
mkdir ${best_dir}
echo "Best designs of RNAs in ${best_dir}"

init_seed=0
exit_code=0

echo "++++++++++++++++++ Design ++++++++++++++++++"
while [ $exit_code -eq 0 ]
do
    epi_f=`expr $epi % 20`
    init_seed=`expr $init_seed % 9`
    epi=`expr $epi + 1`
    echo "================== Episode ${epi} =================="
    
    # id_batch_dir="${log_dir}id_batch_${epi}_${epo}.txt"
    # echo "Save id batchs in ${id_batch_dir}"
    # python ./utils/get_ids.py --done_dir ${done_dir} --n_rna ${n_rna} --batch_size ${PLAY_BATCH_SIZE} --save_dir ${id_batch_dir} --forbid_id 22 --set_init_ids

    # for id_batch_str in `cat ${id_batch_dir}`
    # do
    #     epo_show=`expr $epo + 1`
    #     echo "------------------ Epoch ${epo_show} ------------------"
    #     echo $id_batch_str

    echo "Load model from ${LOAD_MODEL_DIR}"

    if [ "$epi_f" -eq 0 ]
    then
        echo $epi_f
        python ${LOCAL_DIR}/design_epi_single.py --structure ${dotB} --best_dir ${best_dir} --done_dir ${done_dir} --agent_dir ${LOAD_MODEL_DIR} --episode ${epi} --step ${STEP} --gpu_order ${GPU} --n_traj ${N_TRAJ} --n_step ${N_STEP} --final_traj ${F_TRAJ} --final_step ${F_STEP} --use_task_pool --slack_threshold ${SLACK_THRESHOLD} --fold_version ${RNAfold_Version} --use_freeze --init_seed ${init_seed} #--use_mp
        exit_code=$? 
    else
        echo "Renew solutions."
        python ${LOCAL_DIR}/design_epi_single.py --structure ${dotB} --best_dir ${best_dir} --done_dir ${done_dir} --agent_dir ${LOAD_MODEL_DIR} --episode ${epi} --step ${STEP} --load_best --gpu_order ${GPU} --n_traj ${N_TRAJ} --n_step ${N_STEP} --final_traj ${F_TRAJ} --final_step ${F_STEP} --use_task_pool --slack_threshold ${SLACK_THRESHOLD} --fold_version ${RNAfold_Version} --use_freeze #--use_mp 
        exit_code=$?
    fi

    # if [ $exit_code -eq 1 ]
    # then
    #     break
    # fi 
done