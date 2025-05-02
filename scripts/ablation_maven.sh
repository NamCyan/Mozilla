export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
set -e

method=NashMTL
clusters=4
gen_rate=20
naloss_ratio=4
fairgrad_alpha=0.8
robust_step_size=0.001
lm_temp=2
class_alpha=0.2
#cp -r  $BASH_SOURCE $log_p/run.sh

for i in 0 1 2 3 4
do  
    log_p=results/maven/${method}_ot
    save_model=checkpoints/maven/${method}
    if [ ! -d  "${log_p}/perm_${i}" ]; 
    then
        mkdir -p ${log_p}/perm_${i}
    fi

    CUDA_VISIBLE_DEVICES=0 python run_train.py \
    --ot --sam 1 --lmh --lm_temp ${lm_temp} --class_alpha ${class_alpha} --robust_step_size ${robust_step_size}\
    --log-dir "${log_p}/perm_${i}" --feature-root "data/features" --stream-file "data/MAVEN/streams.json" \
    --mul_task_type ${method} --fairgrad_alpha ${fairgrad_alpha} \
    --perm-id ${i} --dropout "normal" --p 0.2 \
    --mul_distill --mul_task --kt --kt2 \
    --train-epoch 5 --patience 5  \
    --naloss_ratio ${naloss_ratio} \
    --generate --generate_ratio ${gen_rate} \
    --batch-size 128 --learning-rate 1e-4 \
    --mode herding --clusters ${clusters} --num_sam_loss 2 > ${log_p}/perm_${i}/exp.log

done


# --lmh
# --generate --generate_ratio 10 
# --loss_trick