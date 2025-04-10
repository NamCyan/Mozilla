# currently kmean get distribution
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
set -e

method=ExcessMTL
robust_step_size=0.0001
# log_p=results/ace/${method}
# save_model=checkpoints/ace/${method}
clusters=8
gen_rate=10
lm_temp=2
class_alpha=0.2
naloss_ratio=40


for i in 0 1 2 3 4
do

    log_p=results/ace/${method}_woLMH_woSR_woSAM_woLosstrick_narate-${naloss_ratio}_cluster-${clusters}_gen-${gen_rate}_robust_step_size-${robust_step_size}
    # log_p=results/ace/woLMH_woMOO_woNAloss_woSR
    save_model=checkpoints/ace/${method}_kl

    if [ ! -d  "${log_p}/perm_${i}" ];
    then
        mkdir -p ${log_p}/perm_${i}
    fi

    CUDA_VISIBLE_DEVICES=0 python run_train.py --mul_task_type ${method} \
    --sam 0 --lm_temp ${lm_temp} --class_alpha ${class_alpha} --robust_step_size ${robust_step_size}\
    --feature-root "data/ace_features" --stream-file "data/ACE/streams.json" \
    --datasetname ACE --log-dir "${log_p}/perm_${i}" \
    --load-model ${save_model} --perm-id ${i} --dropout "normal" --p 0.2 \
    --mul_distill --mul_task --kt --kt2 \
    --train-epoch 30 --patience 6 --init-slots 13  \
    --naloss_ratio ${naloss_ratio} --max-slot 34 --batch-size 256 --learning-rate 1e-4 \
    --mode herding --clusters ${clusters} --num_sam_loss 2 > ${log_p}/perm_${i}/exp.log \

done

# --lmh
# --generate --generate_ratio 10 
# --loss_trick