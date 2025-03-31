export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
set -e

method=FairGrad
clusters=4
gen_rate=20
naloss_ratio=4
fairgrad_alpha=2
lm_temp=0.2
class_alpha=0.2
#cp -r  $BASH_SOURCE $log_p/run.sh

for i in 0 1 2 3 4
do  
    log_p=results/maven/mi_woMOO_narate-${naloss_ratio}_cluster-${clusters}_gen-${gen_rate}_lm_temp-${lm_temp}_class_alpha-${class_alpha}
    save_model=checkpoints/maven/${method}
    if [ ! -d  "${log_p}/perm_${i}" ]; 
    then
        mkdir -p ${log_p}/perm_${i}
    fi

    CUDA_VISIBLE_DEVICES=1 python run_train.py \
    --mi --sam 0 --loss_trick --lmh --lm_temp ${lm_temp} --class_alpha ${class_alpha}\
    --log-dir "${log_p}/perm_${i}" --feature-root "data/features" --stream-file "data/MAVEN/streams.json" \
    --mul_task_type ${method} --fairgrad_alpha ${fairgrad_alpha} \
    --perm-id ${i} --dropout "normal" --p 0.2 \
    --mul_distill --kt --kt2 \
    --train-epoch 15 --patience 5  \
    --naloss_ratio ${naloss_ratio} \
    --batch-size 128 --learning-rate 1e-4 \
    --generate --generate_ratio ${gen_rate} \
    --mode herding --clusters ${clusters} --num_sam_loss 2 > ${log_p}/perm_${i}/exp.log

done


# --lmh
# --generate --generate_ratio 10 
# --loss_trick