# currently kmean get distribution
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
set -e

method=NashMTL
gen_rate=10
clusters=4
naloss_ratio=40
log_p=results/ace/${method}_mi_narate-40_cluster-${clusters}_gen-${gen_rate}
save_model=checkpoints/ace/${method}_mi
# mkdir -p $log_p
#cp -r  $BASH_SOURCE $log_p/run.sh

for i in 0 1 2 3 4
do
    if [ ! -d  "${log_p}/perm_${i}" ];
    then
        mkdir -p ${log_p}/perm_${i}
    fi

    # touch ${log_p}/perm_${i}/exp.log
    # cp -r models/nets.py ${log_p}/perm_${i}/
    # cp -r utils/datastream.py ${log_p}/perm_${i}/
    # cp -r utils/worker.py ${log_p}/perm_${i}/

    CUDA_VISIBLE_DEVICES=1 python run_train.py --mul_task_type ${method} \
    --mi --sam 1 --loss_trick --lmh \
    --feature-root "data/ace_features" --stream-file "data/ACE/streams.json" \
    --datasetname ACE --log-dir "${log_p}/perm_${i}" \
    --load-model ${save_model} --perm-id ${i} --dropout "normal" --p 0.2 \
    --mul_distill --mul_task --kt --kt2 \
    --train-epoch 30 --patience 6 --init-slots 13  \
    --generate --generate_ratio ${gen_rate} \
    --naloss_ratio 40  --max-slot 34 --batch-size 256 --learning-rate 1e-4 \
    --mode herding --clusters 4 --num_sam_loss 2 > ${log_p}/perm_${i}/exp.log
done
