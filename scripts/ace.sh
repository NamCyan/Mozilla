# currently kmean get distribution
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
set -e

method=FairGrad
log_p=results/ace/${method}
save_model=checkpoints/ace/${method}
# mkdir -p $log_p
#cp -r  $BASH_SOURCE $log_p/run.sh

for i in 2 4 
do
    if [ ! -d  "${log_p}/perm_${i}" ];
    then
        mkdir -p ${log_p}/perm_${i}
    fi

    # touch ${log_p}/perm_${i}/exp.log
    # cp -r models/nets.py ${log_p}/perm_${i}/
    # cp -r utils/datastream.py ${log_p}/perm_${i}/
    # cp -r utils/worker.py ${log_p}/perm_${i}/

    CUDA_VISIBLE_DEVICES=0 python run_train.py --mul_task_type ${method} --feature-root "data/ace_features" --stream-file "data/ACE/streams.json" --datasetname ACE --log-dir "${log_p}/perm_${i}" --load-model ${save_model} --perm-id ${i} --dropout "normal" --p 0.2 --mul_distill --kt --kt2 --train-epoch 30 --patience 6 --init-slots 13  --generate --generate_ratio 20 --naloss_ratio 40  --max-slot 34 --batch-size 256 --mode herding --clusters 4 --mul_task --num_sam_loss 2 --learning-rate 1e-4 > ${log_p}/perm_${i}/exp.log
done
