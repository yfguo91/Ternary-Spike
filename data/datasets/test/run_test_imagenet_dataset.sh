LD_LIBRARY_PATH=/mnt/lustre/zhouyucong/tmp/Prototype/third_party/lib64:/mnt/lustre/zhouyucong/tmp/Prototype/third_party/lib64/ceph:$LD_LIBRARY_PATH \
srun --mpi=pmi2 -p $1 -n$2 --gres=gpu:8 --ntasks-per-node=8 --cpus-per-task=5 \
 python test_imagenet_dataset.py \
 --read-from petrel
