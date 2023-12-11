import time
import sys
import argparse
import linklink as link
sys.path.append('../../..')
from easydict import EasyDict  # noqa: E402

from data.datasets.imagenet_dataset import ImageNetDataset  # noqa: E402
from utils.dist import dist_init  # noqa: E402

parser = argparse.ArgumentParser(description='test imagenet dataset')
parser.add_argument('--read-from', type=str, default='mc')
config = parser.parse_args()
config = EasyDict(vars(config))

if config.read_from == 'ceph' or config.read_from == 'petrel':
    config.train_root = 's3://zhouyucong.imagenet/train'
    config.train_meta = '/mnt/lustre/share/images/meta/train.txt'
else:
    config.train_root = '/mnt/lustre/share/images/train'
    config.train_meta = '/mnt/lustre/share/images/meta/train.txt'

rank, _ = dist_init()

dataset = ImageNetDataset(root_dir=config.train_root, meta_file=config.train_meta, read_from=config.read_from)

print('num: {}'.format(len(dataset)))

beg = time.time()
it = 0
for img, label in dataset:
    if rank == 0 and it % 100 == 0:
        end = time.time()
        print('########## iter {}, avg time: {:.3f}s ###########'.format(it, (end-beg)/100), flush=True)
        beg = time.time()
    it += 1

link.finalize()
