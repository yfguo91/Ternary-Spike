import time
import sys
import argparse
from easydict import EasyDict
import linklink as link
sys.path.append('../..')


from data import make_imagenet_train_data  # noqa: E402
from utils.dist import dist_init  # noqa: E402

parser = argparse.ArgumentParser(description='test for imagenet train data')
parser.add_argument('--use-dali', action='store_true')
parser.add_argument('--read-from', type=str, default='mc')
parser.add_argument('--batch-size', type=int, default=64)
parser.add_argument('--dali-workers', type=int, default=4)
parser.add_argument('--workers', type=int, default=2)
parser.add_argument('--input-size', type=int, default=224)
parser.add_argument('--max-iter', type=int, default=5000)
parser.add_argument('--last-iter', type=int, default=-1)

config = parser.parse_args()
config = EasyDict(vars(config))

if config.read_from == 'ceph' or config.read_from == 'petrel':
    config.train_root = 's3://zhouyucong.imagenet/train'
    config.train_meta = '/mnt/lustre/share/images/meta/train.txt'
else:
    config.train_root = '/mnt/lustre/share/images/train'
    config.train_meta = '/mnt/lustre/share/images/meta/train.txt'

config.augmentation = {}
config.augmentation.colorjitter = None
config.augmentation.rotation = 0
config.pin_memory = True

rank, _ = dist_init()

train_data = make_imagenet_train_data(config)
train_loader = train_data['loader']

print('num: {}'.format(len(train_loader)))

it = 0
beg = time.time()
for img, label in train_loader:
    if rank == 0 and it % 10 == 0:
        end = time.time()
        print('########## iter {}, avg time: {:.3f}s ###########'.format(it, (end-beg)/10), flush=True)
        beg = time.time()
    it += 1

link.finalize()
