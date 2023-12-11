import argparse
import os

#os.system('wandb login xxx')
#import wandb
from time import time
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
import torchvision
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data.distributed import DistributedSampler
from kdutils import seed_all, GradualWarmupScheduler
#from torchvision.models.resnet import resnet18
from torchvision import transforms
from models import *
#from models import ImageNet_cnn, ImageNet_snn
#from loss_kd import feature_loss,  logits_loss
#from spikingjelly.clock_driven import functional
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"

seed_all(1000)



def get_model(name):
    return func_dict[name]


parser = argparse.ArgumentParser(description="ImageNet_SNN_Training")
parser.add_argument("--local_rank", type=int, default=0)
parser.add_argument("--datapath", type=str, default='/home/xlab/xdata/imagenet/')
parser.add_argument("--model", type=str, default='18')
parser.add_argument("--tea_model", type=str, default='18')
parser.add_argument("--model_weight", type=str, default="resnet50_timm.pth")
parser.add_argument("--batch", type=int, default=40)
parser.add_argument("--epoch", type=int, default=320)
parser.add_argument("--warm_up", action='store_true', default=False)
parser.add_argument("--beta", type=float, default=10.)
parser.add_argument("--after_beta", type=float, default=0.01)
parser.add_argument("--load_weight", action='store_true', default=False)
parser.add_argument("--feature_epochs", type=int, default=10)
parser.add_argument('--spike', action='store_true', help='use spiking network')  
parser.add_argument('--lr', default=1e-3, type=float, help='initial learning rate')
parser.add_argument('--step', default=4, type=int, help='snn step')
parser.add_argument('--num_gpu', default=8, type=int, help='snn step')
parser.add_argument('--world-size', default=2, type=int,
                    help='number of distributed processes')
parser.add_argument('--dist-url', default='env://', help='url used to set up distributed training')
args = parser.parse_args()
torch.distributed.init_process_group(backend='nccl')
torch.cuda.set_device(args.local_rank)

best_acc = 0.3
sta = time()

# ----------------------------
if args.local_rank == 0:
    print("batch:{}_epoch:{}_lr:{}_".format(args.batch, args.epoch, args.lr))
    print("Model downloading")
    
######## input model #######
model = ResNet34(num_classes=1000)


######## save model #######
model_save_name = 'raw/snn-res18-4-320.pth'


######## load weight #######
#model.load_state_dict(torch.load('raw/ann-resnet18.pth', map_location='cpu'))

######## change to snn #######
if args.spike is True:
    model = SpikeModel(model, args.step)
    model.set_spike_state(True)
    
######## init bias #######    
#model = init_bias(model)    
    
SNN = model.cuda()

######## show parameters #######
n_parameters = sum(p.numel() for p in SNN.parameters() if p.requires_grad)
if args.local_rank == 0:
    print('number of params:', n_parameters)
    print(SNN)

######## parallel #######
SNN = torch.nn.SyncBatchNorm.convert_sync_batchnorm(SNN)
SNN = torch.nn.parallel.DistributedDataParallel(SNN, device_ids=[[args.local_rank]],output_device=[args.local_rank], find_unused_parameters=False)
model_without_ddp = SNN.module

######## amp #######
loss_fun = torch.nn.CrossEntropyLoss().cuda()
scaler = torch.cuda.amp.GradScaler()

######## split BN #######

parameters = split_weights(SNN)
optimer = torch.optim.SGD(params=parameters, lr=1e-1, momentum=0.9, weight_decay=1e-4)
#optimer = torch.optim.AdamW(params=parameters, lr=1e-3, betas=(0.9, 0.999), weight_decay=5e-3)
#optimer = torch.optim.AdamW(params=SNN.parameters(), lr=1e-3, betas=(0.9, 0.999), weight_decay=5e-3)

scheduler = CosineAnnealingLR(optimer, T_max=args.epoch, eta_min=0)
scheduler_warm = None
if args.warm_up:
    scheduler_warm = GradualWarmupScheduler(optimer, multiplier=1, total_epoch=5, after_scheduler=scheduler)
writer = None

# ------------------------------
if args.local_rank == 0:
    print("Datasets Download")

traindir = args.datapath + 'train'
valdir = args.datapath + 'val'

###data

train_dataset = torchvision.datasets.ImageFolder(
    traindir,
    transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ]))

test_dataset = torchvision.datasets.ImageFolder(
    valdir,
    transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ]))

samper_train = DistributedSampler(train_dataset)
train_data = DataLoader(train_dataset, batch_size=args.batch, sampler=samper_train, num_workers=args.num_gpu * 5,
                        pin_memory=True)
test_data = DataLoader(test_dataset, batch_size=args.batch, shuffle=False, num_workers=args.num_gpu * 5,
                       pin_memory=True)

if __name__ == '__main__':

    for i in range(args.epoch):
        loss_ce_all = 0
        start_time = time()
        right = 0
        SNN.train()
        samper_train.set_epoch(args.epoch)
        if args.local_rank == 0:
            print("epoch:{}".format(i))
        for step, (imgs, target) in enumerate(train_data):
            imgs, target = imgs.cuda(non_blocking=True), target.cuda(non_blocking=True)
            with torch.cuda.amp.autocast():
                output = SNN(imgs,is_drop=False)

                loss = loss_fun(output,target)

            right = (output.argmax(1) == target).sum() + right
            loss_ce_all += loss.item()


            optimer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimer)
            scaler.update()

            if step % 100 == 0 and args.local_rank == 0:
                print("step:{:.2f} loss_ce:{:.2f}".format(step / len(train_data), loss.item()))
        accuracy1 = right / (len(train_dataset)) * args.num_gpu
        if args.warm_up:
            scheduler_warm.step()
        else:
            scheduler.step()

        SNN.eval()
        right = 0

        with torch.no_grad():
            for (imgs, target) in test_data:
                imgs, target = imgs.cuda(non_blocking=True), target.cuda(non_blocking=True)
                output  = SNN(imgs,is_drop=False)
                right = (output.argmax(1) == target).sum() + right

            accuracy = right / len(test_dataset)
            end_time = time()
            if args.local_rank == 0:
                print("epoch:{} time:{:.0f}  loss:{:.4f} train_acc:{:.4f} tets_acc:{:.4f} eta:{:.2f}".format(i,end_time - start_time,loss_ce_all,accuracy1,accuracy, (end_time - start_time) * (args.epoch - i - 1) / 3600))
                if accuracy > best_acc:
                    best_acc = accuracy
                    print("best_acc:{:.4f}".format(best_acc))
                    torch.save(model_without_ddp.state_dict(), model_save_name)
                #print({"test_acc": accuracy, "train_acc": accuracy1, "loss_ce_all": loss_ce_all, 'epoch': i, })

    end = time()
    print(end - sta)
    print("best_acc:{:.4f}".format(best_acc))
    
#python3 -m torch.distributed.launch --nproc_per_node 8 --nnode 1 --master_port=25641 Train.py --spike --step 4
