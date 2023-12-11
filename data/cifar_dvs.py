import random
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import RandomResizedCrop, Resize
import torchvision
import torch.nn.functional as F
import torch
import os


class DVSCifar10(Dataset):
    def __init__(self, root, train=True, transform=None, target_transform=None):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.train = train
        self.resize = Resize(size=(48, 48), interpolation=torchvision.transforms.InterpolationMode.NEAREST)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        data, target = torch.load(self.root + '/{}.pt'.format(index))
        if self.train:
            data = self.resize(data.permute([3, 0, 1, 2]))
        if self.transform is not None:
            flip = random.random() > 0.5
            if flip:
                data = torch.flip(data, dims=(3,))
            off1 = random.randint(-5, 5)
            off2 = random.randint(-5, 5)
            data = torch.roll(data, shifts=(off1, off2), dims=(2, 3))

        if self.target_transform is not None:
            target = self.target_transform(target)

        return data, target.long()

    def __len__(self):
        return len(os.listdir(self.root))


# data, target = torch.load('/Users/liyuhang/Documents/GitHub/DFSNN/cifar-dvs/0.pt')
#
# data = data.permute([3, 0, 1, 2])
# data = F.interpolate(data, size=(42, 42), )
# print(data)

# import numpy as np
# a = np.array([74.1, 74.1, 74.0])
# print(a.mean(), a.std())