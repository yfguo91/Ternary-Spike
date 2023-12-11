import os.path as osp
import io
from PIL import Image

from .base_dataset import BaseDataset


def pil_loader(img_bytes, filepath):
    # warnings.filterwarnings("ignore", "(Possibly )?corrupt EXIF data", UserWarning)
    buff = io.BytesIO(img_bytes)
    try:
        with Image.open(buff) as img:
            img = img.convert('RGB')
    except IOError:
        print('Failed in loading {}'.format(filepath))
    return img


class ImageNetDataset(BaseDataset):
    def __init__(self, root_dir, meta_file, transform=None, read_from='mc'):
        self.root_dir = root_dir
        self.read_from = read_from
        self.transform = transform

        with open(meta_file) as f:
            lines = f.readlines()

        self.num = len(lines)
        self.metas = []
        for line in lines:
            path, cls = line.rstrip().split()
            self.metas.append((path, int(cls)))

        super(ImageNetDataset, self).__init__(read_from=read_from)

    def __len__(self):
        return self.num

    def __getitem__(self, idx):
        filepath = osp.join(self.root_dir, self.metas[idx][0])
        cls = self.metas[idx][1]
        img_bytes = self.read_file(filepath)

        if self.transform is None:
            return img_bytes, cls

        img = pil_loader(img_bytes, filepath)
        img = self.transform(img)

        return img, cls
