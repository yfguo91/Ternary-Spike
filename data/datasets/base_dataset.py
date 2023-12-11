import mc
from torch.utils.data import Dataset
import numpy as np
# import ceph
# from petrel_client.client import Client


class BaseDataset(Dataset):
    def __init__(self, read_from='mc'):
        self.read_from = read_from
        self.initialized = False

    def _init_memcached(self):
        if not self.initialized:
            server_list_config_file = "/mnt/lustre/share/memcached_client/server_list.conf"
            client_config_file = "/mnt/lustre/share/memcached_client/client.conf"
            self.mclient = mc.MemcachedClient.GetInstance(server_list_config_file, client_config_file)
            self.initialized = True

    def _init_ceph(self):
        if not self.initialized:
            # self.s3_client = ceph.S3Client()
            self.initialized = True

    def _init_petrel(self):
        if not self.initialized:
            # self.client = Client(enable_mc=True)
            self.initialized = True

    def read_file(self, filepath):
        if self.read_from == 'fake':
            if self.initialized:
                filebytes = self.saved_filebytes
            else:
                filebytes = self.saved_filebytes = np.fromfile(filepath, dtype=np.uint8)
                self.initialized = True
        elif self.read_from == 'mc':
            self._init_memcached()
            value = mc.pyvector()
            self.mclient.Get(filepath, value)
            value_str = mc.ConvertBuffer(value)
            filebytes = np.frombuffer(value_str.tobytes(), dtype=np.uint8)
        elif self.read_from == 'ceph':
            self._init_ceph()
            value = self.s3_client.Get(filepath)
            filebytes = np.frombuffer(value, dtype=np.uint8)
        elif self.read_from == 'petrel':
            self._init_petrel()
            value = self.client.Get(filepath)
            filebytes = np.frombuffer(value, dtype=np.uint8)
        elif self.read_from == 'fs':
            filebytes = np.fromfile(filepath, dtype=np.uint8)
        else:
            raise RuntimeError("unknown value for read_from: {}".format(self.read_from))

        return filebytes
