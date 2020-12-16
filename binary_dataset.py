import os
import glob
import numpy as np
import struct
from torch.utils.data import Dataset


class BinaryDataset(Dataset):
    def __init__(self, path, input_dim, seq_len, binary_unit=1):
        self.files = glob.glob(os.path.join(path, '*'))
        self.input_dim = input_dim
        self.seq_len = seq_len
        self.binary_unit = binary_unit
        self.binary_list = []
        self.labels = []
        for i in range(len(self.files)):
            binary, label = self.processing(i)
            self.binary_list.append(binary)
            self.labels.append(label)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        return self.binary_list[idx], self.labels[idx]

    def conv2float(self, binary_list):
        res = 0.0
        for i in range(len(binary_list)):
            res += binary_list[i] * (255 ** i)
        return res

    def processing(self, idx):
        float_binary = []
        binary_list = []

        f = open(self.files[idx], "rb")
        try:
            byte = f.read(1)
            while byte:
                int_val = struct.unpack('B', byte)[0]
                int_val = float(int_val / 255)
                float_binary.append(float(int_val))
                byte = f.read(1)
        finally:
            f.close()
        float_binary = np.asarray(float_binary)
        if len(float_binary) > self.input_dim:
            float_binary = float_binary[:self.input_dim]
        else:
            temp = np.zeros(self.input_dim)
            temp[:len(float_binary)] = float_binary
            float_binary = temp
        float_binary = float_binary.astype(np.float32)
        return float_binary, os.path.basename(self.files[idx])

