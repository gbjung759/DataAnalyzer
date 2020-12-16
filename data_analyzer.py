import os
import numpy as np
import random
import torch
import sys
import pickle
from tqdm import trange
from binary_dataset import BinaryDataset
from stacked_conv_auto_encoder import StackedConvAutoEncoder
from dense_auto_encoder import DenseAutoEncoder
from torch.utils.data.dataloader import DataLoader


class DataAnalyzer:
    def __init__(self, datapath, optimizer, epochs, loss_function,
                 learning_rate, batch_size, early_stop_patience, input_dim, seq_len, embedding_dim):
        self.optimizer = optimizer.lower()
        assert type(self.optimizer) is str, 'optimizer_name의 type은 string이 되어야 합니다.'
        self.loss_function = loss_function
        assert type(loss_function) is str, 'loss_ft type은 string이 되어야 합니다.'
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.early_stop_patience = early_stop_patience
        self.input_dim = input_dim
        self.seq_len = seq_len
        self.embedding_dim = embedding_dim
        self.dataloader = DataLoader(
            BinaryDataset(path=datapath, input_dim=self.input_dim, seq_len=seq_len),
            batch_size=batch_size,
            shuffle=True,
            # drop_last=True,
            num_workers=2
        )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.auto_encoder = DenseAutoEncoder(batch_size=self.batch_size,
                                                   input_dim=self.input_dim,
                                                   seq_len=self.seq_len,
                                                   embedding_dim=embedding_dim).to(self.device)

        self.optimizer = self.get_optimizer(self.auto_encoder.parameters(), self.optimizer, self.learning_rate)
        self.loss_function = self.get_loss_function(self.loss_function)

    @staticmethod
    def set_seed(seed):
        """
        for reproducibility
        """
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def get_optimizer(self, parameters, optimizer_name, learning_rate):
        if optimizer_name == 'adam':
            return torch.optim.Adam(parameters, lr=learning_rate, weight_decay=1e-5)
        elif optimizer_name == 'adamw':
            return torch.optim.AdamW(parameters, lr=learning_rate, weight_decay=1e-5)
        elif optimizer_name == 'rmsprop':
            return torch.optim.RMSprop(parameters, lr=learning_rate)
        elif optimizer_name == 'sgd':
            return torch.optim.SGD(parameters, lr=learning_rate)
        else:
            raise ValueError('optimizer_name이 pytorch에 존재하지 않습니다. 다시 확인하세요.')

    def get_loss_function(self, loss_ft):
        if loss_ft == 'mseloss':
            return torch.nn.MSELoss()
        elif loss_ft == 'crossentropyloss':
            return torch.nn.CrossEntropyLoss()
        elif loss_ft == 'nllloss':
            return torch.nn.NLLLoss()
        else:
            raise ValueError('loss_function이 pytorch에 존재하지 않습니다. 다시 확인하세요.')

    def train(self):
        self.set_seed(42)
        train_iterator = trange(self.epochs, desc="Epoch")

        print("\n***** Running training *****")
        print("  Num Epochs = {}".format(self.epochs))
        print("  Train Batch size = {}".format(self.batch_size))
        print("  Device = ", self.device)

        min_loss = sys.maxsize
        anger = 0
        for epoch in train_iterator:
            loss_in_epoch = 0.0
            for j, [binary, _] in enumerate(self.dataloader):
                binary = binary.to(self.device)
                output = self.auto_encoder(binary)
                criterion = self.loss_function(output, binary)

                criterion.backward()
                self.optimizer.zero_grad()
                self.optimizer.step()
                loss_in_epoch += criterion.item()
            if loss_in_epoch < min_loss:
                min_loss = loss_in_epoch
                anger = 0
                torch.save(self.auto_encoder, './model/stacked_conv_autoencoder.pkl')
            else:
                anger += 1
            if anger > self.early_stop_patience:
                break
            if (epoch + 1) % 1 == 0:
                print("  Epoch / Total Epoch : {} / {}".format(epoch + 1, self.epochs))
                print("  Loss : {:.4f}".format(loss_in_epoch))

    def load_model(self, path):
        self.auto_encoder = torch.load(path)

    def reconstruct(self, path):
        print("***** Running Reconstruction *****")
        labels = None
        reconstructed = None
        rbinary_list = []
        with torch.no_grad():
            for [binary, label] in self.dataloader:
                binary = binary.to(self.device)
                output = self.auto_encoder(binary)
                if reconstructed is None:
                    reconstructed = output.detach().cpu().numpy()
                    labels = label
                else:
                    reconstructed = np.append(reconstructed, output.detach().cpu().numpy(), axis=0)
                    labels.extend(label)

        for rbinary in reconstructed:
            rbinary_list.append(list(rbinary.flatten()))

        for i, rbinary in enumerate(rbinary_list):
            with open(os.path.join(path, labels[i]), 'wb') as f:
                pickle.dump(rbinary, f)
