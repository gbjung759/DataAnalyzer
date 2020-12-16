import torch


class DenseAutoEncoder(torch.nn.Module):
    def __init__(self, batch_size, input_dim, seq_len, embedding_dim):
        super(DenseAutoEncoder, self).__init__()
        self.input_dim = input_dim
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.embedding_dim = embedding_dim
        # encoder
        self.hiddens = [1024, 512]
        self.enc1 = torch.nn.Linear(in_features=self.input_dim, out_features=self.hiddens[0])
        self.enc2 = torch.nn.Linear(in_features=self.hiddens[0], out_features=self.hiddens[1])
        self.enc3 = torch.nn.Linear(in_features=self.hiddens[1], out_features=self.embedding_dim)
        # decoder
        self.dec1 = torch.nn.Linear(in_features=self.embedding_dim, out_features=self.hiddens[1])
        self.dec2 = torch.nn.Linear(in_features=self.hiddens[1], out_features=self.hiddens[0])
        self.dec3 = torch.nn.Linear(in_features=self.hiddens[0], out_features=self.input_dim)
        self.dropout = torch.nn.Dropout(0.5)

    def forward(self, x):
        x = torch.nn.functional.relu(self.enc1(x))
        x = torch.nn.functional.relu(self.enc2(x))
        x = torch.nn.functional.relu(self.enc3(x))
        x = torch.nn.functional.relu(self.dec1(x))
        x = torch.nn.functional.relu(self.dec2(x))
        x = torch.nn.functional.relu(self.dec3(x))
        return x


