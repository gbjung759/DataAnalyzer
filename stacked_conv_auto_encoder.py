import torch


class StackedConvAutoEncoder(torch.nn.Module):
    def __init__(self, batch_size, input_dim, seq_len, embedding_dim):
        super(StackedConvAutoEncoder, self).__init__()
        self.input_dim = input_dim
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.embedding_dim = embedding_dim

        self.encoder = torch.nn.Sequential(
            # input_dim/seq_len (128) -> seq_len (64) -> 128 (32) -> emd_dim
            torch.nn.Conv1d(int(self.input_dim/self.seq_len), self.seq_len, 8, stride=1),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(self.seq_len), # [batch, seq_len, filter]
            torch.nn.Conv1d(self.seq_len, 128, 2, stride=1),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(128),
            torch.nn.Conv1d(128, self.embedding_dim, 2, stride=1),
            torch.nn.ReLU()
        )
        self.decoder = torch.nn.Sequential(
            # emd_dim (32) -> 128 (64)-> seq_len(128) -> input_dim/seq_len
            torch.nn.ConvTranspose1d(self.embedding_dim, 128, 2, stride=1),  # batch x 128 x 14 x 14
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(128),
            torch.nn.ConvTranspose1d(128, self.seq_len, 4, stride=1),  # batch x 64 x 14 x 14
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(self.seq_len),
            torch.nn.ConvTranspose1d(self.seq_len, int(self.input_dim/self.seq_len), 8, stride=1),  # batch x 16 x 14 x 14
        )

    def forward(self, x):
        # input [batch, -1, seq_len] [1, 165, 16]
        x = self.encoder(x) # [batch, embedding_dim, 93]
        output = self.decoder(x) # [4, 10, 1024]
        return output
