import torch
import torch.nn as nn


class DeepRNN(nn.Module):
    def __init__(
        self,
        seq_feature_dim,
        hidden_dim,
        linear_dim,
        n_outputs,
        batch_first=True,
        n_lstms=0,
        dropout_p=0,
    ):
        super(DeepRNN, self).__init__()
        self.batch_first = batch_first
        self.hidden_dim = hidden_dim
        self.dropout_p = dropout_p
        self.linear_dim = linear_dim

        self.upsize_linear = nn.Linear(seq_feature_dim, self.hidden_dim)
        self.bilstm = nn.LSTM(
            self.hidden_dim,
            self.hidden_dim,
            batch_first=self.batch_first,
            bidirectional=True,
        )
        self.n_lstms = n_lstms

        self.bilstm = nn.LSTM(
            self.hidden_dim,
            self.hidden_dim,
            batch_first=self.batch_first,
            bidirectional=True,
        )

        # self.bilstm_dropout = nn.Dropout(self.dropout_p)

        if n_lstms > 0:
            self.lstms = nn.ModuleList(
                [
                    nn.LSTM(
                        self.hidden_dim,
                        self.hidden_dim,
                        batch_first=self.batch_first,
                        bidirectional=False,
                    )
                    for _ in range(self.n_lstms)
                ]
            )

            self.lstm_dropouts = nn.ModuleList(
                [nn.Dropout(self.dropout_p) for _ in range(self.n_lstms)]
            )

        self.linear = nn.Linear(self.hidden_dim, self.linear_dim)
        self.linear_dropout = nn.Dropout(self.dropout_p)

        self.output_layer = nn.Linear(self.linear_dim, n_outputs)

    def forward(self, x, non_padding_mask):
        seq_lens = torch.sum(non_padding_mask, dim=1)

        x = x[:, : max(seq_lens), :]
        x = self.upsize_linear(x)
        packed_seq = nn.utils.rnn.pack_padded_sequence(
            x, seq_lens.cpu(), batch_first=self.batch_first, enforce_sorted=False
        )

        packed_bilstm_output = self.bilstm(packed_seq)

        padded_bilstm_output = nn.utils.rnn.pad_packed_sequence(
            packed_bilstm_output[0], batch_first=self.batch_first
        )

        lstm_output = (
            padded_bilstm_output[0][:, :, : self.hidden_dim]
            + padded_bilstm_output[0][:, :, self.hidden_dim :]
        )

        if self.n_lstms > 0:
            for idx, lstm in enumerate(self.lstms):
                x = lstm_output + x
                x = self.lstm_dropouts[idx](x)
                packed_seq = nn.utils.rnn.pack_padded_sequence(
                    x,
                    seq_lens.cpu(),
                    batch_first=self.batch_first,
                    enforce_sorted=False,
                )

                packed_lstm_output = lstm(packed_seq)

                lstm_output = nn.utils.rnn.pad_packed_sequence(
                    packed_lstm_output[0], batch_first=self.batch_first
                )[0]

        mean_bisltm_output = lstm_output.sum(dim=1) / seq_lens.unsqueeze(-1)

        linear_output = self.linear(mean_bisltm_output)
        linear_output = self.linear_dropout(linear_output)
        pred = self.output_layer(linear_output)

        return pred
