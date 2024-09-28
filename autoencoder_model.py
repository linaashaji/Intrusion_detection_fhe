import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, latent_size, num_lstm_layers):
        super(Encoder, self).__init__()

        self.encoder_lstm = nn.LSTM(
            input_size, hidden_size, num_lstm_layers, batch_first=True
        )
        self.latent = nn.Linear(hidden_size, latent_size)

    def forward(self, x):
        lstm_out, (h_n, c_n) = self.encoder_lstm(x)
        h_last = lstm_out[:, -1, :]
        latent = self.latent(h_last)

        return latent

    def encode(self, x):
        lstm_out, _ = self.encoder_lstm(x)
        h_last = lstm_out[:, -1, :]
        latent = self.latent(h_last)
        return latent


class Decoder(nn.Module):
    def __init__(self, input_size, latent_size, sequence_length):
        super(Decoder, self).__init__()

        self.sequence_length = sequence_length

        self.decoder_mlp = nn.Sequential(
            nn.Linear(latent_size, 128),
            nn.ReLU(),
            nn.Linear(128, input_size * sequence_length),
        )

    def forward(self, x):

        decoded = self.decoder_mlp(x)
        return decoded


class Autoencoder(nn.Module):
    def __init__(
        self, input_size, hidden_size, latent_size, sequence_length, num_lstm_layers=1
    ):
        super(Autoencoder, self).__init__()

        self.sequence_length = sequence_length
        self.hidden_size = hidden_size

        self.encoder = Encoder(input_size, hidden_size, latent_size, num_lstm_layers)
        self.decoder = Decoder(input_size, latent_size, sequence_length)

    def forward(self, x):
        latent = self.encoder(x)
        decoded = self.decoder(latent)
        decoded = decoded.view(-1, self.sequence_length, x.size(2))

        return decoded
