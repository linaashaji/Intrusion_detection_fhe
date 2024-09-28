from concrete.ml.deployment import FHEModelDev
from autoencoder_model import Encoder, Decoder, Autoencoder
from concrete.ml.torch.compile import compile_torch_model
import torch
import torch.nn as nn

sequence_length = 50
input_size = 12
latent_size = 8
hidden_size = 64

ae_model = Autoencoder(
    input_size=input_size,
    hidden_size=hidden_size,
    latent_size=latent_size,
    sequence_length=sequence_length,
    num_lstm_layers=1,
)

encoder = ae_model.encoder
encoder.load_state_dict(torch.load("save_models/encoder.pth", weights_only=True))

decoder = ae_model.decoder
decoder.load_state_dict(torch.load("save_models/decoder.pth", weights_only=True))


dummy_input = torch.randn(1, latent_size)
compiled_decoder = compile_torch_model(
    decoder,
    dummy_input.numpy(),
    n_bits=6,
    rounding_threshold_bits={"n_bits": 6, "method": "approximate"},
)


dev = FHEModelDev(path_dir="save_models/compiled_decoder", model=compiled_decoder)
dev.save()
