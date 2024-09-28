from autoencoder_model import Encoder, Decoder, Autoencoder
from torch.utils.data import DataLoader
from datasets import load_dataset
import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np
from concrete.ml.torch.compile import compile_torch_model
from concrete.ml.deployment import FHEModelClient
import time


def process_features(data):
    hex_features = data["features"]
    int_values = np.array([int(h[0], 16) for h in hex_features])
    binary_rep = np.array([list(bin(x)[2:].zfill(12)) for x in int_values])
    data["features"] = binary_rep.astype(float)
    return data


dataset_name = "micpst/can"
split = "train"

full_dataset = load_dataset(dataset_name)
train_val_split = full_dataset["train"].train_test_split(test_size=0.2)
val_dataset = train_val_split["test"]

val_dataset = val_dataset.map(lambda raw_data: process_features(raw_data))
val_dataset = val_dataset.with_format("torch")


batch_size = 1
val_dataloader = DataLoader(val_dataset, batch_size=batch_size)


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

start = time.perf_counter()
dummy_input = torch.randn(1, latent_size)
compiled_decoder = compile_torch_model(
    decoder,
    dummy_input.numpy(),
    n_bits=6,
    rounding_threshold_bits={"n_bits": 6, "method": "approximate"},
)
end = time.perf_counter()
print(f"Compilation time: {end - start:.4f} seconds")


criterion = nn.MSELoss()


encoder.eval()
decoder.eval()
with torch.no_grad():  # Disable gradient computation for validation
    val_loss = 0.0
    for i, data in tqdm(enumerate(val_dataloader)):
        # Client
        features = data["features"]
        # Client
        latent = encoder(features)

        # Server
        decrypted_output = compiled_decoder.forward(latent.numpy(), fhe="simulate")

        # Client
        decrypted_output = torch.tensor(decrypted_output).view(
            -1, ae_model.sequence_length, features.size(2)
        )

        loss = criterion(decrypted_output, features)
        val_loss += loss.item()

        if i > 10:
            break

    avg_val_loss = val_loss / 10

    print(f"Val Loss after Encryption: {avg_val_loss:.4f}")
