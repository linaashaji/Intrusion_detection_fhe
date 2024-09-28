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
from sklearn.metrics import accuracy_score, f1_score


def process_features(data):
    hex_features = data["features"]
    int_values = np.array([int(h[0], 16) for h in hex_features])
    binary_rep = np.array([list(bin(x)[2:].zfill(12)) for x in int_values])
    data["features"] = binary_rep.astype(float)
    return data


dataset_name = "micpst/can"
split = "test"

full_dataset = load_dataset(dataset_name)
test_dataset = full_dataset[split]

test_dataset = test_dataset.map(lambda raw_data: process_features(raw_data))
test_dataset = test_dataset.with_format("torch")


batch_size = 1
val_dataloader = DataLoader(test_dataset, batch_size=batch_size)


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

criterion = nn.MSELoss()


y_pred, y_true = [], []
encoder.eval()
decoder.eval()
with torch.no_grad():  # Disable gradient computation for validation
    val_loss = 0.0
    for i, data in tqdm(enumerate(val_dataloader)):
        # Client
        features = data["features"]
        labels = data["label"]
        # Client
        latent = encoder(features)

        # Server
        output = decoder.forward(latent)

        # Client
        output = output.view(-1, ae_model.sequence_length, features.size(2))

        loss = criterion(output, features)
        val_loss += loss.item()

        pred = loss.item() > 0.05
        gt = labels.item() > 0

        y_pred.append(pred)
        y_true.append(gt)

        if i > 1000:
            break

    avg_val_loss = val_loss / i

    print(f"Test Loss: {avg_val_loss:.4f}")


accuracy = accuracy_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)

print(f"Accuracy: {accuracy}")
print(f"F1 Score: {f1}")
