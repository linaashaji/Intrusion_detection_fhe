from autoencoder_model import Encoder, Decoder, Autoencoder
from torch.utils.data import DataLoader
from datasets import load_dataset
import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np
import torch.optim as optim


def process_features(data):
    hex_features = data["features"]
    int_values = np.array([int(h[0], 16) for h in hex_features])
    binary_rep = np.array([list(bin(x)[2:].zfill(12)) for x in int_values])
    data["features"] = binary_rep.astype(float)
    return data


# Training function
def train_autoencoder(
    model, train_loader, val_dataloader, optimizer, criterion, epochs=20
):
    model.train()  # Set the model to training mode
    best_loss = 1000
    for epoch in range(epochs):
        running_loss = 0.0
        for data in tqdm(train_loader):
            features = data["features"]

            # Zero the gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(features)

            # Compute the loss
            loss = criterion(outputs, features)  # Reconstruction loss
            loss.backward()  # Backpropagation

            # Optimize the weights
            optimizer.step()

            # Accumulate the loss for monitoring
            running_loss += loss.item()

        # Print the average loss per epoch
        avg_train_loss = running_loss / len(train_loader)

        # Validation loop
        model.eval()  # Set to evaluation mode (no gradients are calculated)
        val_loss = 0.0
        with torch.no_grad():  # Disable gradient computation for validation
            for data in tqdm(val_dataloader):
                features = data["features"]
                outputs = model(features)
                loss = criterion(outputs, features)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_dataloader)
        print(
            f"Epoch [{epoch+1}/{epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}"
        )

        if avg_val_loss < best_loss:
            best_loss = avg_val_loss

            encoder = model.encoder
            decoder = model.decoder
            torch.save(encoder.state_dict(), "save_models/encoder.pth")
            torch.save(decoder.state_dict(), "save_models/decoder.pth")
            torch.save(model.state_dict(), "save_models/model.pth")

            print("new model saved")


dataset_name = "micpst/can"
split = "train"

full_dataset = load_dataset(dataset_name)
train_val_split = full_dataset["train"].train_test_split(test_size=0.2)
train_dataset = train_val_split["train"]
val_dataset = train_val_split["test"]

train_dataset = train_dataset.map(lambda raw_data: process_features(raw_data))
train_dataset = train_dataset.with_format("torch")

val_dataset = val_dataset.map(lambda raw_data: process_features(raw_data))
val_dataset = val_dataset.with_format("torch")


batch_size = 64
train_dataloader = DataLoader(train_dataset, batch_size=batch_size)
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

criterion = nn.MSELoss()
optimizer = optim.Adam(ae_model.parameters(), lr=0.001)

train_autoencoder(
    ae_model, train_dataloader, val_dataloader, optimizer, criterion, epochs=25
)
