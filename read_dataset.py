from torch.utils.data import Dataset
from datasets import load_dataset
import numpy as np
import torch
from sklearn.preprocessing import OneHotEncoder


class IntrusionDetection(Dataset):
    def __init__(self, dataset_name, split):
        full_dataset = load_dataset(dataset_name)
        self.dataset = full_dataset[split]

        # Define the hex character set
        # hex_characters = "0123456789ABCDEF"

        # Create a mapping from hex characters to integers (tokenization)
        # self.hex_to_int = {char: i for i, char in enumerate(hex_characters)}
        self.onehot_encoder = OneHotEncoder(sparse=False)

    def tokenize_hex(self, hex_features):
        return [[self.hex_to_int[char] for char in hex_str] for hex_str in hex_features]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        hex_features = self.dataset["features"][idx]
        hex_features = ["350", "2C0", "430", "4B1", "1F1", "153", "002"]
        # hex_features = np.array(hex_features).reshape(-1, 1)
        # onehot_encoded = self.onehot_encoder.fit_transform(hex_features)

        # tokenized_features = self.tokenize_hex(hex_features)
        # tokenized_tensor = torch.tensor(tokenized_features, dtype=torch.long)
        return torch.zeros(5), self.dataset["label"][idx]
        # int_values = np.array([int(h, 16) for h in hex_features])
        # binary_rep = np.array([list(bin(x)[2:].zfill(12)) for x in int_values])
        # features = binary_rep.astype(int)
        # return torch.tensor(features), self.dataset["label"][idx]
