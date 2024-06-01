import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.nn import functional as F
from einops import rearrange
from tqdm import tqdm

import math
import os
import urllib.request
from zipfile import ZipFile

from transformers import AutoTokenizer

from mamba.model import Mamba

torch.autograd.set_detect_anomaly(True)

# Configuration flags and hyperparameters
USE_MAMBA = 1
DIFFERENT_H_STATES_RECURRENT_UPDATE_MECHANISM = 0

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

d_model = 8
state_size = 128  # Example state size
seq_len = 100  # Example sequence length
batch_size = 256  # Example batch size
last_batch_size = 81  # only for the very last batch of the dataset
current_batch_size = batch_size
different_batch_size = False
h_new = None
temp_buffer = None


class Enwiki8Dataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data['input_ids'])

    def __getitem__(self, idx):
        item = {key: val[idx].clone().detach() for key, val in self.data.items()}
        return item


# Define a function for padding
def pad_sequences_3d(sequences, max_len=None, pad_value=0):
    # Assuming sequences is a tensor of shape (batch_size, seq_len, feature_size)
    batch_size, seq_len, feature_size = sequences.shape

    if max_len is None:
        max_len = seq_len + 1

    # Initialize padded_sequences with the pad_value
    padded_sequences = torch.full((batch_size, max_len, feature_size), fill_value=pad_value, dtype=sequences.dtype,
                                  device=sequences.device)
    # Pad each sequence to the max_len
    padded_sequences[:, :seq_len, :] = sequences

    return padded_sequences


def train(model, tokenizer, data_loader, optimizer, criterion, device, max_grad_norm=1.0, DEBUGGING_IS_ON=False):
    model.train()
    total_loss = 0
    for batch in data_loader:
        optimizer.zero_grad()

        input_data = batch['input_ids'].clone().to(device)
        attention_mask = batch['attention_mask'].clone().to(device)

        target = input_data[:, 1:]
        input_data = input_data[:, :-1]

        # Pad all the sequences in the batch:
        input_data = pad_sequences_3d(input_data, pad_value=tokenizer.pad_token_id)
        target = pad_sequences_3d(target, max_len=input_data.size(1), pad_value=tokenizer.pad_token_id)

        if USE_MAMBA:
            output = model(input_data)
            loss = criterion(output, target)

        loss.backward(retain_graph=True)

        for name, param in model.named_parameters():
            if 'out_proj.bias' not in name:
                # clip weights but not bias for out_proj
                torch.nn.utils.clip_grad_norm_(param, max_norm=max_grad_norm)

        if DEBUGGING_IS_ON:
            for name, parameter in model.named_parameters():
                if parameter.grad is not None:
                    print(f"{name} gradient: {parameter.grad.data.norm(2)}")
                else:
                    print(f"{name} has no gradient")

        if USE_MAMBA and DIFFERENT_H_STATES_RECURRENT_UPDATE_MECHANISM:
            model.S6.h[:current_batch_size, ...].copy_(temp_buffer)

        optimizer.step()

        total_loss += loss.item()
    return total_loss / len(data_loader)


def evaluate(model, data_loader, criterion, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in data_loader:
            input_data = batch['input_ids'].clone().detach().to(device)
            attention_mask = batch['attention_mask'].clone().detach().to(device)

            target = input_data[:, 1:]
            input_data = input_data[:, :-1]

            # Pad all the sequences in the batch:
            input_data = pad_sequences_3d(input_data, pad_value=tokenizer.pad_token_id)
            target = pad_sequences_3d(target, max_len=input_data.size(1), pad_value=tokenizer.pad_token_id)

            if USE_MAMBA:
                output = model(input_data)
                loss = criterion(output, target)
            total_loss += loss.item()
    return total_loss / len(data_loader)


def calculate_perplexity(loss):
    return math.exp(loss)


def load_enwiki8_dataset():
    print(f"Download and extract enwiki8 data")
    url = "http://mattmahoney.net/dc/enwik8.zip"
    urllib.request.urlretrieve(url, "enwik8.zip")

    with ZipFile("enwik8.zip") as f:
        data = f.read("enwik8").decode("utf-8")

    return data


# Tokenize and encode the dataset
def encode_dataset(tokenizer, text_data):
    def batch_encode(tokenizer, text_data, batch_size=1000):
        # Tokenize in batches
        batched_input_ids = []
        for i in range(0, len(text_data), batch_size):
            batch = text_data[i:i + batch_size]
            inputs = tokenizer(batch, add_special_tokens=True, truncation=True,
                               padding='max_length', max_length=seq_len,
                               return_tensors='pt')
            batched_input_ids.append(inputs['input_ids'])
        return torch.cat(batched_input_ids)

    # Assuming enwiki8_data is a list of sentences
    input_ids = batch_encode(tokenizer, enwiki8_data)

    # vocab_size is the number of unique tokens in the tokenizer's vocabulary
    global vocab_size
    vocab_size = len(tokenizer.vocab)  # Note that for some tokenizers, we might access the vocab directly
    print(f"vocab_size = {vocab_size}")

    # Create an embedding layer
    # embedding_dim is the size of the embedding vectors (MAMBA model's D)
    embedding_layer = nn.Embedding(num_embeddings=vocab_size, embedding_dim=d_model)

    # Pass `input_ids` through the embedding layer
    # This will change `input_ids` from shape [B, L] to [B, L, D]
    def batch_embedding_calls(input_ids, embedding_layer, batch_size=256):
        # Check if input_ids is already a tensor, if not convert it
        if not isinstance(input_ids, torch.Tensor):
            input_ids = torch.tensor(input_ids, dtype=torch.long)

        # Calculate the number of batches needed
        num_batches = math.ceil(input_ids.size(0) / batch_size)

        # List to hold the output embeddings
        output_embeddings = []

        # Process each batch
        for i in range(num_batches):
            # Calculate start and end indices for the current batch
            start_idx = i * batch_size
            end_idx = start_idx + batch_size

            # Get the batch
            input_id_batch = input_ids[start_idx:end_idx]

            # Call the embedding layer
            with torch.no_grad():  # No need gradients for this operation
                batch_embeddings = embedding_layer(input_id_batch)

            # Append the result to the list
            output_embeddings.append(batch_embeddings)

        # Concatenate the embeddings from each batch into a single tensor
        all_embeddings = torch.cat(output_embeddings, dim=0)

        return all_embeddings

    # `input_ids` is a list or tensor of the input IDs and `embedding_layer` is model's embedding layer
    if USE_MAMBA:
        # Set `batch_size` to a value that works for memory constraints
        encoded_inputs = batch_embedding_calls(input_ids, embedding_layer, batch_size=1).float()

    attention_mask = (input_ids != tokenizer.pad_token_id).type(input_ids.dtype)

    return encoded_inputs, attention_mask


# Load a pretrained tokenizer
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
# Assuming encoded_inputs is a preprocessed tensor of shape [num_samples, seq_len, d_model]
encoded_inputs_file = 'encoded_inputs_mamba.pt'

if os.path.exists(encoded_inputs_file):
    print("Loading pre-tokenized data...")
    encoded_inputs = torch.load(encoded_inputs_file)
else:
    print("Tokenizing raw data...")
    enwiki8_data = load_enwiki8_dataset()
    encoded_inputs, attention_mask = encode_dataset(tokenizer, enwiki8_data)
    torch.save(encoded_inputs, encoded_inputs_file)
    print(f"finished tokenizing data")

# Combine into a single dictionary
data = {
    'input_ids': encoded_inputs,
    'attention_mask': attention_mask
}

# Split the data into train and validation sets
total_size = len(data['input_ids'])
train_size = int(total_size * 0.8)

train_data = {key: val[:train_size] for key, val in data.items()}
val_data = {key: val[train_size:] for key, val in data.items()}

train_dataset = Enwiki8Dataset(train_data)
val_dataset = Enwiki8Dataset(val_data)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Initialize the model

model = Mamba(seq_len, d_model, state_size, device).to(device)

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=5e-6)

# Training loop
num_epochs = 25  # Number of epochs to train for

for epoch in tqdm(range(num_epochs)):  # loop over the dataset multiple times
    train_loss = train(model, tokenizer, train_loader, optimizer, criterion, device, max_grad_norm=10.0,
                       DEBUGGING_IS_ON=False)
    val_loss = evaluate(model, val_loader, criterion, device)
    val_perplexity = calculate_perplexity(val_loss)
    print(
        f'Epoch: {epoch + 1}, Training Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}, Validation Perplexity: {val_perplexity:.4f}')
