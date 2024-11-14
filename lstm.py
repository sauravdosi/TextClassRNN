import json
import torch
import torch.nn as nn
import torch.optim as optim
import random
import time
import itertools
from argparse import ArgumentParser
from tqdm import tqdm
from torch.nn.utils.rnn import pad_sequence

unk = "UNK"
# Custom LSTM model with an embedding layer
class LSTMModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super(LSTMModel, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        embedded = self.embedding(x)
        _, (hidden, _) = self.lstm(embedded)
        output = self.fc(hidden[-1])
        return output


def load_data(train_data, val_data):
    with open(train_data) as training_f:
        training = json.load(training_f)
    with open(val_data) as valid_f:
        validation = json.load(valid_f)

    tra = []
    val = []
    for elt in training:
        tra.append((elt["text"].split(), int(elt["stars"] - 1)))
    for elt in validation:
        val.append((elt["text"].split(), int(elt["stars"] - 1)))

    return tra, val

def make_vocab(data):
    vocab = set()
    for text, _ in data:
        vocab.update(text)
    return vocab

def make_indices(vocab):
    word2index = {word: i for i, word in enumerate(vocab)}
    index2word = {i: word for word, i in word2index.items()}
    return vocab, word2index, index2word

def convert_to_vector_representation(data, word2index):
    vectorized_data = []
    for text, label in data:
        indices = [word2index[word] for word in text if word in word2index]
        vectorized_data.append((torch.tensor(indices), label))
    return vectorized_data


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--train_data", required=True, help="path to training data")
    parser.add_argument("--val_data", required=True, help="path to validation data")
    args = parser.parse_args()

    # Fix random seeds
    random.seed(42)
    torch.manual_seed(42)

    # Load data
    print("========== Loading data ==========")
    train_data, valid_data = load_data(args.train_data, args.val_data)
    vocab = make_vocab(train_data)
    vocab, word2index, index2word = make_indices(vocab)

    print("========== Vectorizing data ==========")
    train_data = convert_to_vector_representation(train_data, word2index)
    valid_data = convert_to_vector_representation(valid_data, word2index)

    # Hyperparameters
    embedding_dim = 50
    hidden_dim = 128
    output_dim = 5  # Assuming 5 possible star ratings
    epochs = 20
    batch_size = 512
    learning_rate = 0.05
    momentum = 0.9  # Only relevant if using SGD with momentum

    # Initialize model, loss function, and optimizer
    model = LSTMModel(vocab_size=len(vocab), embedding_dim=embedding_dim, hidden_dim=hidden_dim, output_dim=output_dim)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training and evaluation loops
    for epoch in range(epochs):
        # Training
        model.train()
        random.shuffle(train_data)
        total_loss, correct, total = 0, 0, 0
        for i in range(0, len(train_data), batch_size):
            batch = train_data[i:i + batch_size]
            texts, labels = zip(*batch)
            texts_padded = pad_sequence(texts, batch_first=True)
            labels = torch.tensor(labels)

            optimizer.zero_grad()
            outputs = model(texts_padded)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            correct += (outputs.argmax(dim=1) == labels).sum().item()
            total += len(labels)

        print(
            f"Epoch {epoch + 1}/{epochs}, Training Loss: {total_loss / len(train_data):.4f}, Training Accuracy: {correct / total:.4f}")

        # Validation
        model.eval()
        total_loss, correct, total = 0, 0, 0
        with torch.no_grad():
            for i in range(0, len(valid_data), batch_size):
                batch = valid_data[i:i + batch_size]
                texts, labels = zip(*batch)
                texts_padded = pad_sequence(texts, batch_first=True)
                labels = torch.tensor(labels)

                outputs = model(texts_padded)
                loss = criterion(outputs, labels)
                total_loss += loss.item()
                correct += (outputs.argmax(dim=1) == labels).sum().item()
                total += len(labels)

        print(f"Epoch {epoch + 1}/{epochs}, Validation Loss: {total_loss / len(valid_data):.4f}, Validation Accuracy: {correct / total:.4f}")
