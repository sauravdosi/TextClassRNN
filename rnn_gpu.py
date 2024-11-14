import torch
import torch.nn as nn
import torch.optim as optim
import pickle
from tqdm import tqdm
import string
import json
import random
import time
import numpy as np
import pandas as pd

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


def load_data(train_data, val_data):
    with open(train_data) as training_f:
        training = json.load(training_f)
    with open(val_data) as valid_f:
        validation = json.load(valid_f)

    tra = []
    val = []
    for elt in training:
        tra.append((elt["text"].split(),int(elt["stars"]-1)))
    for elt in validation:
        val.append((elt["text"].split(),int(elt["stars"]-1)))
    return tra, val


class RNN(nn.Module):
    def __init__(self, input_dim, h):
        super(RNN, self).__init__()
        self.h = h
        self.numOfLayer = 1
        self.rnn = nn.RNN(input_dim, h, self.numOfLayer, nonlinearity='tanh').to(device)
        self.W = nn.Linear(h, 5).to(device)
        self.softmax = nn.LogSoftmax(dim=1)
        self.loss = nn.NLLLoss()

    def compute_Loss(self, predicted_vector, gold_labels):
        return self.loss(predicted_vector, gold_labels)

    def forward(self, inputs):
        seq_len, batch_size, _ = inputs.size()
        h_t_minus_1 = torch.zeros(self.numOfLayer, batch_size, self.h, device=device)
        output_matrix = torch.zeros(seq_len, batch_size, 5, device=device)

        for i in range(seq_len):
            _, h_t = self.rnn(inputs[i].unsqueeze(0), h_t_minus_1)
            output_matrix[i] = self.W(h_t.squeeze(0))
            h_t_minus_1 = h_t

        output_sum = output_matrix.sum(dim=0)
        predicted_vector = self.softmax(output_sum)
        return predicted_vector


def get_correct_samples(predictions, labels):
    _, predicted_classes = torch.max(predictions, dim=1)
    # print("PREDICTIONS")
    # print(predicted_classes)
    # print("LABELS")
    # print(labels)
    # time.sleep(10)
    correct = (predicted_classes == labels).sum().item()
    # accuracy = correct / labels.size(0)
    return correct


if __name__ == "__main__":
    args = {"hidden_dim": 20, "epochs": 30}
    hidden_dim = args["hidden_dim"]
    epochs = args["epochs"]

    # Load your data here
    train_data, valid_data = load_data("./training.json", "./validation.json")
    model = RNN(200, hidden_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.05)
    # word_embedding = pickle.load(open('./word_embedding.pkl', 'rb'))
    #
    # glove_file = "./glove.840B.300d.txt"
    glove_file = "./glove.twitter.27B.200d.txt"

    # Initialize an empty dictionary to store the embeddings
    word_embedding = {}

    # Open and read the GloVe file
    with open(glove_file, 'r', encoding='utf-8') as f:
        for line in f:
            # Strip any trailing whitespace/newlines and split by spaces from the end
            line = line.strip()
            # Split the line into parts, keeping only the last 300 (or 100, 200 depending on GloVe dimension) as the vector
            parts = line.rsplit(' ', 200)  # Adjust the number to match your vector dimension if necessary
            word = ' '.join(parts[:-200])  # Join everything before the last 300 values as the word
            vector = np.array(parts[-200:], dtype='float32')  # Convert the last 300 elements to float32

            # Add to dictionary
            word_embedding[word] = vector

    minibatch_size = 512
    N = len(train_data)
    results = []

    for epoch in range(epochs):
        random.shuffle(train_data)
        model.train()
        total_train_loss = 0
        total_train_correct = 0

        # Training loop
        for minibatch_index in tqdm(range(N // minibatch_size)):
            optimizer.zero_grad()
            batch_inputs = []
            batch_labels = []

            for example_index in range(minibatch_size):
                input_words, gold_label = train_data[minibatch_index * minibatch_size + example_index]
                input_words = " ".join(input_words).translate(str.maketrans("", "", string.punctuation)).split()
                vectors = [word_embedding[i.lower()] if i.lower() in word_embedding.keys() else word_embedding['unk']
                           for i
                           in input_words]
                batch_inputs.append(torch.tensor(vectors, device=device))
                batch_labels.append(gold_label)

            # Pad sequences in the batch to the same length and stack them
            batch_inputs = nn.utils.rnn.pad_sequence(batch_inputs, batch_first=False)
            batch_labels = torch.tensor(batch_labels, device=device)

            # Forward pass
            output = model(batch_inputs)
            # print(output.shape)
            # time.sleep(10)
            loss = model.compute_Loss(output, batch_labels)

            # Backward pass
            loss.backward()
            optimizer.step()

            # Accumulate loss and accuracy
            total_train_loss += loss.item()
            total_train_correct += get_correct_samples(output, batch_labels)
            # break

        avg_train_loss = total_train_loss / (N // minibatch_size)
        avg_train_accuracy = total_train_correct / N

        # Validation loop
        model.eval()
        total_val_loss = 0
        total_val_correct = 0
        val_batch_inputs = []
        val_batch_labels = []
        with torch.no_grad():
            random.shuffle(valid_data)
            for val_input_words, val_gold_label in valid_data:
                # input_words, gold_label = train_data[minibatch_index * minibatch_size + example_index]
                # input_words = " ".join(input_words).translate(str.maketrans("", "", string.punctuation)).split()
                # vectors = [word_embedding[i.lower()] if i.lower() in word_embedding.keys() else word_embedding['unk']
                #            for i
                #            in input_words]

                val_input_words = " ".join(val_input_words).translate(str.maketrans("", "", string.punctuation)).split()
                val_vectors = [
                    word_embedding[i.lower()] if i.lower() in word_embedding.keys() else word_embedding['unk'] for i
                    in val_input_words]
                val_batch_inputs.append(torch.tensor(val_vectors, device=device))
                val_batch_labels.append(val_gold_label)

            # print(val_vectors)

                # Pad sequences in the batch to the same length and stack them
            val_batch_inputs = nn.utils.rnn.pad_sequence(val_batch_inputs, batch_first=False)
            val_batch_labels = torch.tensor(val_batch_labels, device=device)
            # print(val_batch_inputs)
            # val_input = nn.utils.rnn.pad_sequence([torch.tensor(val_vectors, device=device)], batch_first=False)

            # Forward pass
            val_output = model(val_batch_inputs)
            val_loss = model.compute_Loss(val_output, val_batch_labels)

            # Accumulate loss and accuracy
            total_val_loss += val_loss.item()
            total_val_correct += get_correct_samples(val_output, val_batch_labels)

        # print("VAL PREDICTIONS")
        # _, predicted_classes = torch.max(val_output, dim=1)
        # print(predicted_classes)
        # print("VAL GOLD LABELS")
        # print(val_batch_labels)
        avg_val_loss = total_val_loss
        avg_val_accuracy = total_val_correct / len(valid_data)

        # Print epoch results
        print(f"Epoch {epoch + 1}: Train Loss = {avg_train_loss:.4f}, Train Accuracy = {avg_train_accuracy:.4f}")
        print(f"             Validation Loss = {avg_val_loss:.4f}, Validation Accuracy = {avg_val_accuracy:.4f}")

        # Save model after each epoch
        # torch.save(model.state_dict(), f"model_glove_h20_batch512_epoch_{epoch + 1}.pth")
        results.append({
            "epoch": epoch + 1,
            "loss": avg_train_loss,
            "train_accuracy": avg_train_accuracy,
            "val_accuracy": avg_val_accuracy})

        pd.DataFrame(results).to_csv("./best_rnn_glove_twitter_results.csv", index=False)

