import os
import sys
import argparse
from tqdm import tqdm
from transformers import BertModel, BertPreTrainedModel
from tensor2tensor.data_generators import text_encoder
import torch
from torch import nn
from torch.nn import CrossEntropyLoss
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Set random seeds for reproducibility
torch.manual_seed(0)
np.random.seed(0)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define hyperparameters
batch_size = 32
dropout_rate = 0.25
bert_dim = 768
num_labels = 2  # 0: prose, 1: poetry

class BertForSequenceClassification(nn.Module):
    def __init__(self, bertPath=None, freeze_bert=False):
        super(BertForSequenceClassification, self).__init__()
        self.bert = BertModel.from_pretrained(bertPath)

        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False

        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(bert_dim, num_labels)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):
        outputs = self.bert(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, num_labels), labels.view(-1))
            return loss, logits
        else:
            return logits

def read_data(filename):
    data = pd.read_csv(filename)
    sentences = data['Sentence'].tolist()
    labels = data['Genre'].map({'prose': 0, 'poetry': 1}).tolist()
    return sentences, labels

def tokenize_sentences(sentences, tokenizer, max_length=512):
    tokenized_sentences = []
    for sentence in sentences:
        tokens = tokenizer.encode(sentence)
        tokens = tokens[:max_length - 2]  # Truncate to max_length - 2 (to account for special tokens)
        tokens = [tokenizer.vocab_size] + tokens + [tokenizer.vocab_size + 1]
        attention_mask = [1] * len(tokens)
        tokenized_sentences.append((tokens, attention_mask))
    return tokenized_sentences

def get_batches(tokenized_sentences, labels, batch_size):
    input_ids = [torch.tensor(sentence[0]) for sentence in tokenized_sentences]
    attention_masks = [torch.tensor(sentence[1]) for sentence in tokenized_sentences]
    labels = torch.tensor(labels)
    
    dataset = torch.utils.data.TensorDataset(torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True),
                                             torch.nn.utils.rnn.pad_sequence(attention_masks, batch_first=True),
                                             labels)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader

def train(model, train_dataloader, optimizer, epochs):
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for batch in tqdm(train_dataloader, desc=f"Epoch {epoch + 1}"):
            input_ids, attention_masks, labels = batch
            input_ids = input_ids.to(device)
            attention_masks = attention_masks.to(device)
            labels = labels.to(device)
            
            loss, _ = model(input_ids, attention_mask=attention_masks, labels=labels)
            train_loss += loss.item()
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        avg_train_loss = train_loss / len(train_dataloader)
        print(f"Epoch {epoch + 1} - Train Loss: {avg_train_loss:.4f}")

def evaluate(model, test_dataloader):
    model.eval()
    test_accuracy = 0
    with torch.no_grad():
        for batch in test_dataloader:
            input_ids, attention_masks, labels = batch
            input_ids = input_ids.to(device)
            attention_masks = attention_masks.to(device)
            labels = labels.to(device)

            logits = model(input_ids, attention_mask=attention_masks)
            predictions = torch.argmax(logits, dim=1)
            test_accuracy += (predictions == labels).sum().item()

    test_accuracy /= len(test_dataloader.dataset)
    print(f"Test Accuracy: {test_accuracy:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--bertPath', help='path to pre-trained BERT', required=True)
    parser.add_argument('--tokenizerPath', help='path to Latin subword tokenizer', required=True)
    parser.add_argument('--inputFile', help='CSV file with prose/poetry data', required=True)
    parser.add_argument('--epochs', type=int, default=5, help='number of training epochs')

    args = parser.parse_args()

    bertPath = args.bertPath
    tokenizerPath = args.tokenizerPath
    inputFile = args.inputFile
    epochs = args.epochs

    sentences, labels = read_data(inputFile)

    # Split data into train and test sets
    train_sentences, test_sentences, train_labels, test_labels = train_test_split(
        sentences, labels, test_size=0.9, random_state=42
    )

    # Load custom subword tokenizer
    tokenizer = text_encoder.SubwordTextEncoder(tokenizerPath)

    # Tokenize sentences
    train_tokenized_sentences = tokenize_sentences(train_sentences, tokenizer)
    test_tokenized_sentences = tokenize_sentences(test_sentences, tokenizer)

    # Create data loaders
    train_dataloader = get_batches(train_tokenized_sentences, train_labels, batch_size)
    test_dataloader = get_batches(test_tokenized_sentences, test_labels, batch_size)

    # Initialize the model
    model = BertForSequenceClassification(bertPath=bertPath, freeze_bert=False)
    model.to(device)

    # Set optimizer
    optimizer = optim.Adam(model.parameters(), lr=2e-5)

    # Train the model
    train(model, train_dataloader, optimizer, epochs)

    # Evaluate the model
    evaluate(model, test_dataloader)
