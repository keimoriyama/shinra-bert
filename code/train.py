from preprocess import preprocess
from model import MyBertSequenceClassification
from transformers import BertModel, BertConfig, BertTokenizer
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import argparse

bert_version = "bert-base-cased"
tokenizer = BertTokenizer.from_pretrained(bert_version)



def tokenize_text(text):
    return tokenizer(text, return_tensors='pt', padding=True)


class ShinraDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        text = self.data[index][0]
        label = self.data[index][1]
        return text, label

    def __len__(self):
        return len(self.data)


def collate_fn(batch):
    texts, labels = list(zip(*batch))
    texts = list(texts)
    label = torch.tensor(labels)
    tokenized_texts = tokenize_text(texts)
    return tokenized_texts, label


def main():
    # 
    cfg = BertConfig.from_pretrained(bert_version)
    bert = BertModel(cfg)
    data, label_index_dict = preprocess(debug)
    # ここをラベルの数に変える
    class_num = max(label_index_dict.keys())
    model = MyBertSequenceClassification(bert, class_num)
    train_data, test_data = train_test_split(data)
    train_dataset = ShinraDataset(train_data)
    test_dataset = ShinraDataset(test_data)
    train_dataloader = DataLoader(train_dataset, batch_size=2, collate_fn=collate_fn, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=64, collate_fn=collate_fn, shuffle=False)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    epoch = 10
    train_losses, val_losses = [], []
    for i in range(epoch):
        print(f"epoch: {i}")
        train_loss = train(model, train_dataloader, criterion, optimizer, i)
        val_loss = val(model, test_dataloader, criterion, i)
        train_losses.append(train_loss)
        val_losses.append(val_loss)


def train(model, dataloader, criterion, optimizer, epoch):
    losses = []
    for text, label in tqdm(dataloader, desc=f'training epoch {epoch}', total=len(dataloader)):
        output = model(text)
        loss = criterion(output, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return sum(losses) / len(losses)


def val(model, dataloader, criterion, epoch):
    losses = []
    for text, label in tqdm(dataloader, desc=f"validating epoch {epoch}", total=len(dataloader)):
        with torch.no_grad():
            output = model(text)
            loss = criterion(output, label)
            losses.append(loss.item())
    return sum(loss) / len(loss)


if __name__ == '__main__':
    main()
