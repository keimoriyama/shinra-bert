from preprocess import preprocess
from model import MyBertSequenceClassification
from transformers import BertConfig, BertTokenizer
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import pytorch_lightning as pl
import argparse
from omegaconf import OmegaConf

bert_version = "bert-base-cased"
tokenizer = BertTokenizer.from_pretrained(bert_version)


def tokenize_text(text):
    return tokenizer(text,
                     return_tensors='pt',
                     padding="max_length",
                     truncation=True)


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


parser = argparse.ArgumentParser()


def main():
    parser.add_argument("--config_file")
    args = parser.parse_args()
    config = OmegaConf.load("./config/" + args.config_file)
    debug = config.debug
    data_path = config.data.data_path
    file_label_name = config.data.file_label_name
    file_data_name = config.data.file_data_name

    cfg = BertConfig.from_pretrained(bert_version)
    data, label_index_dict = preprocess(debug, data_path, file_data_name, file_label_name)
    class_num = max(label_index_dict.keys())
    criterion = torch.nn.CrossEntropyLoss()
    model = MyBertSequenceClassification(cfg, class_num, criterion)
    train_data, test_data = train_test_split(data)
    test_data, val_data = train_test_split(test_data)
    train_dataset = ShinraDataset(train_data)
    test_dataset = ShinraDataset(test_data)
    val_dataset = ShinraDataset(val_data)
    train_dataloader = DataLoader(train_dataset, 
                                    batch_size=batch_size, 
                                    collate_fn=collate_fn, 
                                    shuffle=True)
    val_dataloader = DataLoader(val_dataset, 
                                batch_size=batch_size, 
                                collate_fn=collate_fn, 
                                shuffle=True)
    test_dataloader = DataLoader(test_dataset, 
                                batch_size=batch_size, 
                                collate_fn=collate_fn, 
                                shuffle=False)
    trainer = pl.Trainer(max_epochs=10)
    trainer.fit(model, train_dataloader, val_dataloader)
    trainer.test(model, dataloaders=test_dataloader)
    """
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
"""


if __name__ == '__main__':
    main()
