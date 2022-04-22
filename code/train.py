from preprocess import preprocess
from model import MyBertSequenceClassification
from transformers import BertConfig, BertTokenizer
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import pytorch_lightning as pl
import argparse
from tqdm import tqdm
from omegaconf import OmegaConf
from torch.utils.tensorboard import SummaryWriter


writer = SummaryWriter(log_dir="./logs")


device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
n_max_gpus = torch.cuda.device_count()
device_ids = list(range(n_max_gpus))
print(device)
print(f"{n_max_gpus} GPUs available")

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
    batch_size = config.data.batch_size
    num_workers = config.data.num_workers
    epoch = config.train.epoch

    cfg = BertConfig.from_pretrained(bert_version)
    data, label_index_dict = preprocess(debug, data_path, file_data_name, file_label_name)
    class_num = max(label_index_dict.keys())
    criterion = torch.nn.CrossEntropyLoss()
    model = MyBertSequenceClassification(cfg, class_num, criterion)
    # 並列でGPUを使うための設定
    # rank = xxx
    # model = model.to(rank)
    # model = DDP(model,  device_ids=[rank])
    model = model.to(device)
    print(model)

    train_data, test_data = train_test_split(data)
    test_data, val_data = train_test_split(test_data)
    train_dataset = ShinraDataset(train_data)
    test_dataset = ShinraDataset(test_data)
    val_dataset = ShinraDataset(val_data)
    train_dataloader = DataLoader(train_dataset, 
                                    batch_size=batch_size, 
                                    collate_fn=collate_fn,
                                    num_workers= num_workers,
                                    shuffle=True)
    val_dataloader = DataLoader(val_dataset, 
                                batch_size=batch_size, 
                                collate_fn=collate_fn, 
                                num_workers= num_workers,
                                shuffle=False)
    test_dataloader = DataLoader(test_dataset, 
                                batch_size=batch_size, 
                                collate_fn=collate_fn, 
                                num_workers= num_workers,
                                shuffle=False)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    for i in range(epoch):
        print(f"epoch: {i}")
        train_loss = train(model, train_dataloader, criterion, optimizer, i)
        val_loss = val(model, test_dataloader, criterion, i)
        writer.add_scalar("train loss", train_loss, i)
        writer.add_scalar("validation loss", val_loss, i)
    test_acc, test_loss = test(model, test_dataloader, criterion)
    writer.close()


def train(model, dataloader, criterion, optimizer, epoch):
    model.train()
    losses = []
    for text, label in tqdm(dataloader, desc=f'training epoch {epoch}', total=len(dataloader)):
        text = text.to(device)
        label = label.to(device)
        output = model(text)
        loss = criterion(output, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return sum(losses) / len(losses)


def val(model, dataloader, criterion, epoch):
    model.eval()
    losses = []
    for text, label in tqdm(dataloader, desc=f"validating epoch {epoch}", total=len(dataloader)):
        text = text.to(device)
        label = label.to(device)
        with torch.no_grad():
            output = model(text)
            loss = criterion(output, label)
            losses.append(loss.item())
    return sum(losses) / len(losses)

def test(model, dataloader, criterion):
    model.eval()
    for text, label in tqdm(dataloader, desc=f"testing model", total=len(dataloader)):
        text = text.to(device)
        label = label.to(device)
        with torch.no_grad():
            output = model(text)
        loss = criterion(output, label)
        pred = torch.argmax(output, dim=1)
        acc = torch.sum(pred == label).item() / len(pred)
    return acc, loss

if __name__ == '__main__':
    main()
