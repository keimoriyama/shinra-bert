from preprocess import preprocess
from model import  BertModelForClassification, MyBertSequenceClassification
from transformers import BertConfig
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import torch.multiprocessing as mp

import os

from sklearn.model_selection import train_test_split
import argparse
from omegaconf import OmegaConf

from tqdm import tqdm
from typing import OrderedDict

from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import MLFlowLogger


device = "cuda" if torch.cuda.is_available() else "cpu"


class ShinraDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        ids = self.data[index][0][0]
        attn_mask = self.data[index][1][0]
        token_type = self.data[index][2][0]
        label = self.data[index][3]
        return ids, attn_mask, token_type, label

    def __len__(self):
        return len(self.data)


def collate_fn(batch):
    # 入力データの辞書を作る
    texts, attn_mask, token_types, labels = list(zip(*batch))
    texts = torch.tensor(list(texts))
    attn_mask = torch.tensor(list(attn_mask))
    token_types = torch.tensor(list(token_types))
    label = torch.tensor(labels)
    inputs = {
        "input_ids": texts,
        "attention_mask": attn_mask,
        "token_type_ids": token_types,
    }
    return inputs, label


parser = argparse.ArgumentParser()
"""
def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size= world_size)
"""

def main():
    seed_everything(10)
    mlflow.start_run()

    parser.add_argument("--config_file")
    args = parser.parse_args()
    config = OmegaConf.load("./config/" + args.config_file)

    debug = config.debug
    data_path = config.data.data_path
    file_label_name = config.data.file_label_name
    file_data_name = config.data.file_data_name
    batch_size = config.data.batch_size
    data_type = config.data.data_type
    num_workers = config.data.num_workers
    epoch = config.train.epoch
    lr = config.optim.learning_rate
    rank = config.train.rank
    exp_name = config.exp_name

    num_devices = torch.cuda.device_count()
    print(f"{num_devices} GPUs available")

    bert_version = "bert-base-cased"
    mlf_logger = MLFlowLogger(experiment_name = exp_name)
    mlf_logger.log_hyperparams(config.data)
    mlf_logger.log_hyperparams(config.train)
    # mlflow.log_params(config.data)
    # mlflow.log_params(config.train)

    print("reading dataset")
    cfg = BertConfig.from_pretrained(bert_version)
    data, label_index_dict = preprocess(
        debug, data_path, file_data_name, file_label_name, bert_version, data_type
    )
    class_num = max(label_index_dict.keys()) + 1
    train_data, test_data = train_test_split(data)
    test_data, val_data = train_test_split(test_data)

    print("making dataset")
    train_dataset = ShinraDataset(train_data)
    test_dataset = ShinraDataset(test_data)
    val_dataset = ShinraDataset(val_data)

    # setup(rank, num_devices)

    # train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, num_replicas=dist.get_world_size(), rank=dist.get_rank(), shuffle=True)

    train_dataloader = DataLoader(train_dataset,
                                  batch_size=batch_size,
                                  collate_fn=collate_fn,
                                  num_workers=num_workers,
                                  )
    val_dataloader = DataLoader(val_dataset,
                                batch_size=batch_size,
                                collate_fn=collate_fn,
                                num_workers=num_workers,
                                # persistent_workers = True,
                                shuffle=False)
    test_dataloader = DataLoader(test_dataset,
                                 batch_size=batch_size,
                                 collate_fn=collate_fn,
                                 num_workers=num_workers,
                                 # persistent_workers = True,
                                 shuffle=False)
    criterion = torch.nn.CrossEntropyLoss()
    # model = MyBertSequenceClassification(cfg, class_num, criterion, lr)
    
    
    model = BertModelForClassification(cfg, class_num)
    # model = DDP(model, device_ids=[rank])
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    train_loss = train(model, i, train_dataloader, optimizer, criterion, config)
    valid_acc, valid_loss = validate(model, i, val_dataloader, criterion, config)
    mlflow.log_metric(key="train loss", value=train_loss, step=i + 1)
    mlflow.log_metric(key="validation loss", value=valid_loss, step=i + 1)
    mlflow.log_metric(key="validation accuracy", value=valid_acc, step=i + 1)

    test(model, test_dataloader, criterion)
    mlflow.end_run()
   
def train(model, epoch, dataloader,optimizer, criterion, cfg):
    rank = cfg.train.rank
    with tqdm(dataloader) as pbar:
        pbar.set_description(f"Train [Epoch {epoch + 1}/{cfg.train.epoch}")
        loss_mean = 0
        for text, label in pbar:
            # import ipdb;ipdb.set_trace()
            text["input_ids"] = text["input_ids"].to(rank)
            text["attention_mask"] = text["attention_mask"].to(rank)
            text["token_type_ids"] = text["token_type_ids"].to(rank)
            label = label.to(rank)
            out = model(text)
            loss = criterion(out, label)
            loss_mean += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pbar.set_postfix(OrderedDict(Loss=loss.item()))
        loss_mean = loss_mean / len(dataloader)
        return loss_mean


def validate(model, epoch, dataloader, criterion, cfg):
    rank = cfg.train.rank
    with tqdm(dataloader) as pbar:
        pbar.set_description(f"Validation [Epoch {epoch + 1}/{cfg.train.epoch}")
        acc, loss_mean = 0, 0
        for text, label in dataloader:
            # import ipdb;ipdb.set_trace()
            text["input_ids"] = text["input_ids"].to(device)
            text["attention_mask"] = text["attention_mask"].to(device)
            text["token_type_ids"] = text["token_type_ids"].to(device)
            label = label.to(device)
            with torch.no_grad():
                out = model(text)
                loss = criterion(out, label)
            pred = out.argmax(axis=1)
            acc += torch.sum(pred == label).item() / len(label)
            loss_mean += loss.item()
            pbar.set_postfix(OrderedDict(Loss=loss_mean, Acc=acc))
    loss_mean = loss_mean / len(dataloader)
    return acc, loss_mean


def test(model, dataloader, criterion):
    with tqdm(dataloader) as pbar:
        pbar.set_description(f"Testing")
        acc, loss_mean = 0, 0
        for text, label in dataloader:
            text["input_ids"] = text["input_ids"].to(rank)
            text["attention_mask"] = text["attention_mask"].to(device)
            text["token_type_ids"] = text["token_type_ids"].to(device)
            label = label.to(device)
            with torch.no_grad():
                out = model(text)
                loss = criterion(out, label)
            pred = out.argmax(axis=1)
            acc += torch.sum(pred == label).item() / len(label)
            loss_mean += loss.item()
            pbar.set_postfix(OrderedDict(Loss=loss_mean, Acc=acc))
    loss_mean = loss_mean / len(dataloader)
    return acc, loss_mean


if __name__ == "__main__":
    main()
